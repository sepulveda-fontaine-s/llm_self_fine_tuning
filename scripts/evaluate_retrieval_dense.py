import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


INPUT_PATH = Path(
    "data/processed/retrieval_benchmark.json"
)

OUTPUT_PATH = Path(
    "results/evaluation/retrieval_dense_e5_small_v2.json"
)

MODEL_NAME = "intfloat/e5-small-v2"

DOCUMENT_BATCH_SIZE = 128
QUERY_BATCH_SIZE = 128
SEARCH_BATCH_SIZE = 128

MAX_LENGTH = 512

K_VALUES = (1, 3, 5, 10)


def average_pool(
    last_hidden_states,
    attention_mask,
):
    last_hidden = (
        last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(),
            0.0,
        )
    )

    return (
        last_hidden.sum(dim=1)
        / attention_mask.sum(
            dim=1
        )[..., None]
    )


def encode_texts(
    texts,
    prefix,
    tokenizer,
    model,
    device,
    batch_size,
):
    all_embeddings = []

    start_time = time.perf_counter()

    for start in range(
        0,
        len(texts),
        batch_size,
    ):
        batch_texts = [
            prefix + text
            for text in texts[
                start:start + batch_size
            ]
        ]

        inputs = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(
                    **inputs
                )

                embeddings = average_pool(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                )

                embeddings = F.normalize(
                    embeddings.float(),
                    p=2,
                    dim=1,
                )

        all_embeddings.append(
            embeddings.cpu()
        )

        processed = min(
            start + batch_size,
            len(texts),
        )

        if (
            processed % 1000 < batch_size
            or processed == len(texts)
        ):
            print(
                f"encoded "
                f"{processed}/"
                f"{len(texts)}"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return (
        torch.cat(
            all_embeddings,
            dim=0,
        ),
        elapsed,
    )


def empty_accumulator():
    return {
        "count": 0,
        "reciprocal_rank_sum": 0.0,
        "ranks": [],
        "hits": {
            k: 0
            for k in K_VALUES
        },
    }


def update_metrics(
    accumulator,
    rank,
):
    accumulator["count"] += 1

    accumulator[
        "reciprocal_rank_sum"
    ] += 1.0 / rank

    accumulator["ranks"].append(
        rank
    )

    for k in K_VALUES:
        if rank <= k:
            accumulator[
                "hits"
            ][k] += 1


def finalize_metrics(accumulator):
    count = accumulator["count"]

    ranks = accumulator["ranks"]

    return {
        "queries":
            count,

        **{
            f"recall_at_{k}":
                accumulator["hits"][k]
                / count
            for k in K_VALUES
        },

        "mrr":
            accumulator[
                "reciprocal_rank_sum"
            ]
            / count,

        "mean_gold_rank":
            statistics.mean(ranks),

        "median_gold_rank":
            statistics.median(ranks),
    }


def percentile(
    values,
    fraction,
):
    ordered = sorted(values)

    index = int(
        fraction
        * (len(ordered) - 1)
    )

    return ordered[index]


def main():
    print(
        "=== DENSE RETRIEVAL EVALUATION ==="
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    with INPUT_PATH.open(
        encoding="utf-8"
    ) as file:
        benchmark = json.load(file)

    documents = benchmark["documents"]
    queries = benchmark["queries"]

    print(
        "model:",
        MODEL_NAME,
    )

    print(
        "documents:",
        len(documents),
    )

    print(
        "queries:",
        len(queries),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    device = torch.device("cuda")

    model.to(device)
    model.eval()

    document_texts = [
        document["text"]
        for document in documents
    ]

    query_texts = [
        query["question"]
        for query in queries
    ]

    print(
        "\n=== ENCODE DOCUMENTS ==="
    )

    document_embeddings, document_encoding_seconds = (
        encode_texts(
            texts=document_texts,
            prefix="passage: ",
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=DOCUMENT_BATCH_SIZE,
        )
    )

    print(
        "\n=== ENCODE QUERIES ==="
    )

    query_embeddings, query_encoding_seconds = (
        encode_texts(
            texts=query_texts,
            prefix="query: ",
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=QUERY_BATCH_SIZE,
        )
    )

    document_embeddings = (
        document_embeddings.to(device)
    )

    document_id_to_index = {
        document["document_id"]: index
        for index, document
        in enumerate(documents)
    }

    overall = empty_accumulator()
    answerable = empty_accumulator()
    unanswerable = empty_accumulator()

    results = []
    search_times = []

    print(
        "\n=== SEARCH ==="
    )

    for start in range(
        0,
        len(queries),
        SEARCH_BATCH_SIZE,
    ):
        end = min(
            start + SEARCH_BATCH_SIZE,
            len(queries),
        )

        query_batch = (
            query_embeddings[
                start:end
            ].to(device)
        )

        search_start = time.perf_counter()

        similarities = (
            query_batch
            @ document_embeddings.T
        )

        top_scores, top_indices = (
            torch.topk(
                similarities,
                k=10,
                dim=1,
            )
        )

        search_elapsed = (
            time.perf_counter()
            - search_start
        )

        per_query_search_time = (
            search_elapsed
            / (end - start)
        )

        search_times.extend(
            [
                per_query_search_time
                for _ in range(
                    end - start
                )
            ]
        )

        similarities_cpu = (
            similarities.cpu()
        )

        top_scores = top_scores.cpu()
        top_indices = top_indices.cpu()

        for local_index, query_index in enumerate(
            range(start, end)
        ):
            query = queries[
                query_index
            ]

            gold_index = (
                document_id_to_index[
                    query[
                        "gold_document_id"
                    ]
                ]
            )

            gold_score = (
                similarities_cpu[
                    local_index,
                    gold_index,
                ].item()
            )

            row_scores = (
                similarities_cpu[
                    local_index
                ]
            )

            higher = int(
                (
                    row_scores
                    > gold_score
                ).sum().item()
            )

            equal_before = int(
                (
                    (
                        row_scores[
                            :gold_index
                        ]
                        == gold_score
                    )
                ).sum().item()
            )

            gold_rank = (
                1
                + higher
                + equal_before
            )

            update_metrics(
                overall,
                gold_rank,
            )

            if query["answerable"]:
                update_metrics(
                    answerable,
                    gold_rank,
                )
            else:
                update_metrics(
                    unanswerable,
                    gold_rank,
                )

            results.append(
                {
                    "query_id":
                        query["query_id"],
                    "answerable":
                        query[
                            "answerable"
                        ],
                    "gold_document_id":
                        query[
                            "gold_document_id"
                        ],
                    "gold_rank":
                        gold_rank,
                    "top_10": [
                        {
                            "rank":
                                rank + 1,
                            "document_id":
                                documents[
                                    int(
                                        top_indices[
                                            local_index,
                                            rank,
                                        ].item()
                                    )
                                ][
                                    "document_id"
                                ],
                            "score":
                                float(
                                    top_scores[
                                        local_index,
                                        rank,
                                    ].item()
                                ),
                        }
                        for rank in range(10)
                    ],
                }
            )

        print(
            f"searched "
            f"{end}/"
            f"{len(queries)}"
        )

        del similarities
        del similarities_cpu
        del query_batch

    total_search_seconds = sum(
        search_times
    )

    summary = {
        "method":
            "dense_retrieval",
        "model":
            MODEL_NAME,
        "pooling":
            "attention-mask mean pooling",
        "normalization":
            "L2",
        "query_prefix":
            "query: ",
        "passage_prefix":
            "passage: ",
        "corpus_documents":
            len(documents),

        "overall":
            finalize_metrics(
                overall
            ),

        "answerable":
            finalize_metrics(
                answerable
            ),

        "unanswerable":
            finalize_metrics(
                unanswerable
            ),

        "performance": {
            "document_encoding_seconds":
                document_encoding_seconds,

            "query_encoding_seconds":
                query_encoding_seconds,

            "search_seconds":
                total_search_seconds,

            "mean_search_latency_ms":
                statistics.mean(
                    search_times
                )
                * 1000.0,

            "p95_search_latency_ms":
                percentile(
                    search_times,
                    0.95,
                )
                * 1000.0,

            "search_queries_per_second":
                len(queries)
                / total_search_seconds,
        },
    }

    output = {
        "summary":
            summary,
        "results":
            results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n=== DENSE RETRIEVAL SUMMARY ==="
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nsaved:",
        OUTPUT_PATH,
    )

    print(
        "\n=== COMPLETE ==="
    )


if __name__ == "__main__":
    main()