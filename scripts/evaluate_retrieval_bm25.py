import heapq
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/retrieval_benchmark.json"
)

OUTPUT_PATH = Path(
    "results/evaluation/retrieval_bm25.json"
)

K_VALUES = (1, 3, 5, 10)

K1 = 1.5
B = 0.75

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text):
    return TOKEN_PATTERN.findall(
        text.lower()
    )


def build_index(documents):
    print("=== BUILD BM25 INDEX ===")

    document_ids = []
    document_lengths = []
    postings = defaultdict(list)

    for doc_index, document in enumerate(
        documents
    ):
        document_ids.append(
            document["document_id"]
        )

        tokens = tokenize(
            document["text"]
        )

        document_lengths.append(
            len(tokens)
        )

        frequencies = Counter(tokens)

        for term, frequency in frequencies.items():
            postings[term].append(
                (
                    doc_index,
                    frequency,
                )
            )

    document_count = len(documents)

    average_document_length = (
        sum(document_lengths)
        / document_count
    )

    idf = {}

    for term, posting_list in postings.items():
        document_frequency = len(
            posting_list
        )

        idf[term] = math.log(
            1.0
            + (
                (
                    document_count
                    - document_frequency
                    + 0.5
                )
                / (
                    document_frequency
                    + 0.5
                )
            )
        )

    return {
        "document_ids":
            document_ids,
        "document_lengths":
            document_lengths,
        "average_document_length":
            average_document_length,
        "postings":
            postings,
        "idf":
            idf,
    }


def score_query(
    question,
    index,
):
    query_terms = set(
        tokenize(question)
    )

    scores = defaultdict(float)

    average_length = (
        index[
            "average_document_length"
        ]
    )

    document_lengths = index[
        "document_lengths"
    ]

    for term in query_terms:
        posting_list = index[
            "postings"
        ].get(term)

        if not posting_list:
            continue

        term_idf = index["idf"][term]

        for doc_index, term_frequency in posting_list:
            document_length = (
                document_lengths[
                    doc_index
                ]
            )

            denominator = (
                term_frequency
                + K1
                * (
                    1.0
                    - B
                    + B
                    * document_length
                    / average_length
                )
            )

            scores[doc_index] += (
                term_idf
                * (
                    term_frequency
                    * (K1 + 1.0)
                    / denominator
                )
            )

    return scores


def exact_gold_rank(
    scores,
    gold_index,
    document_count,
):
    gold_score = scores.get(
        gold_index,
        0.0,
    )

    if gold_score > 0.0:
        higher = 0
        equal_before = 0

        for doc_index, score in scores.items():
            if score > gold_score:
                higher += 1

            elif (
                score == gold_score
                and doc_index < gold_index
            ):
                equal_before += 1

        return (
            1
            + higher
            + equal_before
        )

    scored_before_gold = sum(
        doc_index < gold_index
        for doc_index in scores
    )

    zero_score_before_gold = (
        gold_index
        - scored_before_gold
    )

    return (
        1
        + len(scores)
        + zero_score_before_gold
    )


def top_k_documents(
    scores,
    document_count,
    k=10,
):
    ranked = heapq.nlargest(
        k,
        scores.items(),
        key=lambda item: (
            item[1],
            -item[0],
        ),
    )

    ranked_indices = [
        doc_index
        for doc_index, _ in ranked
    ]

    if len(ranked_indices) < k:
        selected = set(
            ranked_indices
        )

        for doc_index in range(
            document_count
        ):
            if doc_index not in selected:
                ranked.append(
                    (
                        doc_index,
                        0.0,
                    )
                )

                selected.add(
                    doc_index
                )

                if len(ranked) == k:
                    break

    return ranked


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
        "queries": count,

        **{
            f"recall_at_{k}":
                accumulator[
                    "hits"
                ][k]
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


def percentile(values, fraction):
    ordered = sorted(values)

    index = int(
        fraction
        * (len(ordered) - 1)
    )

    return ordered[index]


def main():
    print(
        "=== BM25 RETRIEVAL EVALUATION ==="
    )

    with INPUT_PATH.open(
        encoding="utf-8"
    ) as file:
        benchmark = json.load(file)

    documents = benchmark["documents"]
    queries = benchmark["queries"]

    print(
        "documents:",
        len(documents),
    )

    print(
        "queries:",
        len(queries),
    )

    indexing_start = time.perf_counter()

    index = build_index(
        documents
    )

    indexing_seconds = (
        time.perf_counter()
        - indexing_start
    )

    document_id_to_index = {
        document_id: doc_index
        for doc_index, document_id
        in enumerate(
            index["document_ids"]
        )
    }

    overall = empty_accumulator()
    answerable = empty_accumulator()
    unanswerable = empty_accumulator()

    query_times = []
    results = []

    print("\n=== RETRIEVE ===")

    for query_number, query in enumerate(
        queries,
        start=1,
    ):
        query_start = time.perf_counter()

        scores = score_query(
            query["question"],
            index,
        )

        gold_index = (
            document_id_to_index[
                query[
                    "gold_document_id"
                ]
            ]
        )

        gold_rank = exact_gold_rank(
            scores=scores,
            gold_index=gold_index,
            document_count=len(
                documents
            ),
        )

        ranked = top_k_documents(
            scores=scores,
            document_count=len(
                documents
            ),
            k=10,
        )

        query_seconds = (
            time.perf_counter()
            - query_start
        )

        query_times.append(
            query_seconds
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
                            rank,
                        "document_id":
                            index[
                                "document_ids"
                            ][doc_index],
                        "score":
                            score,
                    }
                    for rank, (
                        doc_index,
                        score,
                    )
                    in enumerate(
                        ranked,
                        start=1,
                    )
                ],
            }
        )

        if (
            query_number % 500 == 0
            or query_number == len(
                queries
            )
        ):
            print(
                f"processed "
                f"{query_number}/"
                f"{len(queries)}"
            )

    retrieval_seconds = sum(
        query_times
    )

    summary = {
        "method":
            "BM25",
        "parameters": {
            "k1": K1,
            "b": B,
        },
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
            "indexing_seconds":
                indexing_seconds,
            "retrieval_seconds":
                retrieval_seconds,
            "mean_query_latency_ms":
                statistics.mean(
                    query_times
                )
                * 1000.0,
            "p95_query_latency_ms":
                percentile(
                    query_times,
                    0.95,
                )
                * 1000.0,
            "queries_per_second":
                len(queries)
                / retrieval_seconds,
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

    print("\n=== BM25 SUMMARY ===")

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