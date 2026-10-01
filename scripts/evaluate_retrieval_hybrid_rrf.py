import json
from pathlib import Path


BM25_PATH = Path(
    "results/evaluation/retrieval_bm25.json"
)

DENSE_PATH = Path(
    "results/evaluation/"
    "retrieval_dense_e5_small_v2.json"
)

OUTPUT_PATH = Path(
    "results/evaluation/"
    "retrieval_hybrid_rrf.json"
)

RRF_K = 60
K_VALUES = (1, 3, 5, 10)


def fuse_results(
    bm25_row,
    dense_row,
):
    scores = {}
    component_ranks = {}

    for method, row in (
        ("bm25", bm25_row),
        ("dense", dense_row),
    ):
        for item in row["top_10"]:
            document_id = item["document_id"]
            rank = item["rank"]

            scores[document_id] = (
                scores.get(
                    document_id,
                    0.0,
                )
                + 1.0 / (RRF_K + rank)
            )

            component_ranks.setdefault(
                document_id,
                {},
            )[method] = rank

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    top_10 = []

    for rank, (
        document_id,
        score,
    ) in enumerate(
        ranked[:10],
        start=1,
    ):
        ranks = component_ranks[
            document_id
        ]

        top_10.append(
            {
                "rank": rank,
                "document_id":
                    document_id,
                "rrf_score":
                    score,
                "bm25_rank":
                    ranks.get("bm25"),
                "dense_rank":
                    ranks.get("dense"),
            }
        )

    gold_document_id = (
        bm25_row["gold_document_id"]
    )

    gold_rank = next(
        (
            item["rank"]
            for item in top_10
            if (
                item["document_id"]
                == gold_document_id
            )
        ),
        None,
    )

    return gold_rank, top_10


def empty_metrics():
    return {
        "queries": 0,
        "hits": {
            k: 0
            for k in K_VALUES
        },
        "reciprocal_rank_sum": 0.0,
    }


def update_metrics(
    metrics,
    gold_rank,
):
    metrics["queries"] += 1

    if gold_rank is None:
        return

    metrics[
        "reciprocal_rank_sum"
    ] += 1.0 / gold_rank

    for k in K_VALUES:
        if gold_rank <= k:
            metrics["hits"][k] += 1


def finalize(metrics):
    count = metrics["queries"]

    return {
        "queries":
            count,

        **{
            f"recall_at_{k}":
                metrics["hits"][k]
                / count
            for k in K_VALUES
        },

        "mrr_at_10":
            metrics[
                "reciprocal_rank_sum"
            ]
            / count,
    }


def main():
    print(
        "=== HYBRID RRF RETRIEVAL ==="
    )

    with BM25_PATH.open(
        encoding="utf-8"
    ) as file:
        bm25 = json.load(file)

    with DENSE_PATH.open(
        encoding="utf-8"
    ) as file:
        dense = json.load(file)

    bm25_results = {
        row["query_id"]: row
        for row in bm25["results"]
    }

    dense_results = {
        row["query_id"]: row
        for row in dense["results"]
    }

    assert (
        bm25_results.keys()
        == dense_results.keys()
    )

    overall = empty_metrics()
    answerable = empty_metrics()
    unanswerable = empty_metrics()

    results = []

    for number, query_id in enumerate(
        bm25_results,
        start=1,
    ):
        bm25_row = bm25_results[
            query_id
        ]

        dense_row = dense_results[
            query_id
        ]

        assert (
            bm25_row["gold_document_id"]
            == dense_row[
                "gold_document_id"
            ]
        )

        assert (
            bm25_row["answerable"]
            == dense_row["answerable"]
        )

        gold_rank, top_10 = fuse_results(
            bm25_row,
            dense_row,
        )

        update_metrics(
            overall,
            gold_rank,
        )

        if bm25_row["answerable"]:
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
                    query_id,
                "answerable":
                    bm25_row[
                        "answerable"
                    ],
                "gold_document_id":
                    bm25_row[
                        "gold_document_id"
                    ],
                "gold_rank":
                    gold_rank,
                "top_10":
                    top_10,
            }
        )

        if (
            number % 500 == 0
            or number
            == len(bm25_results)
        ):
            print(
                f"processed "
                f"{number}/"
                f"{len(bm25_results)}"
            )

    summary = {
        "method":
            "Reciprocal Rank Fusion",
        "rrf_k":
            RRF_K,
        "candidate_pool":
            (
                "union of BM25 top-10 "
                "and dense top-10"
            ),
        "mrr_definition":
            "MRR@10",
        "overall":
            finalize(overall),
        "answerable":
            finalize(answerable),
        "unanswerable":
            finalize(unanswerable),
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
        "\n=== HYBRID RRF SUMMARY ==="
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