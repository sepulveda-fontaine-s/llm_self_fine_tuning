import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from evaluate_grounding_factuality import (
    BATCH_SIZE,
    MAX_LENGTH,
    NLI_MODEL_NAME,
    lexical_support,
    resolve_nli_labels,
    run_nli,
    summarize_all,
)


INPUT_PATH = Path(
    "results/evaluation/"
    "retrieval_conditioned_qa_refined_full.json"
)

BENCHMARK_PATH = Path(
    "data/processed/"
    "retrieval_benchmark.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "results/evaluation/"
    "retrieval_conditioned_grounding_full.json"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
    )

    return parser.parse_args()


def prepare_rows(
    predictions,
    documents,
    limit=None,
):
    if limit is not None:
        predictions = predictions[:limit]

    rows = []

    for result in predictions:
        retrieved_document_id = (
            result[
                "retrieved_document_id"
            ]
        )

        context = documents[
            retrieved_document_id
        ]

        rows.append(
            {
                "id":
                    result["id"],

                "answerable":
                    result["answerable"],

                "prediction":
                    result["prediction"],

                "gold_answers":
                    result["gold_answers"],

                "exact_match":
                    result["exact_match"],

                "f1":
                    result["f1"],

                "abstained":
                    result["abstained"],

                "context":
                    context,

                "retrieved_document_id":
                    retrieved_document_id,

                "gold_document_id":
                    result[
                        "gold_document_id"
                    ],

                "retrieval_gold_at_1":
                    result[
                        "retrieval_gold_at_1"
                    ],

                "gold_rrf_rank":
                    result[
                        "gold_rrf_rank"
                    ],

                "lexically_supported":
                    lexical_support(
                        result["prediction"],
                        context,
                    ),
            }
        )

    return rows


def summarize_by_retrieval(rows):
    hit = [
        row
        for row in rows
        if row[
            "retrieval_gold_at_1"
        ]
    ]

    miss = [
        row
        for row in rows
        if not row[
            "retrieval_gold_at_1"
        ]
    ]

    return {
        "overall":
            summarize_all(rows),

        "gold_at_1":
            summarize_all(hit),

        "gold_not_at_1":
            summarize_all(miss),
    }


def main():
    args = parse_args()

    print(
        "=== RETRIEVAL-CONDITIONED "
        "GROUNDING / FACTUALITY ==="
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    with INPUT_PATH.open(
        encoding="utf-8"
    ) as file:
        qa_results = json.load(file)

    with BENCHMARK_PATH.open(
        encoding="utf-8"
    ) as file:
        benchmark = json.load(file)

    documents = {
        row["document_id"]:
            row["text"]
        for row in benchmark[
            "documents"
        ]
    }

    base_rows = prepare_rows(
        qa_results[
            "base"
        ]["predictions"],
        documents,
        args.limit,
    )

    adapted_rows = prepare_rows(
        qa_results[
            "self_supervised"
        ]["predictions"],
        documents,
        args.limit,
    )

    print(
        "base examples:",
        len(base_rows),
    )

    print(
        "self-supervised examples:",
        len(adapted_rows),
    )

    print(
        "NLI evaluator:",
        NLI_MODEL_NAME,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            NLI_MODEL_NAME
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            NLI_MODEL_NAME
        )
    )

    device = torch.device("cuda")

    model.to(device)
    model.eval()

    label_ids = resolve_nli_labels(
        model
    )

    print(
        "\n=== BASE + RRF NLI ==="
    )

    run_nli(
        base_rows,
        tokenizer,
        model,
        device,
        label_ids,
    )

    print(
        "\n=== SELF-SUPERVISED "
        "+ RRF NLI ==="
    )

    run_nli(
        adapted_rows,
        tokenizer,
        model,
        device,
        label_ids,
    )

    base_summary = (
        summarize_by_retrieval(
            base_rows
        )
    )

    adapted_summary = (
        summarize_by_retrieval(
            adapted_rows
        )
    )

    output = {
        "experiment":
            (
                "retrieval_conditioned_"
                "grounding"
            ),

        "retriever":
            "Hybrid RRF top-1",

        "nli_evaluator":
            NLI_MODEL_NAME,

        "max_length":
            MAX_LENGTH,

        "limit":
            args.limit,

        "metric_notes": {
            "nli_entailment_rate":
                (
                    "Retrieved context "
                    "entails generated answer."
                ),

            "nli_non_entailment_rate":
                (
                    "Neutral + contradiction "
                    "against retrieved context."
                ),

            "nli_hallucination_proxy_rate":
                (
                    "Automatic NLI proxy only; "
                    "not direct ground-truth "
                    "hallucination measurement."
                ),

            "supported_but_exact_match_wrong_rate":
                (
                    "Answer is NLI-supported "
                    "but fails exact-match "
                    "task correctness."
                ),
        },

        "base": {
            "summary":
                base_summary,
            "examples":
                base_rows,
        },

        "self_supervised": {
            "summary":
                adapted_summary,
            "examples":
                adapted_rows,
        },
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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
        "\n=== BASE + RRF "
        "GROUNDING SUMMARY ==="
    )

    print(
        json.dumps(
            base_summary,
            indent=2,
        )
    )

    print(
        "\n=== SELF-SUPERVISED "
        "+ RRF GROUNDING SUMMARY ==="
    )

    print(
        json.dumps(
            adapted_summary,
            indent=2,
        )
    )

    print(
        "\nsaved:",
        output_path,
    )

    print(
        "\n=== COMPLETE ==="
    )


if __name__ == "__main__":
    main()