import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from evaluate_downstream_qa import (
    ADAPTED_MODEL_PATH,
    CONFIG_PATH,
    SEED,
    generate_predictions,
    split_validation_by_context,
)


BENCHMARK_PATH = Path(
    "data/processed/retrieval_benchmark.json"
)

RRF_PATH = Path(
    "results/evaluation/"
    "retrieval_hybrid_rrf.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "results/evaluation/"
    "retrieval_conditioned_qa_refined.json"
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


def safe_mean(values):
    if not values:
        return None

    return sum(values) / len(values)


def summarize_subset(results):
    answerable = [
        row
        for row in results
        if row["answerable"]
    ]

    unanswerable = [
        row
        for row in results
        if not row["answerable"]
    ]

    return {
        "examples":
            len(results),

        "answerable_examples":
            len(answerable),

        "unanswerable_examples":
            len(unanswerable),

        "task_exact_match":
            safe_mean(
                [
                    row["exact_match"]
                    for row in results
                ]
            ),

        "task_f1":
            safe_mean(
                [
                    row["f1"]
                    for row in results
                ]
            ),

        "answerable_exact_match":
            safe_mean(
                [
                    row["exact_match"]
                    for row in answerable
                ]
            ),

        "answerable_f1":
            safe_mean(
                [
                    row["f1"]
                    for row in answerable
                ]
            ),

        "answerable_false_abstention_rate":
            safe_mean(
                [
                    float(row["abstained"])
                    for row in answerable
                ]
            ),

        "unanswerable_abstention_accuracy":
            safe_mean(
                [
                    float(row["abstained"])
                    for row in unanswerable
                ]
            ),

        "unanswerable_false_answer_rate":
            (
                None
                if not unanswerable
                else 1.0
                - safe_mean(
                    [
                        float(
                            row["abstained"]
                        )
                        for row in unanswerable
                    ]
                )
            ),
    }


def summarize_by_retrieval(results):
    hit = [
        row
        for row in results
        if row["retrieval_gold_at_1"]
    ]

    miss = [
        row
        for row in results
        if not row["retrieval_gold_at_1"]
    ]

    return {
        "overall":
            summarize_subset(results),

        "gold_at_1":
            summarize_subset(hit),

        "gold_not_at_1":
            summarize_subset(miss),

        "retrieval_gold_at_1_rate":
            len(hit) / len(results),
    }


def attach_retrieval_metadata(
    predictions,
    examples_by_id,
):
    enriched = []

    for prediction in predictions:
        example = examples_by_id[
            prediction["id"]
        ]

        row = dict(prediction)

        row[
            "gold_document_id"
        ] = example[
            "gold_document_id"
        ]

        row[
            "retrieved_document_id"
        ] = example[
            "retrieved_document_id"
        ]

        row[
            "retrieval_gold_at_1"
        ] = example[
            "retrieval_gold_at_1"
        ]

        row[
            "gold_rrf_rank"
        ] = example[
            "gold_rrf_rank"
        ]

        enriched.append(row)

    return enriched


def main():
    args = parse_args()

    print(
        "=== RETRIEVAL-CONDITIONED QA ==="
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    with BENCHMARK_PATH.open(
        encoding="utf-8"
    ) as file:
        benchmark = json.load(file)

    with RRF_PATH.open(
        encoding="utf-8"
    ) as file:
        rrf = json.load(file)

    documents = {
        row["document_id"]:
            row["text"]
        for row in benchmark["documents"]
    }

    retrieval_results = {
        row["query_id"]:
            row
        for row in rrf["results"]
    }

    raw = load_dataset(
        "rajpurkar/squad_v2"
    )

    _, test = split_validation_by_context(
        raw["validation"]
    )

    assert len(test) == 5940

    retrieved_examples = []

    for index in range(len(test)):
        original = test[index]

        query_id = original["id"]

        retrieval = retrieval_results[
            query_id
        ]

        top_1 = retrieval["top_10"][0]

        retrieved_document_id = (
            top_1["document_id"]
        )

        gold_document_id = (
            retrieval[
                "gold_document_id"
            ]
        )

        retrieved_context = documents[
            retrieved_document_id
        ]

        example = {
            "id":
                query_id,

            "question":
                original["question"],

            "answers":
                original["answers"],

            # IMPORTANT:
            # generation now sees the
            # retrieved RRF top-1 context.
            "context":
                retrieved_context,

            "gold_document_id":
                gold_document_id,

            "retrieved_document_id":
                retrieved_document_id,

            "retrieval_gold_at_1":
                (
                    retrieved_document_id
                    == gold_document_id
                ),

            "gold_rrf_rank":
                retrieval[
                    "gold_rank"
                ],
        }

        retrieved_examples.append(
            example
        )

    if args.limit is not None:
        retrieved_examples = (
            retrieved_examples[
                :args.limit
            ]
        )

    examples_by_id = {
        row["id"]: row
        for row in retrieved_examples
    }

    print(
        "examples:",
        len(retrieved_examples),
    )

    retrieval_hits = sum(
        row["retrieval_gold_at_1"]
        for row in retrieved_examples
    )

    print(
        "RRF gold@1:",
        retrieval_hits,
    )

    print(
        "RRF gold@1 rate:",
        retrieval_hits
        / len(retrieved_examples),
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            config["model"]["name"]
        )
    )

    tokenizer.padding_side = "left"

    generation_config = (
        config["generation"]
    )

    print(
        "generation config:",
        generation_config,
    )

    device = torch.device("cuda")

    base_predictions = (
        generate_predictions(
            model_path=(
                config["model"]["name"]
            ),
            model_name=(
                "BASE MODEL + RRF TOP-1"
            ),
            examples=retrieved_examples,
            tokenizer=tokenizer,
            generation_config=(
                generation_config
            ),
            device=device,
        )
    )

    adapted_predictions = (
        generate_predictions(
            model_path=(
                ADAPTED_MODEL_PATH
            ),
            model_name=(
                "SELF-SUPERVISED MODEL "
                "+ RRF TOP-1"
            ),
            examples=retrieved_examples,
            tokenizer=tokenizer,
            generation_config=(
                generation_config
            ),
            device=device,
        )
    )

    base_predictions = (
        attach_retrieval_metadata(
            base_predictions,
            examples_by_id,
        )
    )

    adapted_predictions = (
        attach_retrieval_metadata(
            adapted_predictions,
            examples_by_id,
        )
    )

    base_summary = (
        summarize_by_retrieval(
            base_predictions
        )
    )

    adapted_summary = (
        summarize_by_retrieval(
            adapted_predictions
        )
    )

    output = {
        "experiment":
            "retrieval_conditioned_qa",

        "retriever":
            "Hybrid RRF",

        "rrf_k":
            rrf["summary"]["rrf_k"],

        "retrieved_context":
            "RRF top-1",

        "examples":
            len(retrieved_examples),

        "seed":
            SEED,

        "generation_config":
            generation_config,

        "base": {
            "summary":
                base_summary,

            "predictions":
                base_predictions,
        },

        "self_supervised": {
            "model_path":
                ADAPTED_MODEL_PATH,

            "summary":
                adapted_summary,

            "predictions":
                adapted_predictions,
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
        "\n=== BASE + RRF SUMMARY ==="
    )

    print(
        json.dumps(
            base_summary,
            indent=2,
        )
    )

    print(
        "\n=== SELF-SUPERVISED + RRF SUMMARY ==="
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