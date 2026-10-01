import argparse
import json
import random
import re
import string
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


INPUT_PATH = Path(
    "results/evaluation/"
    "base_vs_self_supervised_refined_downstream_qa.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "results/evaluation/"
    "grounding_factuality_refined.json"
)

NLI_MODEL_NAME = "FacebookAI/roberta-large-mnli"

SEED = 42
BATCH_SIZE = 16
MAX_LENGTH = 512


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


def split_validation_by_context(
    validation_dataset,
):
    context_groups = defaultdict(list)

    for index, example in enumerate(
        validation_dataset
    ):
        context_groups[
            example["context"]
        ].append(index)

    contexts = list(context_groups.keys())

    rng = random.Random(SEED)
    rng.shuffle(contexts)

    validation_indices = []
    test_indices = []

    for context in contexts:
        indices = context_groups[context]

        if (
            len(validation_indices)
            <= len(test_indices)
        ):
            validation_indices.extend(indices)
        else:
            test_indices.extend(indices)

    return (
        validation_dataset.select(
            validation_indices
        ),
        validation_dataset.select(
            test_indices
        ),
    )


def normalize_text(text):
    text = text.lower()

    text = "".join(
        char
        for char in text
        if char not in string.punctuation
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def lexical_support(
    prediction,
    context,
):
    prediction = normalize_text(
        prediction
    )

    context = normalize_text(
        context
    )

    if not prediction:
        return False

    return prediction in context


def resolve_nli_labels(model):
    id2label = {
        int(index): str(label).upper()
        for index, label
        in model.config.id2label.items()
    }

    def find_id(name):
        for index, label in id2label.items():
            if name in label:
                return index

        raise RuntimeError(
            f"Could not find NLI label "
            f"{name} in {id2label}"
        )

    return {
        "contradiction":
            find_id("CONTRADICTION"),
        "neutral":
            find_id("NEUTRAL"),
        "entailment":
            find_id("ENTAILMENT"),
    }


def run_nli(
    rows,
    tokenizer,
    model,
    device,
    label_ids,
):
    candidates = [
        index
        for index, row in enumerate(rows)
        if (
            not row["abstained"]
            and row["prediction"].strip()
        )
    ]

    for start in range(
        0,
        len(candidates),
        BATCH_SIZE,
    ):
        indices = candidates[
            start:start + BATCH_SIZE
        ]

        premises = [
            rows[index]["context"]
            for index in indices
        ]

        hypotheses = [
            rows[index]["prediction"]
            for index in indices
        ]

        inputs = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation="only_first",
            max_length=MAX_LENGTH,
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
                logits = model(
                    **inputs
                ).logits

        probabilities = torch.softmax(
            logits.float(),
            dim=-1,
        ).cpu()

        for position, row_index in enumerate(
            indices
        ):
            probs = probabilities[position]

            contradiction = float(
                probs[
                    label_ids[
                        "contradiction"
                    ]
                ].item()
            )

            neutral = float(
                probs[
                    label_ids["neutral"]
                ].item()
            )

            entailment = float(
                probs[
                    label_ids["entailment"]
                ].item()
            )

            scores = {
                "contradiction":
                    contradiction,
                "neutral":
                    neutral,
                "entailment":
                    entailment,
            }

            predicted_label = max(
                scores,
                key=scores.get,
            )

            rows[row_index][
                "nli_label"
            ] = predicted_label

            rows[row_index][
                "nli_scores"
            ] = scores

        print(
            "NLI processed "
            f"{min(start + BATCH_SIZE, len(candidates))}"
            f"/{len(candidates)}"
        )

    for row in rows:
        if row["abstained"]:
            row["nli_label"] = "abstention"
            row["nli_scores"] = None

        elif not row["prediction"].strip():
            row["nli_label"] = "empty"
            row["nli_scores"] = None

def summarize(rows):
    total = len(rows)

    abstentions = sum(
        row["abstained"]
        for row in rows
    )

    answered = [
        row
        for row in rows
        if (
            not row["abstained"]
            and row["prediction"].strip()
        )
    ]

    nli_evaluated = len(answered)

    entailments = sum(
        row["nli_label"] == "entailment"
        for row in answered
    )

    neutrals = sum(
        row["nli_label"] == "neutral"
        for row in answered
    )

    contradictions = sum(
        row["nli_label"] == "contradiction"
        for row in answered
    )

    lexical = sum(
        row["lexically_supported"]
        for row in answered
    )

    supported_but_incorrect = sum(
        (
            row["nli_label"] == "entailment"
            and row["exact_match"] == 0.0
        )
        for row in answered
    )

    supported_with_positive_f1 = sum(
        (
            row["nli_label"] == "entailment"
            and row["f1"] > 0.0
        )
        for row in answered
    )

    def ratio(value, denominator):
        if denominator == 0:
            return None

        return value / denominator

    return {
        "examples": total,
        "answered_examples":
            nli_evaluated,
        "abstentions":
            abstentions,
        "abstention_rate":
            ratio(
                abstentions,
                total,
            ),

        "lexical_grounding_rate":
            ratio(
                lexical,
                nli_evaluated,
            ),

        "nli_entailment_rate":
            ratio(
                entailments,
                nli_evaluated,
            ),

        "nli_neutral_rate":
            ratio(
                neutrals,
                nli_evaluated,
            ),

        "nli_contradiction_rate":
            ratio(
                contradictions,
                nli_evaluated,
            ),

        "nli_non_entailment_rate":
            ratio(
                neutrals
                + contradictions,
                nli_evaluated,
            ),

        "nli_hallucination_proxy_rate":
            ratio(
                neutrals
                + contradictions,
                nli_evaluated,
            ),

        "supported_but_exact_match_wrong_rate":
            ratio(
                supported_but_incorrect,
                nli_evaluated,
            ),

        "supported_with_positive_f1_rate":
            ratio(
                supported_with_positive_f1,
                nli_evaluated,
            ),
    }


def summarize_all(rows):
    return {
        "overall":
            summarize(rows),

        "answerable":
            summarize(
                [
                    row
                    for row in rows
                    if row["answerable"]
                ]
            ),

        "unanswerable":
            summarize(
                [
                    row
                    for row in rows
                    if not row["answerable"]
                ]
            ),
    }


def prepare_rows(
    predictions,
    contexts,
    limit,
):
    if limit is not None:
        predictions = predictions[:limit]

    rows = []

    for result in predictions:
        example_id = result["id"]

        context = contexts[
            example_id
        ]

        rows.append(
            {
                "id": example_id,
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
                "lexically_supported":
                    lexical_support(
                        result[
                            "prediction"
                        ],
                        context,
                    ),
            }
        )

    return rows


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        "=== GROUNDING / FACTUALITY EVALUATION ==="
    )

    with INPUT_PATH.open(
        encoding="utf-8"
    ) as file:
        evaluation = json.load(file)

    raw = load_dataset(
        "rajpurkar/squad_v2"
    )

    _, test = split_validation_by_context(
        raw["validation"]
    )

    assert len(test) == 5940

    contexts = {
        example["id"]:
            example["context"]
        for example in test
    }

    base_rows = prepare_rows(
        evaluation["base"]["predictions"],
        contexts,
        args.limit,
    )

    adapted_rows = prepare_rows(
        evaluation[
            "self_supervised"
        ]["predictions"],
        contexts,
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

    tokenizer = AutoTokenizer.from_pretrained(
        NLI_MODEL_NAME
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

    print("\n=== BASE NLI ===")

    run_nli(
        base_rows,
        tokenizer,
        model,
        device,
        label_ids,
    )

    print(
        "\n=== SELF-SUPERVISED NLI ==="
    )

    run_nli(
        adapted_rows,
        tokenizer,
        model,
        device,
        label_ids,
    )

    base_summary = summarize_all(
        base_rows
    )

    adapted_summary = summarize_all(
        adapted_rows
    )

    output = {
        "nli_evaluator":
            NLI_MODEL_NAME,
        "max_length":
            MAX_LENGTH,
        "limit":
            args.limit,
        "metric_notes": {
            "lexical_grounding_rate":
                (
                    "Normalized generated answer "
                    "appears verbatim in context."
                ),
            "nli_entailment_rate":
                (
                    "NLI evaluator classifies "
                    "context -> answer as entailment."
                ),
            "nli_hallucination_proxy_rate":
                (
                    "Neutral + contradiction among "
                    "non-abstaining answers. "
                    "This is an automatic proxy, "
                    "not a direct factuality ground truth."
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

    print("\n=== BASE SUMMARY ===")
    print(
        json.dumps(
            base_summary,
            indent=2,
        )
    )

    print(
        "\n=== SELF-SUPERVISED SUMMARY ==="
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


if __name__ == "__main__":
    main()