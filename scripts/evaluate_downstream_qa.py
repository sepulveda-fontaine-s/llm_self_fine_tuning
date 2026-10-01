import gc
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


CONFIG_PATH = "configs/training.yaml"

ADAPTED_MODEL_PATH = (
    "results/checkpoints/"
    "self_supervised_v3_refined/best_model"
)

OUTPUT_PATH = Path(
    "results/evaluation/"
    "base_vs_self_supervised_refined_downstream_qa.json"
)
SEED = 42
BATCH_SIZE = 16

SYSTEM_MESSAGE = (
    "Answer the question using only the provided context. "
    "Return only the answer supported by the context. "
    "If the context does not contain the answer, respond exactly: "
    "I don't know."
)


def split_validation_by_context(validation_dataset):
    context_groups = defaultdict(list)

    for index, example in enumerate(validation_dataset):
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

        if len(validation_indices) <= len(test_indices):
            validation_indices.extend(indices)
        else:
            test_indices.extend(indices)

    return (
        validation_dataset.select(validation_indices),
        validation_dataset.select(test_indices),
    )


def build_squad_messages(example):
    user_message = (
        f"Context:\n{example['context']}\n\n"
        f"Question:\n{example['question']}"
    )

    if len(example["answers"]["text"]) == 0:
        answer = "I don't know."
    else:
        answer = example["answers"]["text"][0]

    return [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": user_message,
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            value,
        )

    def remove_punctuation(value):
        return "".join(
            char
            for char in value
            if char not in string.punctuation
        )

    def normalize_whitespace(value):
        return " ".join(value.split())

    return normalize_whitespace(
        remove_articles(
            remove_punctuation(
                text.lower()
            )
        )
    )


def exact_match_score(
    prediction,
    ground_truth,
):
    return float(
        normalize_answer(prediction)
        == normalize_answer(ground_truth)
    )


def f1_score(
    prediction,
    ground_truth,
):
    prediction_tokens = (
        normalize_answer(prediction).split()
    )

    ground_truth_tokens = (
        normalize_answer(
            ground_truth
        ).split()
    )

    if (
        len(prediction_tokens) == 0
        or len(ground_truth_tokens) == 0
    ):
        return float(
            prediction_tokens
            == ground_truth_tokens
        )

    common = (
        Counter(prediction_tokens)
        & Counter(ground_truth_tokens)
    )

    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = (
        num_same
        / len(prediction_tokens)
    )

    recall = (
        num_same
        / len(ground_truth_tokens)
    )

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def best_answerable_scores(
    prediction,
    gold_answers,
):
    exact_match = max(
        exact_match_score(
            prediction,
            answer,
        )
        for answer in gold_answers
    )

    f1 = max(
        f1_score(
            prediction,
            answer,
        )
        for answer in gold_answers
    )

    return exact_match, f1


def is_abstention(prediction):
    return (
        normalize_answer(prediction)
        == normalize_answer(
            "I don't know."
        )
    )


def generate_predictions(
    model_path,
    model_name,
    examples,
    tokenizer,
    generation_config,
    device,
):
    print(f"\n=== {model_name} ===")

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_path,
            dtype=torch.bfloat16,
        )
        .to(device)
    )

    model.eval()
    model.config.use_cache = True

    eos_token_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids(
            "<|im_end|>"
        ),
    ]

    results = []

    for start in range(
        0,
        len(examples),
        BATCH_SIZE,
    ):
        batch_examples = examples[
            start:start + BATCH_SIZE
        ]

        prompts = []

        for example in batch_examples:
            messages = (
                build_squad_messages(
                    example
                )[:-1]
            )

            prompt = (
                tokenizer
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        input_length = (
            inputs["input_ids"].shape[1]
        )

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=(
                    generation_config[
                        "max_new_tokens"
                    ]
                ),
                do_sample=(
                    generation_config[
                        "do_sample"
                    ]
                ),
                repetition_penalty=(
                    generation_config[
                        "repetition_penalty"
                    ]
                ),
                eos_token_id=eos_token_ids,
                pad_token_id=(
                    tokenizer.pad_token_id
                ),
            )

        generated = outputs[
            :,
            input_length:
        ]

        predictions = (
            tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        )

        for example, prediction in zip(
            batch_examples,
            predictions,
        ):
            prediction = prediction.strip()

            gold_answers = (
                example["answers"]["text"]
            )

            answerable = (
                len(gold_answers) > 0
            )

            if answerable:
                (
                    exact_match,
                    f1,
                ) = best_answerable_scores(
                    prediction,
                    gold_answers,
                )

                abstained = (
                    is_abstention(
                        prediction
                    )
                )

            else:
                abstained = (
                    is_abstention(
                        prediction
                    )
                )

                exact_match = float(
                    abstained
                )

                f1 = float(
                    abstained
                )

            results.append(
                {
                    "id": example["id"],
                    "answerable":
                        answerable,
                    "prediction":
                        prediction,
                    "gold_answers":
                        gold_answers,
                    "exact_match":
                        exact_match,
                    "f1": f1,
                    "abstained":
                        abstained,
                }
            )

        print(
            "processed "
            f"{min(start + BATCH_SIZE, len(examples))}"
            f"/{len(examples)}"
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def summarize(results):
    answerable = [
        result
        for result in results
        if result["answerable"]
    ]

    unanswerable = [
        result
        for result in results
        if not result["answerable"]
    ]

    answerable_em = sum(
        result["exact_match"]
        for result in answerable
    ) / len(answerable)

    answerable_f1 = sum(
        result["f1"]
        for result in answerable
    ) / len(answerable)

    answerable_abstentions = sum(
        result["abstained"]
        for result in answerable
    )

    correct_abstentions = sum(
        result["abstained"]
        for result in unanswerable
    )

    abstention_accuracy = (
        correct_abstentions
        / len(unanswerable)
    )

    false_answer_rate = (
        1.0
        - abstention_accuracy
    )

    task_exact_match = sum(
        result["exact_match"]
        for result in results
    ) / len(results)

    task_f1 = sum(
        result["f1"]
        for result in results
    ) / len(results)

    return {
        "examples": len(results),
        "answerable_examples":
            len(answerable),
        "unanswerable_examples":
            len(unanswerable),
        "answerable_exact_match":
            answerable_em,
        "answerable_f1":
            answerable_f1,
        "answerable_false_abstention_rate":
            (
                answerable_abstentions
                / len(answerable)
            ),
        "unanswerable_abstention_accuracy":
            abstention_accuracy,
        "unanswerable_false_answer_rate":
            false_answer_rate,
        "task_exact_match":
            task_exact_match,
        "task_f1":
            task_f1,
    }


def main():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device("cuda")

    print(
        "=== DOWNSTREAM QA EVALUATION ==="
    )

    raw = load_dataset(
        "rajpurkar/squad_v2"
    )

    _, test = (
        split_validation_by_context(
            raw["validation"]
        )
    )

    assert len(test) == 5940

    examples = [
        test[index]
        for index in range(len(test))
    ]

    print("test examples:", len(examples))

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

    base_results = (
        generate_predictions(
            model_path=(
                config["model"]["name"]
            ),
            model_name="BASE MODEL",
            examples=examples,
            tokenizer=tokenizer,
            generation_config=(
                generation_config
            ),
            device=device,
        )
    )

    adapted_results = (
        generate_predictions(
            model_path=(
                ADAPTED_MODEL_PATH
            ),
            model_name=(
                "SELF-SUPERVISED MODEL"
            ),
            examples=examples,
            tokenizer=tokenizer,
            generation_config=(
                generation_config
            ),
            device=device,
        )
    )

    base_summary = summarize(
        base_results
    )

    adapted_summary = summarize(
        adapted_results
    )

    output = {
        "split": "test",
        "test_examples": len(examples),
        "seed": SEED,
        "generation_config":
            generation_config,
        "base": {
            "summary":
                base_summary,
            "predictions":
                base_results,
        },
        "self_supervised": {
            "model_path":
                ADAPTED_MODEL_PATH,
            "summary":
                adapted_summary,
            "predictions":
                adapted_results,
        },
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

    print("\n=== BASE SUMMARY ===")

    for key, value in (
        base_summary.items()
    ):
        print(f"{key}: {value}")

    print(
        "\n=== SELF-SUPERVISED SUMMARY ==="
    )

    for key, value in (
        adapted_summary.items()
    ):
        print(f"{key}: {value}")

    print("\nsaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()