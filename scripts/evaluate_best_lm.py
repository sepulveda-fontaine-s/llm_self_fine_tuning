from pathlib import Path
import json
import math

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM


BASELINE_PATH = Path(
    "results/baseline/base_lm_validation.json"
)

MODEL_PATH = (
    "results/checkpoints/"
    "self_supervised_v3_refined/best_model"
)

DATASET_PATH = (
    "data/processed/"
    "self_supervised_squad_v3_packed_512"
)

OUTPUT_PATH = Path(
    "results/evaluation/"
    "base_vs_self_supervised_refined_lm.json"
)

BATCH_SIZE = 8


def collate_batch(examples):
    input_ids = torch.tensor(
        [example["input_ids"] for example in examples],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [example["attention_mask"] for example in examples],
        dtype=torch.long,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_prediction_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(**batch)

            prediction_tokens = int(
                batch["attention_mask"][:, 1:]
                .sum()
                .item()
            )

            total_loss += (
                outputs.loss.item()
                * prediction_tokens
            )

            total_prediction_tokens += (
                prediction_tokens
            )

    loss = (
        total_loss
        / total_prediction_tokens
    )

    return {
        "validation_loss": loss,
        "perplexity": math.exp(loss),
        "prediction_tokens":
            total_prediction_tokens,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    with BASELINE_PATH.open(
        encoding="utf-8"
    ) as file:
        baseline = json.load(file)

    dataset = load_from_disk(
        DATASET_PATH
    )

    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda")

    model = (
        AutoModelForCausalLM
        .from_pretrained(MODEL_PATH)
        .to(device)
    )

    adapted = evaluate(
        model,
        validation_loader,
        device,
    )

    loss_difference = (
        adapted["validation_loss"]
        - baseline["validation_loss"]
    )

    loss_change_percent = (
        loss_difference
        / baseline["validation_loss"]
        * 100
    )

    perplexity_difference = (
        adapted["perplexity"]
        - baseline["perplexity"]
    )

    perplexity_change_percent = (
        perplexity_difference
        / baseline["perplexity"]
        * 100
    )

    result = {
        "dataset": DATASET_PATH,
        "split": "validation",
        "validation_blocks":
            len(dataset["validation"]),
        "batch_size": BATCH_SIZE,
        "base_model": {
            "name": baseline["model"],
            "validation_loss":
                baseline["validation_loss"],
            "perplexity":
                baseline["perplexity"],
        },
        "adapted_model": {
            "path": MODEL_PATH,
            "selected_epoch": 1,
            "validation_loss":
                adapted["validation_loss"],
            "perplexity":
                adapted["perplexity"],
        },
        "comparison": {
            "loss_difference":
                loss_difference,
            "loss_change_percent":
                loss_change_percent,
            "perplexity_difference":
                perplexity_difference,
            "perplexity_change_percent":
                perplexity_change_percent,
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
            result,
            file,
            indent=2,
        )

    print("=== BASE MODEL ===")
    print(
        "validation loss:",
        baseline["validation_loss"],
    )
    print(
        "perplexity:",
        baseline["perplexity"],
    )

    print("\n=== SELF-SUPERVISED MODEL ===")
    print(
        "validation loss:",
        adapted["validation_loss"],
    )
    print(
        "perplexity:",
        adapted["perplexity"],
    )

    print("\n=== COMPARISON ===")
    print(
        "loss difference:",
        loss_difference,
    )
    print(
        "loss change %:",
        loss_change_percent,
    )
    print(
        "perplexity difference:",
        perplexity_difference,
    )
    print(
        "perplexity change %:",
        perplexity_change_percent,
    )

    print("\nsaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()