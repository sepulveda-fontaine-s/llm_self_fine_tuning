from pathlib import Path
import json
import math

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_PATH = "data/processed/self_supervised_squad_v3_packed_512"
OUTPUT_PATH = Path("results/baseline/base_lm_validation.json")

BATCH_SIZE = 8


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    print("=== BASE LM VALIDATION ===")
    print("model:", MODEL_NAME)
    print("device:", torch.cuda.get_device_name(0))

    dataset = load_from_disk(DATASET_PATH)
    validation = dataset["validation"]

    validation.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )

    loader = DataLoader(
        validation,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
    ).to(device)

    model.eval()

    total_loss = 0.0
    total_prediction_tokens = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = input_ids.clone()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Causal LM loss predicts tokens 1..L-1 from tokens 0..L-2.
            prediction_tokens = int(
                attention_mask[:, 1:].sum().item()
            )

            total_loss += (
                outputs.loss.item()
                * prediction_tokens
            )
            total_prediction_tokens += prediction_tokens

    validation_loss = total_loss / total_prediction_tokens
    perplexity = math.exp(validation_loss)

    result = {
        "model": MODEL_NAME,
        "dataset": DATASET_PATH,
        "split": "validation",
        "blocks": len(validation),
        "block_size": len(validation[0]["input_ids"]),
        "batch_size": BATCH_SIZE,
        "dtype": "bfloat16",
        "prediction_tokens": total_prediction_tokens,
        "validation_loss": validation_loss,
        "perplexity": perplexity,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
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

    print("\n=== RESULT ===")
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nsaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()