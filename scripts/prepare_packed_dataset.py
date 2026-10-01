from pathlib import Path
import json

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
INPUT_PATH = Path("data/processed/self_supervised_squad_v3")
OUTPUT_PATH = Path("data/processed/self_supervised_squad_v3_packed_512")
MANIFEST_PATH = Path(
    "data/manifests/self_supervised_squad_v3_packed_512.json"
)

BLOCK_SIZE = 512


def pack_split(dataset, tokenizer):
    all_token_ids = []

    eos_token_id = tokenizer.eos_token_id

    if eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token.")

    for text in dataset["text"]:
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        all_token_ids.extend(token_ids)
        all_token_ids.append(eos_token_id)

    total_tokens = len(all_token_ids)

    usable_tokens = (
        total_tokens // BLOCK_SIZE
    ) * BLOCK_SIZE

    remainder_tokens = total_tokens - usable_tokens

    all_token_ids = all_token_ids[:usable_tokens]

    blocks = [
        all_token_ids[i:i + BLOCK_SIZE]
        for i in range(0, usable_tokens, BLOCK_SIZE)
    ]

    packed = Dataset.from_dict(
        {
            "input_ids": blocks,
            "attention_mask": [
                [1] * BLOCK_SIZE
                for _ in blocks
            ],
        }
    )

    stats = {
        "documents": len(dataset),
        "total_tokens_including_eos": total_tokens,
        "usable_tokens": usable_tokens,
        "remainder_tokens_dropped": remainder_tokens,
        "remainder_ratio": (
            remainder_tokens / total_tokens
            if total_tokens
            else 0.0
        ),
        "blocks": len(blocks),
        "block_size": BLOCK_SIZE,
    }

    return packed, stats


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. Refusing to overwrite."
        )

    print("=== LOADING CORPUS ===")

    corpus = load_from_disk(str(INPUT_PATH))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("=== PACKING TRAIN ===")
    train, train_stats = pack_split(
        corpus["train"],
        tokenizer,
    )

    print("=== PACKING VALIDATION ===")
    validation, validation_stats = pack_split(
        corpus["validation"],
        tokenizer,
    )

    packed = DatasetDict(
        {
            "train": train,
            "validation": validation,
        }
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    packed.save_to_disk(str(OUTPUT_PATH))

    manifest = {
        "model": MODEL_NAME,
        "objective": "causal_language_modeling",
        "training_method": (
            "full-parameter self-supervised fine-tuning"
        ),
        "block_size": BLOCK_SIZE,
        "document_separator": "eos_token",
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "train": train_stats,
        "validation": validation_stats,
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print("\n=== PACKED DATASET ===")
    print(packed)

    print("\n=== TRAIN ===")
    for key, value in train_stats.items():
        print(f"{key}: {value}")

    print("\n=== VALIDATION ===")
    for key, value in validation_stats.items():
        print(f"{key}: {value}")

    print("\n=== SAMPLE BLOCK ===")
    print("length:", len(train[0]["input_ids"]))
    print(
        "decoded prefix:",
        tokenizer.decode(
            train[0]["input_ids"][:100]
        ),
    )

    print("\nsaved dataset:", OUTPUT_PATH)
    print("saved manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    main()