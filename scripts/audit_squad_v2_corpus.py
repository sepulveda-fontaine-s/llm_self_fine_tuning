from collections import Counter
from hashlib import sha256

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "rajpurkar/squad_v2"


def context_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def audit_split(dataset, split_name, tokenizer):
    split = dataset[split_name]

    contexts = [example["context"].strip() for example in split]
    unique_contexts = list(dict.fromkeys(contexts))

    lengths = np.asarray(
        [
            len(
                tokenizer(
                    text,
                    add_special_tokens=False,
                )["input_ids"]
            )
            for text in unique_contexts
        ]
    )

    print(f"\n=== {split_name.upper()} ===")
    print("examples:", len(split))
    print("contexts:", len(contexts))
    print("unique contexts:", len(unique_contexts))
    print("duplicate-context rows:", len(contexts) - len(unique_contexts))
    print("total characters:", sum(len(x) for x in unique_contexts))
    print("total tokens:", int(lengths.sum()))

    print("\n=== TOKEN LENGTHS PER UNIQUE CONTEXT ===")
    print("min:", int(lengths.min()))
    print("p50:", int(np.percentile(lengths, 50)))
    print("p90:", int(np.percentile(lengths, 90)))
    print("p95:", int(np.percentile(lengths, 95)))
    print("p99:", int(np.percentile(lengths, 99)))
    print("max:", int(lengths.max()))

    return unique_contexts


def main():
    print("=== LOADING SQUAD V2 ===")

    dataset = load_dataset(DATASET_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(dataset)

    train_contexts = audit_split(
        dataset,
        "train",
        tokenizer,
    )

    validation_contexts = audit_split(
        dataset,
        "validation",
        tokenizer,
    )

    train_hashes = {context_hash(x) for x in train_contexts}
    validation_hashes = {context_hash(x) for x in validation_contexts}

    overlap = train_hashes & validation_hashes

    print("\n=== TRAIN / VALIDATION OVERLAP ===")
    print("exact context overlap:", len(overlap))

    print("\n=== SAMPLE TRAIN CONTEXT ===")
    print(train_contexts[0])


if __name__ == "__main__":
    main()