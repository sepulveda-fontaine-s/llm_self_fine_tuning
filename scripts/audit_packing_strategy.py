from datasets import load_from_disk
import numpy as np
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_PATH = "data/processed/self_supervised_squad_v3"
BLOCK_SIZES = [256, 512, 1024]


def audit_split(dataset, split_name, tokenizer):
    split = dataset[split_name]

    lengths = []
    total_tokens = 0

    for text in split["text"]:
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        # One EOS token separates documents during packing.
        length_with_eos = len(token_ids) + 1

        lengths.append(length_with_eos)
        total_tokens += length_with_eos

    lengths = np.asarray(lengths)

    print(f"\n=== {split_name.upper()} ===")
    print("documents:", len(split))
    print("total tokens incl. EOS:", total_tokens)

    print("\n=== DOCUMENT TOKEN LENGTHS INCL. EOS ===")
    print("min:", int(lengths.min()))
    print("p50:", int(np.percentile(lengths, 50)))
    print("p90:", int(np.percentile(lengths, 90)))
    print("p95:", int(np.percentile(lengths, 95)))
    print("p99:", int(np.percentile(lengths, 99)))
    print("max:", int(lengths.max()))

    print("\n=== PACKING CANDIDATES ===")

    for block_size in BLOCK_SIZES:
        full_blocks = total_tokens // block_size
        remainder = total_tokens % block_size

        dropped_ratio = (
            remainder / total_tokens
            if total_tokens > 0
            else 0.0
        )

        print(f"\nblock_size: {block_size}")
        print("full blocks:", full_blocks)
        print("remainder tokens:", remainder)
        print("drop remainder ratio:", dropped_ratio)


def main():
    print("=== LOADING PREPARED CORPUS ===")

    dataset = load_from_disk(DATASET_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(dataset)

    audit_split(
        dataset,
        "train",
        tokenizer,
    )

    audit_split(
        dataset,
        "validation",
        tokenizer,
    )


if __name__ == "__main__":
    main()