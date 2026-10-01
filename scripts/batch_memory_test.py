import gc

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_PATH = "data/processed/self_supervised_squad_v3_packed_512"

BATCH_SIZES = [4, 8, 12, 16]

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


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


def run_test(dataset, batch_size):
    device = torch.device("cuda")

    loader = DataLoader(
        dataset.select(range(batch_size)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    batch = next(iter(loader))

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        outputs = model(**batch)
        loss = outputs.loss

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )

    optimizer.step()

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        / (1024 ** 3)
    )

    result = {
        "batch_size": batch_size,
        "tokens_per_batch": batch_size * 512,
        "loss": float(loss.item()),
        "grad_norm": float(grad_norm),
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
    }

    del batch
    del outputs
    del optimizer
    del model
    del loader

    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("=== BATCH MEMORY TEST ===")
    print("model:", MODEL_NAME)
    print("gpu:", torch.cuda.get_device_name(0))
    print("sequence length: 512")

    dataset = load_from_disk(DATASET_PATH)["train"]

    results = []

    for batch_size in BATCH_SIZES:
        print(f"\n=== BATCH SIZE {batch_size} ===")

        try:
            result = run_test(
                dataset,
                batch_size,
            )

            results.append(result)

            for key, value in result.items():
                print(f"{key}: {value}")

            print("status: SUCCESS")

        except torch.OutOfMemoryError:
            print("status: CUDA OOM")

            gc.collect()
            torch.cuda.empty_cache()

            break

    print("\n=== SUMMARY ===")

    for result in results:
        print(
            f"batch={result['batch_size']} | "
            f"tokens={result['tokens_per_batch']} | "
            f"allocated={result['peak_allocated_gib']:.2f} GiB | "
            f"reserved={result['peak_reserved_gib']:.2f} GiB"
        )


if __name__ == "__main__":
    main()