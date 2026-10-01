import math
import time

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_PATH = "data/processed/self_supervised_squad_v3_packed_512"

BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_OPTIMIZER_STEPS = 4

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 42


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
                batch["attention_mask"][:, 1:].sum().item()
            )

            total_loss += (
                outputs.loss.item()
                * prediction_tokens
            )

            total_prediction_tokens += prediction_tokens

    model.train()

    return total_loss / total_prediction_tokens


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda")

    dataset = load_from_disk(DATASET_PATH)

    train_subset = dataset["train"].select(range(64))
    validation_subset = dataset["validation"].select(range(16))

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        collate_fn=collate_batch,
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
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

    optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()

    print("=== SELF-SUPERVISED TRAINING SMOKE TEST ===")
    print("model:", MODEL_NAME)
    print("gpu:", torch.cuda.get_device_name(0))
    print("train blocks:", len(train_subset))
    print("validation blocks:", len(validation_subset))
    print("batch size:", BATCH_SIZE)
    print(
        "gradient accumulation:",
        GRADIENT_ACCUMULATION_STEPS,
    )
    print(
        "effective blocks per update:",
        BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    )
    print("learning rate:", LEARNING_RATE)

    start_time = time.perf_counter()

    optimizer_step = 0
    accumulated_tokens = 0

    initial_validation_loss = evaluate(
        model,
        validation_loader,
        device,
    )

    print(
        "initial validation loss:",
        initial_validation_loss,
    )

    for batch_index, batch in enumerate(
        train_loader,
        start=1,
    ):
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            outputs = model(**batch)
            loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss: {loss.item()}"
            )

        prediction_tokens = int(
            batch["attention_mask"][:, 1:].sum().item()
        )

        (
            loss * prediction_tokens
        ).backward()

        accumulated_tokens += prediction_tokens

        if (
            batch_index
            % GRADIENT_ACCUMULATION_STEPS
            == 0
        ):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(
                        accumulated_tokens
                    )

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            if not torch.isfinite(grad_norm):
                raise RuntimeError(
                    f"Non-finite gradient norm: {grad_norm}"
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            optimizer_step += 1

            print(
                f"optimizer step {optimizer_step}: "
                f"loss={loss.item():.6f}, "
                f"grad_norm={float(grad_norm):.6f}"
            )

            accumulated_tokens = 0

            if optimizer_step >= MAX_OPTIMIZER_STEPS:
                break

    final_validation_loss = evaluate(
        model,
        validation_loader,
        device,
    )

    elapsed = time.perf_counter() - start_time

    peak_vram_gib = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print("\n=== SMOKE TEST RESULT ===")
    print("optimizer steps:", optimizer_step)
    print(
        "initial validation loss:",
        initial_validation_loss,
    )
    print(
        "final validation loss:",
        final_validation_loss,
    )
    print(
        "final validation perplexity:",
        math.exp(final_validation_loss),
    )
    print("elapsed seconds:", elapsed)
    print("peak VRAM GiB:", peak_vram_gib)
    print("result: SUCCESS")


if __name__ == "__main__":
    main()