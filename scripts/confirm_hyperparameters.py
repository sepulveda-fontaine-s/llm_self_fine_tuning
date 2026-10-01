import csv
import gc
import math
import time
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


CONFIG_PATH = "configs/training.yaml"

OUTPUT_PATH = Path(
    "results/self_supervised_hyperparameter_confirmations.csv"
)


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

    model.train()

    return (
        total_loss
        / total_prediction_tokens
    )


def create_scheduler(
    name,
    optimizer,
    warmup_steps,
    total_steps,
):
    if name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    if name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    raise ValueError(
        f"Unsupported scheduler: {name}"
    )


def optimizer_update(
    model,
    optimizer,
    scheduler,
    accumulated_tokens,
    max_grad_norm,
):
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(
                accumulated_tokens
            )

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm,
    )

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    return float(grad_norm)


def train_candidate(
    candidate,
    config,
    dataset,
    device,
):
    training = config["training"]
    confirmation = config["confirmation"]

    seed = confirmation["seed"]

    batch_size = (
        training["per_device_batch_size"]
    )

    accumulation_steps = (
        training["gradient_accumulation_steps"]
    )

    max_grad_norm = training["max_grad_norm"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_batch,
    )

    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    optimizer_steps_per_epoch = math.ceil(
        len(train_loader)
        / accumulation_steps
    )

    full_training_steps = (
        optimizer_steps_per_epoch
        * training["epochs"]
    )

    warmup_steps = int(
        full_training_steps
        * candidate["warmup_ratio"]
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            config["model"]["name"]
        )
        .to(device)
    )

    model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate["learning_rate"],
        weight_decay=candidate["weight_decay"],
    )

    scheduler = create_scheduler(
        name=candidate["scheduler"],
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=full_training_steps,
    )

    optimizer.zero_grad(set_to_none=True)

    optimizer_step = 0
    micro_steps_since_update = 0
    accumulated_tokens = 0

    torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

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
            batch["attention_mask"][:, 1:]
            .sum()
            .item()
        )

        (
            loss * prediction_tokens
        ).backward()

        accumulated_tokens += prediction_tokens
        micro_steps_since_update += 1

        accumulation_boundary = (
            micro_steps_since_update
            == accumulation_steps
        )

        last_batch = (
            batch_index == len(train_loader)
        )

        if (
            accumulation_boundary
            or last_batch
        ):
            grad_norm = optimizer_update(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                accumulated_tokens=(
                    accumulated_tokens
                ),
                max_grad_norm=max_grad_norm,
            )

            if not math.isfinite(grad_norm):
                raise RuntimeError(
                    "Non-finite gradient norm."
                )

            optimizer_step += 1
            accumulated_tokens = 0
            micro_steps_since_update = 0

            if (
                optimizer_step
                >= confirmation[
                    "optimizer_steps"
                ]
            ):
                break

    validation_loss = evaluate(
        model,
        validation_loader,
        device,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    peak_vram_gib = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    result = {
        "candidate": candidate["name"],
        "confirmation_seed": seed,
        "learning_rate":
            candidate["learning_rate"],
        "weight_decay":
            candidate["weight_decay"],
        "warmup_ratio":
            candidate["warmup_ratio"],
        "scheduler":
            candidate["scheduler"],
        "optimizer_steps":
            optimizer_step,
        "validation_blocks":
            len(dataset["validation"]),
        "validation_loss":
            validation_loss,
        "perplexity":
            math.exp(validation_loss),
        "training_time_seconds":
            elapsed,
        "peak_vram_gib":
            peak_vram_gib,
    }

    del model
    del optimizer
    del scheduler
    del train_loader
    del validation_loader

    gc.collect()
    torch.cuda.empty_cache()

    return result


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

    dataset = load_from_disk(
        config["data"]["dataset_path"]
    )

    results = []

    print("=== HYPERPARAMETER CONFIRMATION ===")
    print(
        "confirmation seed:",
        config["confirmation"]["seed"],
    )
    print(
        "optimizer steps/candidate:",
        config["confirmation"][
            "optimizer_steps"
        ],
    )

    for candidate in (
        config["confirmation"]["candidates"]
    ):
        print(
            "\n=== CONFIRMING",
            candidate["name"],
            "===",
        )

        result = train_candidate(
            candidate=candidate,
            config=config,
            dataset=dataset,
            device=device,
        )

        results.append(result)

        print(
            "validation loss:",
            result["validation_loss"],
        )
        print(
            "perplexity:",
            result["perplexity"],
        )

    results.sort(
        key=lambda item:
        item["validation_loss"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        "\n=== CONFIRMATION RESULTS ==="
    )

    for result in results:
        print(
            f"{result['candidate']}: "
            f"val_loss="
            f"{result['validation_loss']:.9f} "
            f"ppl="
            f"{result['perplexity']:.6f} "
            f"lr={result['learning_rate']} "
            f"wd={result['weight_decay']} "
            f"warmup="
            f"{result['warmup_ratio']} "
            f"scheduler="
            f"{result['scheduler']}"
        )

    print(
        "\nsaved to:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()