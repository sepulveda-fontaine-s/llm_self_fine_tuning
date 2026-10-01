import argparse
import math
import shutil
import time
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


CONFIG_PATH = "configs/training.yaml"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest recovery checkpoint.",
    )

    parser.add_argument(
        "--max-epochs-this-run",
        type=int,
        default=None,
        help="Stop after this many epochs during this invocation.",
    )

    return parser.parse_args()


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

    return total_loss / total_prediction_tokens


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


def save_recovery_checkpoint(
    checkpoint_dir,
    model,
    tokenizer,
    optimizer,
    scheduler,
    epoch,
    global_optimizer_step,
    best_validation_loss,
    best_epoch,
    train_generator,
    config,
):
    final_dir = checkpoint_dir / "last_checkpoint"
    temp_dir = checkpoint_dir / "last_checkpoint_tmp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    model.save_pretrained(
        temp_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(temp_dir)

    torch.save(
        {
            "epoch": epoch,
            "global_optimizer_step":
                global_optimizer_step,
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
            "best_validation_loss":
                best_validation_loss,
            "best_epoch":
                best_epoch,
            "train_generator_state":
                train_generator.get_state(),
            "torch_rng_state":
                torch.get_rng_state(),
            "cuda_rng_state_all":
                torch.cuda.get_rng_state_all(),
        },
        temp_dir / "training_state.pt",
    )

    with (
        temp_dir / "training_config.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    if final_dir.exists():
        shutil.rmtree(final_dir)

    temp_dir.rename(final_dir)


def main():
    args = parse_args()

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

    training = config["training"]

    seed = training["seed"]
    batch_size = training[
        "per_device_batch_size"
    ]
    accumulation_steps = training[
        "gradient_accumulation_steps"
    ]
    learning_rate = training["learning_rate"]
    weight_decay = training["weight_decay"]
    warmup_ratio = training["warmup_ratio"]
    scheduler_name = training["scheduler"]
    max_grad_norm = training["max_grad_norm"]
    epochs = training["epochs"]

    checkpoint_dir = Path(
        config["checkpointing"]["directory"]
    )

    last_checkpoint = (
        checkpoint_dir / "last_checkpoint"
    )

    best_dir = (
        checkpoint_dir / "best_model"
    )

    device = torch.device("cuda")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = load_from_disk(
        config["data"]["dataset_path"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"]
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    start_epoch = 1
    global_optimizer_step = 0
    best_validation_loss = float("inf")
    best_epoch = None

    if args.resume:
        if not last_checkpoint.exists():
            raise FileNotFoundError(
                f"No recovery checkpoint: "
                f"{last_checkpoint}"
            )

        print(
            "=== RESUMING FROM CHECKPOINT ==="
        )
        print("checkpoint:", last_checkpoint)

        model = (
            AutoModelForCausalLM
            .from_pretrained(last_checkpoint)
        )

    else:
        if (
            checkpoint_dir.exists()
            and any(checkpoint_dir.iterdir())
        ):
            raise FileExistsError(
                f"{checkpoint_dir} already contains files. "
                "Refusing to overwrite. "
                "Use --resume if appropriate."
            )

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                config["model"]["name"]
            )
        )

    model.to(device)
    model.train()
    model.config.use_cache = False

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        collate_fn=collate_batch,
    )

    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    micro_batches_per_epoch = len(
        train_loader
    )

    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch
        / accumulation_steps
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * epochs
    )

    warmup_steps = int(
        total_optimizer_steps
        * warmup_ratio
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = create_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_optimizer_steps,
    )

    if args.resume:
        state = torch.load(
            last_checkpoint
            / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )

        optimizer.load_state_dict(
            state["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            state["scheduler_state_dict"]
        )

        start_epoch = state["epoch"] + 1

        global_optimizer_step = state[
            "global_optimizer_step"
        ]

        best_validation_loss = state[
            "best_validation_loss"
        ]

        best_epoch = state["best_epoch"]

        train_generator.set_state(
            state["train_generator_state"]
        )

        torch.set_rng_state(
            state["torch_rng_state"]
        )

        torch.cuda.set_rng_state_all(
            state["cuda_rng_state_all"]
        )

        print("completed epoch:", state["epoch"])
        print("next epoch:", start_epoch)
        print(
            "global optimizer step:",
            global_optimizer_step,
        )

    if start_epoch > epochs:
        print(
            "Training is already complete."
        )
        return

    run_end_epoch = epochs

    if args.max_epochs_this_run is not None:
        run_end_epoch = min(
            epochs,
            start_epoch
            + args.max_epochs_this_run
            - 1,
        )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(
        config["tracking"]["uri"]
    )

    mlflow.set_experiment(
        config["tracking"]["experiment_name"]
    )

    print("\n=== TRAINING PLAN ===")
    print("train blocks:", len(dataset["train"]))
    print(
        "validation blocks:",
        len(dataset["validation"]),
    )
    print(
        "micro-batches/epoch:",
        micro_batches_per_epoch,
    )
    print(
        "optimizer steps/epoch:",
        optimizer_steps_per_epoch,
    )
    print(
        "total optimizer steps:",
        total_optimizer_steps,
    )
    print("warmup steps:", warmup_steps)
    print("start epoch:", start_epoch)
    print("end epoch this run:", run_end_epoch)
    print("learning rate:", learning_rate)
    print("weight decay:", weight_decay)
    print("warmup ratio:", warmup_ratio)
    print("scheduler:", scheduler_name)

    optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model":
                    config["model"]["name"],
                "block_size":
                    config["data"][
                        "block_size"
                    ],
                "learning_rate":
                    learning_rate,
                "weight_decay":
                    weight_decay,
                "warmup_ratio":
                    warmup_ratio,
                "warmup_steps":
                    warmup_steps,
                "scheduler":
                    scheduler_name,
                "batch_size":
                    batch_size,
                "gradient_accumulation":
                    accumulation_steps,
                "effective_blocks":
                    batch_size
                    * accumulation_steps,
                "epochs":
                    epochs,
                "seed":
                    seed,
                "precision":
                    "bf16_autocast",
            }
        )

        for epoch in range(
            start_epoch,
            run_end_epoch + 1,
        ):
            epoch_loss_sum = 0.0
            epoch_prediction_tokens = 0

            accumulated_tokens = 0
            micro_steps_since_update = 0

            for batch_index, batch in enumerate(
                train_loader,
                start=1,
            ):
                batch = {
                    key: value.to(device)
                    for key, value
                    in batch.items()
                }

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    outputs = model(**batch)
                    loss = outputs.loss

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite loss: "
                        f"{loss.item()}"
                    )

                prediction_tokens = int(
                    batch["attention_mask"][:, 1:]
                    .sum()
                    .item()
                )

                (
                    loss
                    * prediction_tokens
                ).backward()

                epoch_loss_sum += (
                    loss.item()
                    * prediction_tokens
                )

                epoch_prediction_tokens += (
                    prediction_tokens
                )

                accumulated_tokens += (
                    prediction_tokens
                )

                micro_steps_since_update += 1

                accumulation_boundary = (
                    micro_steps_since_update
                    == accumulation_steps
                )

                last_batch = (
                    batch_index
                    == micro_batches_per_epoch
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
                        max_grad_norm=(
                            max_grad_norm
                        ),
                    )

                    if not math.isfinite(
                        grad_norm
                    ):
                        raise RuntimeError(
                            "Non-finite "
                            "gradient norm."
                        )

                    global_optimizer_step += 1

                    mlflow.log_metrics(
                        {
                            "learning_rate":
                                scheduler
                                .get_last_lr()[0],
                            "gradient_norm":
                                grad_norm,
                        },
                        step=(
                            global_optimizer_step
                        ),
                    )

                    accumulated_tokens = 0
                    micro_steps_since_update = 0

            train_loss = (
                epoch_loss_sum
                / epoch_prediction_tokens
            )

            validation_loss = evaluate(
                model,
                validation_loader,
                device,
            )

            validation_perplexity = math.exp(
                validation_loss
            )

            mlflow.log_metrics(
                {
                    "epoch_train_loss":
                        train_loss,
                    "epoch_validation_loss":
                        validation_loss,
                    "epoch_validation_perplexity":
                        validation_perplexity,
                },
                step=epoch,
            )

            print(
                f"\n=== EPOCH {epoch} ==="
            )
            print("train loss:", train_loss)
            print(
                "validation loss:",
                validation_loss,
            )
            print(
                "validation perplexity:",
                validation_perplexity,
            )

            if (
                validation_loss
                < best_validation_loss
            ):
                best_validation_loss = (
                    validation_loss
                )
                best_epoch = epoch

                if best_dir.exists():
                    shutil.rmtree(best_dir)

                model.save_pretrained(
                    best_dir,
                    safe_serialization=True,
                )

                tokenizer.save_pretrained(
                    best_dir
                )

                with (
                    best_dir
                    / "training_config.yaml"
                ).open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    yaml.safe_dump(
                        config,
                        file,
                        sort_keys=False,
                    )

                print(
                    "new best checkpoint:",
                    best_dir,
                )

            save_recovery_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_optimizer_step=(
                    global_optimizer_step
                ),
                best_validation_loss=(
                    best_validation_loss
                ),
                best_epoch=best_epoch,
                train_generator=(
                    train_generator
                ),
                config=config,
            )

            print(
                "latest recovery checkpoint:",
                last_checkpoint,
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        peak_vram_gib = (
            torch.cuda
            .max_memory_allocated()
            / (1024 ** 3)
        )

        mlflow.log_metrics(
            {
                "best_validation_loss":
                    best_validation_loss,
                "training_time_seconds_this_run":
                    elapsed,
                "peak_vram_gib":
                    peak_vram_gib,
            }
        )

        print(
            "\n=== RUN COMPLETE ==="
        )
        print(
            "last completed epoch:",
            run_end_epoch,
        )
        print("best epoch:", best_epoch)
        print(
            "best validation loss:",
            best_validation_loss,
        )
        print(
            "training time seconds:",
            elapsed,
        )
        print(
            "peak VRAM GiB:",
            peak_vram_gib,
        )


if __name__ == "__main__":
    main()