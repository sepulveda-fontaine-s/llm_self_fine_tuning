from pathlib import Path
from collections import defaultdict
import hashlib
import json
import random

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


SEED = 42

OUTPUT_PATH = Path("data/processed/self_supervised_squad_v3")
MANIFEST_PATH = Path("data/manifests/self_supervised_squad_v3.json")


def is_answerable(example):
    texts = example["answers"]["text"]
    return len(texts) > 0 and any(text.strip() for text in texts)


def build_v1_train(raw_train):
    answerable_indices = []
    unanswerable_indices = []

    for index, example in enumerate(raw_train):
        if is_answerable(example):
            answerable_indices.append(index)
        else:
            unanswerable_indices.append(index)

    rng = random.Random(SEED)

    rng.shuffle(answerable_indices)
    rng.shuffle(unanswerable_indices)

    selected_indices = (
        answerable_indices[:20_000]
        + unanswerable_indices[:10_000]
    )

    rng.shuffle(selected_indices)

    return raw_train.select(selected_indices)


def build_v3_train(raw_train):
    train_v1 = build_v1_train(raw_train)

    answerable_idx = [
        i
        for i, example in enumerate(train_v1)
        if is_answerable(example)
    ]

    unanswerable_idx = [
        i
        for i, example in enumerate(train_v1)
        if not is_answerable(example)
    ]

    v1_answerable = (
        train_v1
        .select(answerable_idx)
        .shuffle(seed=SEED)
        .select(range(18_000))
    )

    v1_unanswerable = train_v1.select(unanswerable_idx)

    existing_ids = set(train_v1["id"])

    candidate_extra_unanswerable_idx = [
        i
        for i, example in enumerate(raw_train)
        if example["id"] not in existing_ids
        and not is_answerable(example)
    ]

    extra_unanswerable = (
        raw_train
        .select(candidate_extra_unanswerable_idx)
        .shuffle(seed=SEED)
        .select(range(2_000))
    )

    return concatenate_datasets(
        [
            v1_answerable,
            v1_unanswerable,
            extra_unanswerable,
        ]
    ).shuffle(seed=SEED)


def split_validation_by_context(validation_dataset):
    context_groups = defaultdict(list)

    for index, example in enumerate(validation_dataset):
        context_groups[example["context"]].append(index)

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


def unique_context_dataset(dataset):
    seen = set()
    texts = []

    for context in dataset["context"]:
        text = context.strip()

        if text and text not in seen:
            seen.add(text)
            texts.append(text)

    return Dataset.from_dict({"text": texts})


def sha256_texts(dataset):
    digest = hashlib.sha256()

    for text in dataset["text"]:
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")

    return digest.hexdigest()


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. Refusing to overwrite."
        )

    print("=== LOADING SQUAD V2 ===")
    raw = load_dataset("rajpurkar/squad_v2")

    print("=== RECONSTRUCTING V3 TRAIN ===")
    train_v3 = build_v3_train(raw["train"])

    print("=== RECONSTRUCTING SFT VALIDATION / TEST ===")
    validation_qa, test_qa = split_validation_by_context(
        raw["validation"]
    )

    train_lm = unique_context_dataset(train_v3)
    validation_lm = unique_context_dataset(validation_qa)

    train_contexts = set(train_lm["text"])
    validation_contexts = set(validation_lm["text"])
    test_contexts = {
        text.strip()
        for text in test_qa["context"]
        if text.strip()
    }

    assert len(train_v3) == 30_000
    assert len(train_lm) == 14_639

    assert not (train_contexts & validation_contexts)
    assert not (train_contexts & test_contexts)
    assert not (validation_contexts & test_contexts)

    corpus = DatasetDict(
        {
            "train": train_lm,
            "validation": validation_lm,
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    corpus.save_to_disk(str(OUTPUT_PATH))

    manifest = {
        "source_dataset": "rajpurkar/squad_v2",
        "seed": SEED,
        "model": "Qwen/Qwen2.5-0.5B",
        "objective": "causal_language_modeling",
        "training_method": "full-parameter self-supervised fine-tuning",
        "v3_source_rows": len(train_v3),
        "train_unique_contexts": len(train_lm),
        "validation_qa_rows": len(validation_qa),
        "validation_unique_contexts": len(validation_lm),
        "reserved_test_qa_rows": len(test_qa),
        "reserved_test_unique_contexts": len(test_contexts),
        "train_validation_overlap": len(
            train_contexts & validation_contexts
        ),
        "train_test_overlap": len(
            train_contexts & test_contexts
        ),
        "validation_test_overlap": len(
            validation_contexts & test_contexts
        ),
        "train_sha256": sha256_texts(train_lm),
        "validation_sha256": sha256_texts(validation_lm),
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("\n=== FINAL CORPUS ===")
    for key, value in manifest.items():
        print(f"{key}: {value}")

    print("\nsaved corpus:", OUTPUT_PATH)
    print("saved manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    main()