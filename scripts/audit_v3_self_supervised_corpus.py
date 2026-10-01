import random
from collections import defaultdict

from datasets import load_dataset


SEED = 42

TRAIN_ANSWERABLE = 20_000
TRAIN_UNANSWERABLE = 10_000


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
        answerable_indices[:TRAIN_ANSWERABLE]
        + unanswerable_indices[:TRAIN_UNANSWERABLE]
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

    from datasets import concatenate_datasets

    return concatenate_datasets(
        [
            v1_answerable,
            v1_unanswerable,
            extra_unanswerable,
        ]
    ).shuffle(seed=SEED)


def main():
    raw = load_dataset("rajpurkar/squad_v2")

    train_v3 = build_v3_train(raw["train"])

    contexts = [text.strip() for text in train_v3["context"]]
    unique_contexts = list(dict.fromkeys(contexts))

    validation_contexts = set(raw["validation"]["context"])
    train_unique_set = set(unique_contexts)

    print("=== V3 SELF-SUPERVISED CORPUS AUDIT ===")
    print("v3 rows:", len(train_v3))
    print("unique contexts:", len(unique_contexts))
    print("duplicate context rows:", len(contexts) - len(unique_contexts))

    print(
        "overlap with official validation:",
        len(train_unique_set & validation_contexts),
    )

    print("\n=== SAMPLE CONTEXT ===")
    print(unique_contexts[0])


if __name__ == "__main__":
    main()