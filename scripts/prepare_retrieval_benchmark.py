import json
import random
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from datasets import load_dataset


SEED = 42

OUTPUT_PATH = Path(
    "data/processed/retrieval_benchmark.json"
)

MANIFEST_PATH = Path(
    "data/manifests/retrieval_benchmark.json"
)


def context_id(text):
    return sha256(
        text.strip().encode("utf-8")
    ).hexdigest()


def split_validation_by_context(dataset):
    groups = defaultdict(list)

    for index, example in enumerate(dataset):
        groups[example["context"]].append(index)

    contexts = list(groups.keys())

    rng = random.Random(SEED)
    rng.shuffle(contexts)

    validation_indices = []
    test_indices = []

    for context in contexts:
        indices = groups[context]

        if len(validation_indices) <= len(test_indices):
            validation_indices.extend(indices)
        else:
            test_indices.extend(indices)

    return (
        dataset.select(validation_indices),
        dataset.select(test_indices),
    )


def main():
    print("=== PREPARE RETRIEVAL BENCHMARK ===")

    raw = load_dataset(
        "rajpurkar/squad_v2"
    )

    _, test = split_validation_by_context(
        raw["validation"]
    )

    assert len(test) == 5940

    documents = {}

    for split_name in (
        "train",
        "validation",
    ):
        for context in raw[split_name]["context"]:
            text = context.strip()
            doc_id = context_id(text)

            if doc_id not in documents:
                documents[doc_id] = text

    document_rows = [
        {
            "document_id": doc_id,
            "text": text,
        }
        for doc_id, text in documents.items()
    ]

    queries = []

    for example in test:
        context = example["context"].strip()

        gold_document_id = context_id(
            context
        )

        assert gold_document_id in documents

        queries.append(
            {
                "query_id": example["id"],
                "question":
                    example["question"],
                "gold_document_id":
                    gold_document_id,
                "answerable":
                    len(
                        example[
                            "answers"
                        ]["text"]
                    ) > 0,
                "gold_answers":
                    example[
                        "answers"
                    ]["text"],
            }
        )

    output = {
        "documents": document_rows,
        "queries": queries,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
        )

    unique_gold_documents = {
        row["gold_document_id"]
        for row in queries
    }

    manifest = {
        "source":
            "rajpurkar/squad_v2",
        "seed":
            SEED,
        "documents":
            len(document_rows),
        "queries":
            len(queries),
        "unique_gold_documents":
            len(unique_gold_documents),
        "answerable_queries":
            sum(
                row["answerable"]
                for row in queries
            ),
        "unanswerable_queries":
            sum(
                not row["answerable"]
                for row in queries
            ),
        "evaluation_metrics": [
            "Recall@1",
            "Recall@3",
            "Recall@5",
            "Recall@10",
            "MRR",
        ],
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

    print("\n=== BENCHMARK SUMMARY ===")

    for key, value in manifest.items():
        print(f"{key}: {value}")

    print(
        "\nsaved:",
        OUTPUT_PATH,
    )

    print(
        "manifest:",
        MANIFEST_PATH,
    )

    print("\n=== COMPLETE ===")


if __name__ == "__main__":
    main()