import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


BENCHMARK_PATH = Path(
    "data/processed/retrieval_benchmark.json"
)

OUTPUT_DIR = Path(
    "results/deployment/retrieval"
)

MODEL_NAME = "intfloat/e5-small-v2"

BATCH_SIZE = 128
MAX_LENGTH = 512


def average_pool(
    last_hidden_states,
    attention_mask,
):
    last_hidden = (
        last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(),
            0.0,
        )
    )

    return (
        last_hidden.sum(dim=1)
        / attention_mask.sum(
            dim=1
        )[..., None]
    )


def encode_documents(
    texts,
    tokenizer,
    model,
    device,
):
    embeddings = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):
        batch = [
            "passage: " + text
            for text in texts[
                start:start + BATCH_SIZE
            ]
        ]

        inputs = tokenizer(
            batch,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(
                    **inputs
                )

                pooled = average_pool(
                    outputs.last_hidden_state,
                    inputs[
                        "attention_mask"
                    ],
                )

                pooled = F.normalize(
                    pooled.float(),
                    p=2,
                    dim=1,
                )

        embeddings.append(
            pooled.cpu()
        )

        processed = min(
            start + BATCH_SIZE,
            len(texts),
        )

        print(
            f"encoded "
            f"{processed}/"
            f"{len(texts)}"
        )

    return torch.cat(
        embeddings,
        dim=0,
    )


def main():
    print(
        "=== BUILD DEPLOYMENT RETRIEVAL INDEX ==="
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    with BENCHMARK_PATH.open(
        encoding="utf-8"
    ) as file:
        benchmark = json.load(file)

    documents = benchmark[
        "documents"
    ]

    document_texts = [
        row["text"]
        for row in documents
    ]

    print(
        "documents:",
        len(documents),
    )

    print(
        "embedding model:",
        MODEL_NAME,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModel
        .from_pretrained(
            MODEL_NAME
        )
    )

    device = torch.device(
        "cuda"
    )

    model.to(device)
    model.eval()

    embeddings = encode_documents(
        document_texts,
        tokenizer,
        model,
        device,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        embeddings,
        OUTPUT_DIR
        / "dense_embeddings.pt",
    )

    metadata = {
        "embedding_model":
            MODEL_NAME,

        "pooling":
            "attention-mask mean pooling",

        "normalization":
            "L2",

        "passage_prefix":
            "passage: ",

        "max_length":
            MAX_LENGTH,

        "documents":
            documents,
    }

    with (
        OUTPUT_DIR
        / "documents.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "embedding shape:",
        tuple(
            embeddings.shape
        ),
    )

    print(
        "saved embeddings:",
        OUTPUT_DIR
        / "dense_embeddings.pt",
    )

    print(
        "saved metadata:",
        OUTPUT_DIR
        / "documents.json",
    )

    print(
        "=== COMPLETE ==="
    )


if __name__ == "__main__":
    main()