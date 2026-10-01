import json
import math
import re
import statistics
import string
import subprocess
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


CONFIG_PATH = Path("configs/training.yaml")

ADAPTED_MODEL_PATH = (
    "results/checkpoints/"
    "self_supervised_v3_refined/best_model"
)

RETRIEVAL_DIR = Path(
    "results/deployment/retrieval"
)

DOCUMENTS_PATH = (
    RETRIEVAL_DIR / "documents.json"
)

DENSE_EMBEDDINGS_PATH = (
    RETRIEVAL_DIR / "dense_embeddings.pt"
)

NLI_MODEL_NAME = (
    "FacebookAI/roberta-large-mnli"
)

RRF_K = 60
RETRIEVAL_POOL = 10
NLI_MAX_LENGTH = 512

SYSTEM_MESSAGE = (
    "Answer the question using only the provided context. "
    "Return only the answer supported by the context. "
    "If the context does not contain the answer, respond exactly: "
    "I don't know."
)


def utc_now():
    return datetime.now(
        timezone.utc
    )


def tokenize_bm25(text):
    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def average_pool(
    last_hidden_states,
    attention_mask,
):
    masked = (
        last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(),
            0.0,
        )
    )

    return (
        masked.sum(dim=1)
        / attention_mask.sum(
            dim=1
        )[..., None]
    )


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            value,
        )

    def remove_punctuation(value):
        return "".join(
            char
            for char in value
            if char not in string.punctuation
        )

    return " ".join(
        remove_articles(
            remove_punctuation(
                text.lower()
            )
        ).split()
    )


def lexical_support(
    prediction,
    context,
):
    prediction = normalize_answer(
        prediction
    )

    context = normalize_answer(
        context
    )

    if not prediction:
        return False

    if prediction == normalize_answer(
        "I don't know."
    ):
        return False

    return prediction in context


class MetricsStore:
    def __init__(
        self,
        rolling_window_size=500,
    ):
        self.started_at = utc_now()

        self.lock = threading.Lock()

        self.rolling_window_size = (
            rolling_window_size
        )

        self.requests_total = 0
        self.retrieval_requests = 0
        self.answer_requests = 0
        self.errors_total = 0

        self.request_timestamps = deque(
            maxlen=rolling_window_size
        )

        self.retrieval_latencies = deque(
            maxlen=rolling_window_size
        )

        self.generation_latencies = deque(
            maxlen=rolling_window_size
        )

        self.grounding_latencies = deque(
            maxlen=rolling_window_size
        )

        self.total_latencies = deque(
            maxlen=rolling_window_size
        )

        self.history = deque(
            maxlen=120
        )

    def record_retrieval(
        self,
        latency_ms,
        external_request=True,
    ):
        with self.lock:
            if external_request:
                self.requests_total += 1
                self.retrieval_requests += 1
                self.request_timestamps.append(
                    time.time()
                )

            self.retrieval_latencies.append(
                latency_ms
            )

    def record_answer(
        self,
        generation_ms,
        grounding_ms,
        total_ms,
    ):
        with self.lock:
            self.requests_total += 1
            self.answer_requests += 1

            self.request_timestamps.append(
                time.time()
            )

            self.generation_latencies.append(
                generation_ms
            )

            self.grounding_latencies.append(
                grounding_ms
            )

            self.total_latencies.append(
                total_ms
            )

    def record_error(self):
        with self.lock:
            self.errors_total += 1

    @staticmethod
    def percentile(
        values,
        percentile,
    ):
        if not values:
            return None

        ordered = sorted(values)

        index = (
            percentile
            / 100.0
            * (len(ordered) - 1)
        )

        lower = math.floor(index)
        upper = math.ceil(index)

        if lower == upper:
            return float(
                ordered[lower]
            )

        fraction = index - lower

        return float(
            ordered[lower]
            + (
                ordered[upper]
                - ordered[lower]
            )
            * fraction
        )

    @classmethod
    def latency_summary(
        cls,
        values,
    ):
        values = list(values)

        if not values:
            return {
                "last_ms": None,
                "average_ms": None,
                "p50_ms": None,
                "p95_ms": None,
            }

        return {
            "last_ms":
                float(values[-1]),

            "average_ms":
                float(
                    statistics.fmean(
                        values
                    )
                ),

            "p50_ms":
                cls.percentile(
                    values,
                    50,
                ),

            "p95_ms":
                cls.percentile(
                    values,
                    95,
                ),
        }

    def requests_per_minute(self):
        now = time.time()
        threshold = now - 60.0

        return float(
            sum(
                timestamp >= threshold
                for timestamp
                in self.request_timestamps
            )
        )

    def snapshot(
        self,
        gpu_metrics,
    ):
        with self.lock:
            now = utc_now()

            uptime = (
                now
                - self.started_at
            ).total_seconds()

            error_rate = (
                self.errors_total
                / self.requests_total
                if self.requests_total
                else 0.0
            )

            rpm = (
                self.requests_per_minute()
            )

            history_point = {
                "timestamp": now,
                "requests_per_minute":
                    rpm,

                "end_to_end_latency_ms":
                    (
                        self.total_latencies[-1]
                        if self.total_latencies
                        else None
                    ),

                "retrieval_latency_ms":
                    (
                        self.retrieval_latencies[-1]
                        if self.retrieval_latencies
                        else None
                    ),

                "generation_latency_ms":
                    (
                        self.generation_latencies[-1]
                        if self.generation_latencies
                        else None
                    ),

                "grounding_latency_ms":
                    (
                        self.grounding_latencies[-1]
                        if self.grounding_latencies
                        else None
                    ),

                "gpu_memory_usage_percent":
                    gpu_metrics.get(
                        "memory_usage_percent"
                    ),

                "gpu_utilization_percent":
                    gpu_metrics.get(
                        "utilization_percent"
                    ),
            }

            self.history.append(
                history_point
            )

            return {
                "server_started_at":
                    self.started_at,

                "current_time":
                    now,

                "uptime_seconds":
                    uptime,

                "requests_total":
                    self.requests_total,

                "retrieval_requests":
                    self.retrieval_requests,

                "answer_requests":
                    self.answer_requests,

                "errors_total":
                    self.errors_total,

                "error_rate":
                    error_rate,

                "requests_per_minute":
                    rpm,

                "retrieval_latency":
                    self.latency_summary(
                        self.retrieval_latencies
                    ),

                "generation_latency":
                    self.latency_summary(
                        self.generation_latencies
                    ),

                "grounding_latency":
                    self.latency_summary(
                        self.grounding_latencies
                    ),

                "end_to_end_latency":
                    self.latency_summary(
                        self.total_latencies
                    ),

                "gpu":
                    gpu_metrics,

                "rolling_window_size":
                    self.rolling_window_size,

                "history":
                    list(self.history),
            }


class DeploymentService:
    def __init__(self):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.loaded = False

        self.metrics = MetricsStore()

        self.base_model = None
        self.adapted_model = None

        self.generation_tokenizer = None

        self.embedding_model = None
        self.embedding_tokenizer = None

        self.nli_model = None
        self.nli_tokenizer = None

        self.documents = []
        self.document_ids = []
        self.document_texts = []

        self.dense_embeddings = None

        self.bm25_postings = None
        self.bm25_doc_lengths = None
        self.bm25_avg_doc_length = None

        self.base_model_name = None
        self.embedding_model_name = None

        self.generation_config = None

        self.nli_label_ids = None

    def load(self):
        if self.loaded:
            return

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for deployment."
            )

        print(
            "=== LOADING DEPLOYMENT SERVICE ==="
        )

        with CONFIG_PATH.open(
            encoding="utf-8"
        ) as file:
            config = yaml.safe_load(file)

        self.base_model_name = (
            config["model"]["name"]
        )

        self.generation_config = (
            config["generation"]
        )

        self._load_retrieval_assets()

        self._build_bm25_index()

        self._load_embedding_model()

        self._load_generation_models()

        self._load_nli_model()

        self.loaded = True

        print(
            "=== DEPLOYMENT SERVICE READY ==="
        )

    def _load_retrieval_assets(self):
        print(
            "loading retrieval assets..."
        )

        with DOCUMENTS_PATH.open(
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

        self.embedding_model_name = (
            metadata[
                "embedding_model"
            ]
        )

        self.documents = metadata[
            "documents"
        ]

        self.document_ids = [
            row["document_id"]
            for row in self.documents
        ]

        self.document_texts = [
            row["text"]
            for row in self.documents
        ]

        embeddings = torch.load(
            DENSE_EMBEDDINGS_PATH,
            map_location="cpu",
            weights_only=True,
        )

        self.dense_embeddings = (
            embeddings
            .float()
            .to(self.device)
        )

        print(
            "documents:",
            len(self.documents),
        )

        print(
            "dense embeddings:",
            tuple(
                self.dense_embeddings.shape
            ),
        )

    def _build_bm25_index(self):
        print(
            "building BM25 index..."
        )

        postings = defaultdict(
            list
        )

        doc_lengths = []

        for doc_index, text in enumerate(
            self.document_texts
        ):
            tokens = tokenize_bm25(
                text
            )

            doc_lengths.append(
                len(tokens)
            )

            frequencies = Counter(
                tokens
            )

            for term, tf in (
                frequencies.items()
            ):
                postings[term].append(
                    (
                        doc_index,
                        tf,
                    )
                )

        self.bm25_postings = postings

        self.bm25_doc_lengths = (
            doc_lengths
        )

        self.bm25_avg_doc_length = (
            sum(doc_lengths)
            / len(doc_lengths)
        )

        print(
            "BM25 terms:",
            len(postings),
        )

    def _load_embedding_model(self):
        print(
            "loading embedding model:",
            self.embedding_model_name,
        )

        self.embedding_tokenizer = (
            AutoTokenizer.from_pretrained(
                self.embedding_model_name
            )
        )

        self.embedding_model = (
            AutoModel.from_pretrained(
                self.embedding_model_name
            )
            .to(self.device)
        )

        self.embedding_model.eval()

    def _load_generation_models(self):
        print(
            "loading generation tokenizer..."
        )

        self.generation_tokenizer = (
            AutoTokenizer.from_pretrained(
                self.base_model_name
            )
        )

        self.generation_tokenizer.padding_side = (
            "left"
        )

        print(
            "loading base model..."
        )

        self.base_model = (
            AutoModelForCausalLM
            .from_pretrained(
                self.base_model_name,
                dtype=torch.bfloat16,
            )
            .to(self.device)
        )

        self.base_model.eval()
        self.base_model.config.use_cache = True

        print(
            "loading self-supervised model..."
        )

        self.adapted_model = (
            AutoModelForCausalLM
            .from_pretrained(
                ADAPTED_MODEL_PATH,
                dtype=torch.bfloat16,
            )
            .to(self.device)
        )

        self.adapted_model.eval()
        self.adapted_model.config.use_cache = (
            True
        )

    def _load_nli_model(self):
        print(
            "loading NLI model:",
            NLI_MODEL_NAME,
        )

        self.nli_tokenizer = (
            AutoTokenizer.from_pretrained(
                NLI_MODEL_NAME
            )
        )

        self.nli_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                NLI_MODEL_NAME
            )
            .to(self.device)
        )

        self.nli_model.eval()

        self.nli_label_ids = (
            self._resolve_nli_labels()
        )

        print(
            "NLI labels:",
            self.nli_label_ids,
        )

    def _resolve_nli_labels(self):
        labels = {}

        for index, label in (
            self.nli_model.config
            .id2label.items()
        ):
            value = label.lower()

            if "entail" in value:
                labels[
                    "entailment"
                ] = int(index)

            elif "neutral" in value:
                labels[
                    "neutral"
                ] = int(index)

            elif "contrad" in value:
                labels[
                    "contradiction"
                ] = int(index)

        required = {
            "entailment",
            "neutral",
            "contradiction",
        }

        if set(labels) != required:
            raise RuntimeError(
                "Could not resolve NLI labels: "
                f"{self.nli_model.config.id2label}"
            )

        return labels

    def _bm25_search(
        self,
        query,
        top_k=RETRIEVAL_POOL,
    ):
        query_tokens = tokenize_bm25(
            query
        )

        scores = [
            0.0
            for _ in self.documents
        ]

        n_documents = len(
            self.documents
        )

        k1 = 1.5
        b = 0.75

        for term in query_tokens:
            postings = (
                self.bm25_postings.get(
                    term
                )
            )

            if not postings:
                continue

            document_frequency = len(
                postings
            )

            idf = math.log(
                1.0
                + (
                    n_documents
                    - document_frequency
                    + 0.5
                )
                / (
                    document_frequency
                    + 0.5
                )
            )

            for doc_index, tf in postings:
                doc_length = (
                    self.bm25_doc_lengths[
                        doc_index
                    ]
                )

                denominator = (
                    tf
                    + k1
                    * (
                        1.0
                        - b
                        + b
                        * doc_length
                        / self.bm25_avg_doc_length
                    )
                )

                scores[
                    doc_index
                ] += (
                    idf
                    * tf
                    * (k1 + 1.0)
                    / denominator
                )

        ranking = sorted(
            range(n_documents),
            key=lambda index: (
                -scores[index],
                self.document_ids[index],
            ),
        )[:top_k]

        return [
            {
                "index": index,
                "document_id":
                    self.document_ids[
                        index
                    ],
                "score":
                    float(scores[index]),
                "rank":
                    rank,
            }
            for rank, index
            in enumerate(
                ranking,
                start=1,
            )
        ]

    def _dense_search(
        self,
        query,
        top_k=RETRIEVAL_POOL,
    ):
        inputs = (
            self.embedding_tokenizer(
                [
                    "query: "
                    + query
                ],
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():
            outputs = (
                self.embedding_model(
                    **inputs
                )
            )

            pooled = average_pool(
                outputs.last_hidden_state,
                inputs[
                    "attention_mask"
                ],
            )

            query_embedding = (
                F.normalize(
                    pooled.float(),
                    p=2,
                    dim=1,
                )
            )

            similarities = (
                query_embedding
                @ self.dense_embeddings.T
            ).squeeze(0)

        values, indices = torch.topk(
            similarities,
            k=top_k,
        )

        results = []

        for rank, (
            score,
            index,
        ) in enumerate(
            zip(
                values.tolist(),
                indices.tolist(),
            ),
            start=1,
        ):
            results.append(
                {
                    "index":
                        index,

                    "document_id":
                        self.document_ids[
                            index
                        ],

                    "score":
                        float(score),

                    "rank":
                        rank,
                }
            )

        return results

    def retrieve(
        self,
        query,
        top_k=5,
        external_request=True,
    ):
        if not self.loaded:
            raise RuntimeError(
                "Service is not loaded."
            )

        start = time.perf_counter()

        bm25 = self._bm25_search(
            query
        )

        dense = self._dense_search(
            query
        )

        bm25_by_id = {
            row["document_id"]:
                row
            for row in bm25
        }

        dense_by_id = {
            row["document_id"]:
                row
            for row in dense
        }

        candidate_ids = (
            set(bm25_by_id)
            | set(dense_by_id)
        )

        fused = []

        for document_id in (
            candidate_ids
        ):
            score = 0.0

            bm25_row = (
                bm25_by_id.get(
                    document_id
                )
            )

            dense_row = (
                dense_by_id.get(
                    document_id
                )
            )

            if bm25_row:
                score += (
                    1.0
                    / (
                        RRF_K
                        + bm25_row[
                            "rank"
                        ]
                    )
                )

            if dense_row:
                score += (
                    1.0
                    / (
                        RRF_K
                        + dense_row[
                            "rank"
                        ]
                    )
                )

            document_index = (
                bm25_row["index"]
                if bm25_row
                else dense_row["index"]
            )

            fused.append(
                {
                    "document_id":
                        document_id,

                    "document_index":
                        document_index,

                    "rrf_score":
                        float(score),

                    "bm25_rank":
                        (
                            bm25_row["rank"]
                            if bm25_row
                            else None
                        ),

                    "dense_rank":
                        (
                            dense_row["rank"]
                            if dense_row
                            else None
                        ),

                    "bm25_score":
                        (
                            bm25_row["score"]
                            if bm25_row
                            else None
                        ),

                    "dense_score":
                        (
                            dense_row["score"]
                            if dense_row
                            else None
                        ),
                }
            )

        fused.sort(
            key=lambda row: (
                -row["rrf_score"],
                row["document_id"],
            )
        )

        fused = fused[
            :top_k
        ]

        documents = []

        for rank, row in enumerate(
            fused,
            start=1,
        ):
            documents.append(
                {
                    "rank":
                        rank,

                    "document_id":
                        row[
                            "document_id"
                        ],

                    "text":
                        self.document_texts[
                            row[
                                "document_index"
                            ]
                        ],

                    "rrf_score":
                        row[
                            "rrf_score"
                        ],

                    "bm25_rank":
                        row[
                            "bm25_rank"
                        ],

                    "dense_rank":
                        row[
                            "dense_rank"
                        ],

                    "bm25_score":
                        row[
                            "bm25_score"
                        ],

                    "dense_score":
                        row[
                            "dense_score"
                        ],
                }
            )

        torch.cuda.synchronize()

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        self.metrics.record_retrieval(
            latency_ms,
            external_request=(
                external_request
            ),
        )

        return {
            "request_id":
                str(uuid.uuid4()),

            "query":
                query,

            "method":
                "BM25 + E5 + RRF",

            "top_k":
                top_k,

            "retrieval_latency_ms":
                latency_ms,

            "documents":
                documents,
        }

    def _generate(
        self,
        question,
        context,
        model_name,
    ):
        if model_name == "base":
            model = self.base_model

        elif model_name == (
            "self_supervised"
        ):
            model = self.adapted_model

        else:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

        messages = [
            {
                "role": "system",
                "content":
                    SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content":
                    (
                        f"Context:\n"
                        f"{context}\n\n"
                        f"Question:\n"
                        f"{question}"
                    ),
            },
        ]

        prompt = (
            self.generation_tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = (
            self.generation_tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }

        input_length = (
            inputs[
                "input_ids"
            ].shape[1]
        )

        eos_token_ids = [
            self.generation_tokenizer
            .eos_token_id,

            self.generation_tokenizer
            .convert_tokens_to_ids(
                "<|im_end|>"
            ),
        ]

        eos_token_ids = [
            token_id
            for token_id
            in eos_token_ids
            if token_id is not None
            and token_id >= 0
        ]

        torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,

                max_new_tokens=(
                    self.generation_config[
                        "max_new_tokens"
                    ]
                ),

                do_sample=(
                    self.generation_config[
                        "do_sample"
                    ]
                ),

                repetition_penalty=(
                    self.generation_config[
                        "repetition_penalty"
                    ]
                ),

                eos_token_id=
                    eos_token_ids,

                pad_token_id=(
                    self.generation_tokenizer
                    .pad_token_id
                ),
            )

        torch.cuda.synchronize()

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        generated = outputs[
            :,
            input_length:
        ]

        answer = (
            self.generation_tokenizer
            .batch_decode(
                generated,
                skip_special_tokens=True,
            )[0]
            .strip()
        )

        return (
            answer,
            latency_ms,
        )

    def _ground(
        self,
        context,
        answer,
    ):
        torch.cuda.synchronize()

        start = time.perf_counter()

        inputs = self.nli_tokenizer(
            context,
            answer,
            max_length=NLI_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():
            logits = (
                self.nli_model(
                    **inputs
                ).logits
            )

        probabilities = (
            torch.softmax(
                logits.float(),
                dim=-1,
            )[0]
        )

        torch.cuda.synchronize()

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        entailment = float(
            probabilities[
                self.nli_label_ids[
                    "entailment"
                ]
            ].item()
        )

        neutral = float(
            probabilities[
                self.nli_label_ids[
                    "neutral"
                ]
            ].item()
        )

        contradiction = float(
            probabilities[
                self.nli_label_ids[
                    "contradiction"
                ]
            ].item()
        )

        scores = {
            "entailment":
                entailment,
            "neutral":
                neutral,
            "contradiction":
                contradiction,
        }

        label = max(
            scores,
            key=scores.get,
        )

        return (
            {
                "lexical_support":
                    lexical_support(
                        answer,
                        context,
                    ),

                "nli_label":
                    label,

                "entailment_probability":
                    entailment,

                "neutral_probability":
                    neutral,

                "contradiction_probability":
                    contradiction,

                "hallucination_proxy":
                    label
                    != "entailment",
            },
            latency_ms,
        )

    def answer(
        self,
        question,
        model_name,
    ):
        total_start = (
            time.perf_counter()
        )

        retrieval = self.retrieve(
            question,
            top_k=1,
            external_request=False,
        )

        document = retrieval[
            "documents"
        ][0]

        answer, generation_ms = (
            self._generate(
                question,
                document["text"],
                model_name,
            )
        )

        grounding, grounding_ms = (
            self._ground(
                document["text"],
                answer,
            )
        )

        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        self.metrics.record_answer(
            generation_ms,
            grounding_ms,
            total_ms,
        )

        return {
            "request_id":
                str(uuid.uuid4()),

            "timestamp":
                utc_now(),

            "question":
                question,

            "model":
                model_name,

            "answer":
                answer,

            "retrieved_document":
                document,

            "grounding":
                grounding,

            "trace": {
                "retrieval_latency_ms":
                    retrieval[
                        "retrieval_latency_ms"
                    ],

                "generation_latency_ms":
                    generation_ms,

                "grounding_latency_ms":
                    grounding_ms,

                "total_latency_ms":
                    total_ms,
            },
        }

    def gpu_metrics(self):
        if not torch.cuda.is_available():
            return {
                "available": False,
                "device_name": None,
                "memory_allocated_gib": None,
                "memory_reserved_gib": None,
                "memory_total_gib": None,
                "memory_usage_percent": None,
                "utilization_percent": None,
            }

        gib = 1024 ** 3

        allocated = (
            torch.cuda.memory_allocated()
            / gib
        )

        reserved = (
            torch.cuda.memory_reserved()
            / gib
        )

        total = (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / gib
        )

        utilization = None

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )

            utilization = float(
                result.stdout
                .strip()
                .splitlines()[0]
            )

        except Exception:
            utilization = None

        return {
            "available":
                True,

            "device_name":
                torch.cuda.get_device_name(
                    0
                ),

            "memory_allocated_gib":
                allocated,

            "memory_reserved_gib":
                reserved,

            "memory_total_gib":
                total,

            "memory_usage_percent":
                (
                    allocated
                    / total
                    * 100.0
                ),

            "utilization_percent":
                utilization,
        }

    def metrics_snapshot(self):
        return self.metrics.snapshot(
            self.gpu_metrics()
        )

    def health(self):
        cuda = (
            torch.cuda.is_available()
        )

        all_ready = (
            self.loaded
            and self.base_model
            is not None
            and self.adapted_model
            is not None
            and self.embedding_model
            is not None
            and self.nli_model
            is not None
        )

        return {
            "status":
                (
                    "healthy"
                    if all_ready
                    else "degraded"
                ),

            "cuda_available":
                cuda,

            "device":
                str(self.device),

            "model_loaded":
                (
                    self.base_model
                    is not None
                    and self.adapted_model
                    is not None
                ),

            "retriever_loaded":
                (
                    self.dense_embeddings
                    is not None
                ),

            "nli_loaded":
                self.nli_model
                is not None,

            "uptime_seconds":
                (
                    utc_now()
                    - self.metrics.started_at
                ).total_seconds(),
        }


service = DeploymentService()