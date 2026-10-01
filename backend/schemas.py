from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# HEALTH / READINESS
# ============================================================


class HealthResponse(BaseModel):
    status: Literal[
        "healthy",
        "degraded",
        "unhealthy",
    ]

    cuda_available: bool
    device: str

    model_loaded: bool
    retriever_loaded: bool
    nli_loaded: bool

    uptime_seconds: float


class ReadyResponse(BaseModel):
    ready: bool

    model_ready: bool
    retriever_ready: bool
    nli_ready: bool


# ============================================================
# MODEL INFORMATION
# ============================================================


class ModelInfoResponse(BaseModel):
    base_model: str
    adapted_model_path: str

    embedding_model: str
    nli_model: str

    retrieval_method: str
    rrf_k: int

    documents: int

    device: str
    precision: str


# ============================================================
# RETRIEVAL
# ============================================================


class RetrieveRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=1000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class RetrievedDocument(BaseModel):
    rank: int

    document_id: str
    text: str

    rrf_score: float

    bm25_rank: int | None = None
    dense_rank: int | None = None

    bm25_score: float | None = None
    dense_score: float | None = None


class RetrieveResponse(BaseModel):
    request_id: str

    query: str

    method: str
    top_k: int

    retrieval_latency_ms: float

    documents: list[RetrievedDocument]


# ============================================================
# ANSWERING / GENERATION
# ============================================================


class AnswerRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
    )

    model: Literal[
        "base",
        "self_supervised",
    ] = "self_supervised"


class GroundingResult(BaseModel):
    lexical_support: bool

    nli_label: Literal[
        "entailment",
        "neutral",
        "contradiction",
    ]

    entailment_probability: float
    neutral_probability: float
    contradiction_probability: float

    hallucination_proxy: bool


class RequestTrace(BaseModel):
    retrieval_latency_ms: float
    generation_latency_ms: float
    grounding_latency_ms: float
    total_latency_ms: float


class AnswerResponse(BaseModel):
    request_id: str
    timestamp: datetime

    question: str

    model: Literal[
        "base",
        "self_supervised",
    ]

    answer: str

    retrieved_document: RetrievedDocument
    grounding: GroundingResult
    trace: RequestTrace


# ============================================================
# LIVE OBSERVABILITY
# ============================================================


class LatencyMetrics(BaseModel):
    last_ms: float | None = None
    average_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None


class GPUMetrics(BaseModel):
    available: bool

    device_name: str | None = None

    memory_allocated_gib: float | None = None
    memory_reserved_gib: float | None = None
    memory_total_gib: float | None = None

    memory_usage_percent: float | None = None
    utilization_percent: float | None = None


class MetricsHistoryPoint(BaseModel):
    timestamp: datetime

    requests_per_minute: float

    end_to_end_latency_ms: float | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    grounding_latency_ms: float | None = None

    gpu_memory_usage_percent: float | None = None
    gpu_utilization_percent: float | None = None


class MetricsResponse(BaseModel):
    server_started_at: datetime
    current_time: datetime

    uptime_seconds: float

    requests_total: int
    retrieval_requests: int
    answer_requests: int

    errors_total: int
    error_rate: float

    requests_per_minute: float

    retrieval_latency: LatencyMetrics
    generation_latency: LatencyMetrics
    grounding_latency: LatencyMetrics
    end_to_end_latency: LatencyMetrics

    gpu: GPUMetrics

    rolling_window_size: int

    history: list[MetricsHistoryPoint]


# ============================================================
# VALIDATED OFFLINE BENCHMARK
# ============================================================


class RetrievalBenchmarkMetrics(BaseModel):
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float

    mrr_at_10: float


class QABenchmarkMetrics(BaseModel):
    task_exact_match: float
    task_f1: float

    answerable_exact_match: float
    answerable_f1: float

    unanswerable_abstention_accuracy: float
    unanswerable_false_answer_rate: float


class GroundingBenchmarkMetrics(BaseModel):
    nli_entailment_rate: float
    nli_neutral_rate: float
    nli_contradiction_rate: float

    nli_hallucination_proxy_rate: float

    supported_but_exact_match_wrong_rate: float
    supported_with_positive_f1_rate: float


class ModelBenchmarkMetrics(BaseModel):
    qa: QABenchmarkMetrics
    grounding: GroundingBenchmarkMetrics


class BenchmarkResponse(BaseModel):
    benchmark_name: str

    test_examples: int

    retrieval: RetrievalBenchmarkMetrics

    base: ModelBenchmarkMetrics
    self_supervised: ModelBenchmarkMetrics

    notes: list[str]