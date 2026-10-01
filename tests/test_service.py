from types import SimpleNamespace

import pytest
import torch

import backend.service as service_module
from backend.service import (
    DeploymentService,
    MetricsStore,
    average_pool,
    lexical_support,
    normalize_answer,
    tokenize_bm25,
)


def test_tokenize_bm25():
    result = tokenize_bm25(
        "Albert Einstein, Relativity!"
    )

    assert result == [
        "albert",
        "einstein",
        "relativity",
    ]


def test_normalize_answer():
    assert (
        normalize_answer(
            "The Albert Einstein."
        )
        == "albert einstein"
    )


def test_lexical_support_true():
    assert lexical_support(
        "Albert Einstein",
        "The physicist Albert Einstein proposed relativity.",
    ) is True


def test_lexical_support_false_for_unknown():
    assert lexical_support(
        "I don't know.",
        "Some context.",
    ) is False


def test_lexical_support_false_for_empty():
    assert lexical_support(
        "",
        "Some context.",
    ) is False


def test_average_pool():
    hidden = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [100.0, 100.0],
            ]
        ]
    )

    mask = torch.tensor(
        [[1, 1, 0]]
    )

    result = average_pool(
        hidden,
        mask,
    )

    expected = torch.tensor(
        [[2.0, 3.0]]
    )

    assert torch.allclose(
        result,
        expected,
    )


def test_percentile_empty():
    assert (
        MetricsStore.percentile(
            [],
            50,
        )
        is None
    )


def test_percentile_interpolation():
    result = MetricsStore.percentile(
        [10.0, 20.0, 30.0, 40.0],
        50,
    )

    assert result == 25.0


def test_latency_summary_empty():
    result = MetricsStore.latency_summary(
        []
    )

    assert result == {
        "last_ms": None,
        "average_ms": None,
        "p50_ms": None,
        "p95_ms": None,
    }


def test_latency_summary_values():
    result = MetricsStore.latency_summary(
        [10.0, 20.0, 30.0]
    )

    assert result["last_ms"] == 30.0
    assert result["average_ms"] == 20.0
    assert result["p50_ms"] == 20.0
    assert result["p95_ms"] == pytest.approx(
        29.0
    )


def test_metrics_store_records_requests(
    monkeypatch,
):
    store = MetricsStore(
        rolling_window_size=10
    )

    monkeypatch.setattr(
        service_module.time,
        "time",
        lambda: 1000.0,
    )

    store.record_retrieval(
        25.0,
        external_request=True,
    )

    store.record_retrieval(
        15.0,
        external_request=False,
    )

    store.record_answer(
        generation_ms=100.0,
        grounding_ms=20.0,
        total_ms=150.0,
    )

    store.record_error()

    assert store.requests_total == 2
    assert store.retrieval_requests == 1
    assert store.answer_requests == 1
    assert store.errors_total == 1

    assert list(
        store.retrieval_latencies
    ) == [25.0, 15.0]


def test_requests_per_minute(
    monkeypatch,
):
    store = MetricsStore()

    store.request_timestamps.extend(
        [
            940.0,
            950.0,
            999.0,
        ]
    )

    monkeypatch.setattr(
        service_module.time,
        "time",
        lambda: 1000.0,
    )

    assert (
        store.requests_per_minute()
        == 3.0
    )


def test_metrics_snapshot():
    store = MetricsStore(
        rolling_window_size=10
    )

    store.record_retrieval(
        30.0
    )

    store.record_answer(
        generation_ms=100.0,
        grounding_ms=20.0,
        total_ms=160.0,
    )

    gpu = {
        "available": True,
        "memory_usage_percent": 20.0,
        "utilization_percent": 10.0,
    }

    result = store.snapshot(
        gpu
    )

    assert result["requests_total"] == 2
    assert result["gpu"] == gpu

    assert (
        result[
            "retrieval_latency"
        ]["last_ms"]
        == 30.0
    )

    assert (
        result[
            "end_to_end_latency"
        ]["last_ms"]
        == 160.0
    )

    assert len(
        result["history"]
    ) == 1


def test_build_bm25_index():
    service = DeploymentService()

    service.document_texts = [
        "Albert Einstein relativity",
        "Isaac Newton gravity",
    ]

    service._build_bm25_index()

    assert (
        "einstein"
        in service.bm25_postings
    )

    assert (
        service.bm25_doc_lengths
        == [3, 3]
    )

    assert (
        service.bm25_avg_doc_length
        == 3.0
    )


def test_bm25_search():
    service = DeploymentService()

    service.documents = [
        {},
        {},
    ]

    service.document_ids = [
        "doc-a",
        "doc-b",
    ]

    service.document_texts = [
        "Albert Einstein relativity",
        "Isaac Newton gravity",
    ]

    service._build_bm25_index()

    result = service._bm25_search(
        "Einstein",
        top_k=1,
    )

    assert len(result) == 1

    assert (
        result[0]["document_id"]
        == "doc-a"
    )

    assert result[0]["rank"] == 1

    assert result[0]["score"] > 0


def test_resolve_nli_labels():
    service = DeploymentService()

    service.nli_model = (
        SimpleNamespace(
            config=SimpleNamespace(
                id2label={
                    0: "CONTRADICTION",
                    1: "NEUTRAL",
                    2: "ENTAILMENT",
                }
            )
        )
    )

    result = (
        service._resolve_nli_labels()
    )

    assert result == {
        "contradiction": 0,
        "neutral": 1,
        "entailment": 2,
    }


def test_resolve_nli_labels_failure():
    service = DeploymentService()

    service.nli_model = (
        SimpleNamespace(
            config=SimpleNamespace(
                id2label={
                    0: "LABEL_0",
                    1: "LABEL_1",
                }
            )
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        service._resolve_nli_labels()


def test_retrieve_rrf_fusion(
    monkeypatch,
):
    service = DeploymentService()

    service.loaded = True

    service.document_texts = [
        "Document A",
        "Document B",
    ]

    service._bm25_search = (
        lambda query: [
            {
                "index": 0,
                "document_id": "a",
                "score": 10.0,
                "rank": 1,
            },
            {
                "index": 1,
                "document_id": "b",
                "score": 8.0,
                "rank": 2,
            },
        ]
    )

    service._dense_search = (
        lambda query: [
            {
                "index": 1,
                "document_id": "b",
                "score": 0.9,
                "rank": 1,
            },
            {
                "index": 0,
                "document_id": "a",
                "score": 0.8,
                "rank": 2,
            },
        ]
    )

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda: None,
    )

    result = service.retrieve(
        "test query",
        top_k=2,
    )

    assert result["method"] == (
        "BM25 + E5 + RRF"
    )

    assert result["top_k"] == 2
    assert len(
        result["documents"]
    ) == 2

    assert {
        row["document_id"]
        for row
        in result["documents"]
    } == {"a", "b"}


def test_retrieve_requires_loaded():
    service = DeploymentService()

    service.loaded = False

    with pytest.raises(
        RuntimeError,
        match="Service is not loaded",
    ):
        service.retrieve(
            "test"
        )


def test_health_degraded():
    service = DeploymentService()

    service.loaded = False
    service.base_model = None
    service.adapted_model = None
    service.dense_embeddings = None
    service.nli_model = None

    result = service.health()

    assert (
        result["status"]
        == "degraded"
    )

    assert (
        result["model_loaded"]
        is False
    )

    assert (
        result["retriever_loaded"]
        is False
    )


def test_health_healthy():
    service = DeploymentService()

    service.loaded = True
    service.base_model = object()
    service.adapted_model = object()
    service.embedding_model = object()
    service.nli_model = object()
    service.dense_embeddings = object()

    result = service.health()

    assert (
        result["status"]
        == "healthy"
    )

    assert (
        result["model_loaded"]
        is True
    )

    assert (
        result["retriever_loaded"]
        is True
    )

    assert (
        result["nli_loaded"]
        is True
    )


def test_gpu_metrics_without_cuda(
    monkeypatch,
):
    service = DeploymentService()

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    result = (
        service.gpu_metrics()
    )

    assert result == {
        "available": False,
        "device_name": None,
        "memory_allocated_gib": None,
        "memory_reserved_gib": None,
        "memory_total_gib": None,
        "memory_usage_percent": None,
        "utilization_percent": None,
    }