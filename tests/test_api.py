from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import backend.main as main


def make_metrics():
    return SimpleNamespace(
        record_error=lambda: None,
    )


def test_health(monkeypatch):
    fake_service = SimpleNamespace(
        health=lambda: {
            "status": "healthy",
            "cuda_available": True,
            "device": "cuda",
            "model_loaded": True,
            "retriever_loaded": True,
            "nli_loaded": True,
            "uptime_seconds": 10.0,
        }
    )

    monkeypatch.setattr(
        main,
        "service",
        fake_service,
    )

    result = main.health()

    assert result["status"] == "healthy"
    assert result["model_loaded"] is True


def test_ready_true(monkeypatch):
    fake_service = SimpleNamespace(
        health=lambda: {
            "model_loaded": True,
            "retriever_loaded": True,
            "nli_loaded": True,
        }
    )

    monkeypatch.setattr(
        main,
        "service",
        fake_service,
    )

    result = main.ready()

    assert result["ready"] is True
    assert result["model_ready"] is True
    assert result["retriever_ready"] is True
    assert result["nli_ready"] is True


def test_ready_false(monkeypatch):
    fake_service = SimpleNamespace(
        health=lambda: {
            "model_loaded": True,
            "retriever_loaded": False,
            "nli_loaded": True,
        }
    )

    monkeypatch.setattr(
        main,
        "service",
        fake_service,
    )

    result = main.ready()

    assert result["ready"] is False


def test_model_info(monkeypatch):
    fake_service = SimpleNamespace(
        base_model_name="Qwen/Qwen2.5-0.5B",
        embedding_model_name="intfloat/e5-small-v2",
        documents=["a", "b", "c"],
        device="cuda",
    )

    monkeypatch.setattr(
        main,
        "service",
        fake_service,
    )

    monkeypatch.setattr(
        main,
        "ADAPTED_MODEL_PATH",
        "results/checkpoints/model",
    )

    monkeypatch.setattr(
        main,
        "NLI_MODEL_NAME",
        "FacebookAI/roberta-large-mnli",
    )

    monkeypatch.setattr(
        main,
        "RRF_K",
        60,
    )

    result = main.model_info()

    assert result["documents"] == 3
    assert result["rrf_k"] == 60
    assert result["device"] == "cuda"


def test_retrieve_success(monkeypatch):
    fake_service = SimpleNamespace(
        metrics=make_metrics(),
        retrieve=lambda **kwargs: {
            "request_id": "req-1",
            "retrieval_latency_ms": 12.5,
            "documents": [],
        },
    )

    monkeypatch.setattr(
        main,
        "service",
        fake_service,
    )

    request = main.RetrieveRequest(
        query="Albert Einstein",
        top_k=3,
    )

    result = main.retrieve(request)

    assert result["request_id"] == "req-1"
    assert result["retrieval_latency_ms"] == 12.5


def test_retrieve_failure(monkeypatch):
    errors = []

    def fail(**kwargs):
        raise RuntimeError("retrieval error")

    fake_service = SimpleNamespace(
        metrics=SimpleNamespace(
            record_error=lambda: errors.append(1),
        ),
        retrieve=fail,
    )

    monkeypatch.setattr(
        main,
        "service",
        fake_service,
    )

    request = main.RetrieveRequest(
        query="test",
        top_k=1,
    )

    with pytest.raises(HTTPException) as exc:
        main.retrieve(request)

    assert exc.value.status_code == 500
    assert len(errors) == 1


def test_answer_success(monkeypatch):
    fake_service = SimpleNamespace(
        metrics=make_metrics(),
        answer=lambda **kwargs: {
            "request_id": "answer-1",
            "trace": {
                "total_latency_ms": 100.0,
            },
        },
    )

    monkeypatch.setattr(
        main,
        "service",
        fake_service,
    )

    request = main.AnswerRequest(
        question="Who proposed relativity?",
        model="self_supervised",
    )

    result = main.answer(request)

    assert result["request_id"] == "answer-1"
    assert result["trace"]["total_latency_ms"] == 100.0


def test_answer_failure(monkeypatch):
    errors = []

    def fail(**kwargs):
        raise RuntimeError("generation error")

    fake_service = SimpleNamespace(
        metrics=SimpleNamespace(
            record_error=lambda: errors.append(1),
        ),
        answer=fail,
    )

    monkeypatch.setattr(
        main,
        "service",
        fake_service,
    )

    request = main.AnswerRequest(
        question="test question",
        model="base",
    )

    with pytest.raises(HTTPException) as exc:
        main.answer(request)

    assert exc.value.status_code == 500
    assert len(errors) == 1


def test_metrics(monkeypatch):
    fake_service = SimpleNamespace(
        metrics_snapshot=lambda: {
            "requests_total": 7,
        }
    )

    monkeypatch.setattr(
        main,
        "service",
        fake_service,
    )

    result = main.metrics()

    assert result["requests_total"] == 7


def test_benchmark(monkeypatch):
    monkeypatch.setattr(
        main,
        "load_benchmark",
        lambda: {
            "benchmark_name": "test benchmark",
            "test_examples": 5940,
        },
    )

    result = main.benchmark()

    assert result["test_examples"] == 5940