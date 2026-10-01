from backend.schemas import (
    AnswerRequest,
    BenchmarkResponse,
    GroundingResult,
    HealthResponse,
    MetricsResponse,
    RetrieveRequest,
)


def test_schema_imports():
    assert HealthResponse is not None
    assert RetrieveRequest is not None
    assert AnswerRequest is not None
    assert GroundingResult is not None
    assert MetricsResponse is not None
    assert BenchmarkResponse is not None


def test_retrieve_request_defaults():
    request = RetrieveRequest(
        query="What is machine learning?"
    )

    assert request.query == (
        "What is machine learning?"
    )
    assert request.top_k == 5


def test_answer_request_default_model():
    request = AnswerRequest(
        question="What is optimization?"
    )

    assert request.model == (
        "self_supervised"
    )


def test_retrieve_request_top_k_validation():
    try:
        RetrieveRequest(
            query="test",
            top_k=11,
        )
    except Exception:
        assert True
    else:
        assert False