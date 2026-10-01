import logging
import time
from contextlib import asynccontextmanager
from backend.benchmark import load_benchmark
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    AnswerRequest,
    AnswerResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    ReadyResponse,
    RetrieveRequest,
    RetrieveResponse,
    BenchmarkResponse,
)
from backend.service import (
    ADAPTED_MODEL_PATH,
    NLI_MODEL_NAME,
    RRF_K,
    service,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(
    "llm-self-supervised-api"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting deployment service"
    )

    service.load()

    logger.info(
        "Deployment service ready"
    )

    yield

    logger.info(
        "Stopping deployment service"
    )


app = FastAPI(
    title=(
        "Self-Supervised LLM "
        "Deployment API"
    ),
    description=(
        "Production-oriented API for "
        "hybrid retrieval, generation, "
        "grounding and observability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8050",
        "http://localhost:8050",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health():
    return service.health()


@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["System"],
)
def ready():
    health_data = service.health()

    model_ready = (
        health_data["model_loaded"]
    )

    retriever_ready = (
        health_data[
            "retriever_loaded"
        ]
    )

    nli_ready = (
        health_data["nli_loaded"]
    )

    return {
        "ready": (
            model_ready
            and retriever_ready
            and nli_ready
        ),
        "model_ready":
            model_ready,
        "retriever_ready":
            retriever_ready,
        "nli_ready":
            nli_ready,
    }


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["System"],
)
def model_info():
    return {
        "base_model":
            service.base_model_name,

        "adapted_model_path":
            ADAPTED_MODEL_PATH,

        "embedding_model":
            service.embedding_model_name,

        "nli_model":
            NLI_MODEL_NAME,

        "retrieval_method":
            "BM25 + E5 + RRF",

        "rrf_k":
            RRF_K,

        "documents":
            len(service.documents),

        "device":
            str(service.device),

        "precision":
            "bfloat16",
    }


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    tags=["Inference"],
)
def retrieve(
    request: RetrieveRequest,
):
    start = time.perf_counter()

    try:
        response = service.retrieve(
            query=request.query,
            top_k=request.top_k,
            external_request=True,
        )

        logger.info(
            "retrieve request_id=%s "
            "top_k=%s latency_ms=%.2f",
            response["request_id"],
            request.top_k,
            response[
                "retrieval_latency_ms"
            ],
        )

        return response

    except Exception as exc:
        service.metrics.record_error()

        logger.exception(
            "Retrieval request failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Retrieval failed."
            ),
        ) from exc

    finally:
        _ = (
            time.perf_counter()
            - start
        )


@app.post(
    "/answer",
    response_model=AnswerResponse,
    tags=["Inference"],
)
def answer(
    request: AnswerRequest,
):
    try:
        response = service.answer(
            question=request.question,
            model_name=request.model,
        )

        logger.info(
            "answer request_id=%s "
            "model=%s total_ms=%.2f",
            response["request_id"],
            request.model,
            response[
                "trace"
            ][
                "total_latency_ms"
            ],
        )

        return response

    except Exception as exc:
        service.metrics.record_error()

        logger.exception(
            "Answer request failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Answer generation failed."
            ),
        ) from exc


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Observability"],
)
def metrics():
    return (
        service.metrics_snapshot()
    )

@app.get(
    "/benchmark",
    response_model=BenchmarkResponse,
    tags=["Observability"],
)
def benchmark():
    return load_benchmark()