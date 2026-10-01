# LLM Self-Supervised Fine-Tuning - Qwen2.5 


This repository is a hands-on portfolio project designed to demonstrate an inspectable, production-oriented ML/LLM engineering workflow. It is not presented as long-term commercial production experience.
There is a technical report as a follow-up with images, in this repository.

## 1. Technical scope

The implemented system includes:

- SQuAD v2 reused as a self-supervised text corpus: training sees context text only, not QA labels
- strict train/validation/test context separation and frozen downstream QA evaluation
- deterministic corpus construction and 512-token packed causal-LM blocks
- full-parameter fine-tuning of `Qwen2.5-0.5B`; no LoRA/QLoRA adapters and no Hugging Face `Trainer`
- explicit PyTorch/Hugging Face training loop
- Optuna hyperparameter search plus independent confirmation of the selected candidate
- epoch-level validation-loss checkpoint selection
- base-versus-adapted causal-LM evaluation
- downstream SQuAD v2 QA evaluation to test objective/task alignment
- BM25, dense E5, and hybrid Reciprocal Rank Fusion retrieval benchmarks
- retrieval-conditioned QA evaluation
- lexical-support and NLI grounding analysis with an explicit automatic hallucination proxy
- FastAPI backend for health/readiness, retrieval, answering, model metadata, live metrics, and validated benchmark results
- static browser monitoring dashboard with live latency/GPU telemetry and offline benchmark separation
- CUDA inference on an NVIDIA A30 24 GB GPU under SLURM
- 38 automated tests with 78% backend statement coverage

## 2. System flow

```text
Official SQuAD v2
      |
      +------------------------------+
      |                              |
      v                              v
Self-supervised corpus          Frozen QA test set
(context only; no QA labels)         |
      |                              |
      v                              |
Deduplicate / split audit            |
      |                              |
      v                              |
512-token packed blocks              |
      |                              |
      v                              |
Full-parameter continued             |
pretraining of Qwen2.5-0.5B          |
      |                              |
      v                              |
Validation-loss model selection      |
      |                              |
      +-------------+----------------+
                    |
          +---------+---------+
          |                   |
          v                   v
   LM evaluation       Downstream QA evaluation
          |                   |
          |             Retrieval benchmark
          |          BM25 / E5 / Hybrid RRF
          |                   |
          |             Retrieval-conditioned QA
          |                   |
          +---------+---------+
                    |
                    v
          Grounding / NLI analysis
                    |
                    v
         FastAPI + monitoring dashboard
```

The retrieval layer is used as a controlled evaluation and deployment-grounding layer around the self-supervised adaptation experiment; this repository is not intended to replace a dedicated RAG project.

## 3. Dataset construction

Training uses only SQuAD context text. QA labels are not exposed during self-supervised training.

| Item | Value |
|---|---:|
| Unique training contexts | 14,639 |
| Unique validation contexts | 603 |
| Frozen downstream test examples | 5,940 |
| Unique frozen test contexts | 601 |
| Packed block size | 512 tokens |
| Packed training blocks | 4,655 |
| Packed validation blocks | 201 |

The corpus audit verifies that train, validation, and frozen test contexts do not overlap.

## 4. Repository structure

```text
llm-self-supervised-fine-tuning/
├── backend/
│   ├── benchmark.py
│   ├── main.py
│   ├── schemas.py
│   └── service.py
├── configs/
│   └── training.yaml
├── data/
│   ├── manifests/
│   ├── processed/
│   └── raw/
├── docs/
│   └── screenshots/
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── mlruns/
├── results/
│   ├── baseline/
│   ├── checkpoints/
│   ├── deployment/
│   ├── evaluation/
│   └── final_model/
├── scripts/
├── slurm/
├── src/
│   └── llm_continued_pretraining/
├── tests/
│   ├── test_api.py
│   ├── test_benchmark.py
│   ├── test_schemas.py
│   └── test_service.py
├── pyproject.toml
├── pytest.ini
├── requirements-lock.txt
├── README.md
└── technical_report.pdf
```

Large datasets, model weights, dense embedding tensors, experiment databases, and local MLflow state should not be committed to normal Git history. They can be regenerated from the tracked scripts/configuration or restored separately.

## 5. Validated runtime

Canonical validated environment:

- Python `3.11.11`
- NVIDIA A30 24 GB GPU
- SLURM-managed GPU execution
- PyTorch + Hugging Face Transformers
- `bfloat16` model precision for training/evaluation/serving
- FastAPI + Uvicorn backend
- static HTML/CSS/JavaScript frontend
- pytest + pytest-cov

The canonical deployment path requires CUDA because the service loads the base model, adapted model, dense retriever, and NLI classifier together.

## 6. Installation

### 6.1 Create the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 6.2 Install a CUDA-compatible PyTorch build

Install the PyTorch build appropriate to the target GPU driver/CUDA environment.

### 6.3 Install the project

```bash
python -m pip install -e .
python -m pip check
```

For the exact validated environment snapshot, see:

```text
requirements-lock.txt
```

## 7. Required artifacts

The selected adapted checkpoint is expected at:

```text
results/checkpoints/self_supervised_v3_refined/best_model/
```

The deployment retriever expects:

```text
results/deployment/retrieval/documents.json
results/deployment/retrieval/dense_embeddings.pt
```

The `/benchmark` endpoint reads the validated evaluation artifacts from `results/evaluation/`.

These large/generated artifacts are intentionally separate from ordinary Git history. A fresh clone should rebuild or restore them before GPU serving.

## 8. Self-supervised preprocessing

The experiment is deliberately label-free during adaptation:

```text
SQuAD training contexts
        |
        v
context deduplication
        |
        v
tokenization
        |
        v
concatenate token stream
        |
        v
pack into fixed 512-token blocks
        |
        v
next-token causal-LM objective
```

The downstream QA test set is frozen and excluded from the self-supervised corpus. This prevents downstream evaluation labels from leaking into training.

## 9. Full-parameter training configuration

Final selected configuration:

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` |
| Epochs | 3 |
| Per-device batch size | 8 |
| Gradient accumulation | 2 |
| Effective batch size | 16 |
| Precision | `bf16` |
| Learning rate | `1.5207292892805172e-05` |
| Weight decay | `0.0407392908424372` |
| Warmup ratio | `0.031301953382248254` |
| Scheduler | cosine |
| Max grad norm | `1.9869442371619883` |
| Seed | 42 |

This is full-parameter training. No PEFT adapter is used.

## 10. Hyperparameter selection

The project uses a two-stage search process:

1. broad Optuna search over learning rate, weight decay, warmup ratio, and scheduler;
2. refined Optuna search with a narrower region plus max-gradient-norm search, followed by an independent confirmation run.

The final configuration comes from refined Optuna trial `19` and was confirmed independently before full training.

## 11. Training and causal-LM validation

Epoch-level validation:

| Epoch | Validation loss |
|---:|---:|
| 1 | **2.672082** |
| 2 | 2.675994 |
| 3 | 2.677708 |

Epoch 1 is selected as the best checkpoint; later epochs show mild overfitting.

Definitive held-out LM comparison:

| Metric | Base | Self-supervised FT | Relative change |
|---|---:|---:|---:|
| Validation loss | 2.749122 | **2.672082** | **-2.80%** |
| Perplexity | 15.6289 | **14.4701** | **-7.41%** |

The continued-pretraining objective therefore improves the held-out causal-language-model objective.

## 12. Direct downstream QA evaluation

The frozen 5,940-example SQuAD v2 test set is also used to test whether the LM improvement transfers to context-grounded QA without supervised QA adaptation.

| Metric | Base | Self-supervised FT |
|---|---:|---:|
| Task EM | 0.0042 | 0.0019 |
| Task F1 | **0.1070** | 0.0855 |
| Answerable F1 | **0.2136** | 0.1707 |
| Unanswerable abstention accuracy | 0.0000 | 0.0000 |

The self-supervised model improves LM loss but regresses on QA. Both models fail to learn abstention from the self-supervised objective. This is the central objective/task-alignment result of the project.

## 13. Retrieval benchmark

The deployment/evaluation retrieval corpus contains 20,233 unique official SQuAD train+validation contexts. The frozen 5,940 QA examples are queries, with their associated SQuAD context used as the retrieval target.

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR / MRR@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.6869 | 0.8121 | 0.8481 | 0.8886 | 0.7604 |
| E5 small-v2 | 0.6470 | 0.8024 | 0.8458 | 0.8904 | 0.7370 |
| Hybrid RRF | **0.7210** | **0.8620** | **0.9029** | **0.9357** | **0.7985** |

Hybrid RRF combines BM25 and dense E5 rankings with `k = 60` and gives the strongest retrieval performance in this controlled benchmark.

## 14. Retrieval-conditioned QA

Using the hybrid-RRF top-1 retrieved context:

| Metric | Base + RRF | Self-supervised + RRF |
|---|---:|---:|
| Overall task F1 | **0.0894** | 0.0727 |
| Answerable F1 | **0.1784** | 0.1451 |
| F1 when gold context is rank 1 | **0.1133** | 0.0924 |
| F1 when gold context is not rank 1 | **0.0275** | 0.0216 |
| Unanswerable abstention accuracy | 0.0000 | 0.0000 |

Retrieval success strongly affects QA quality, but the adapted model remains below the base model even when the gold context is retrieved at rank 1. Retrieval and model/task alignment are therefore separate bottlenecks.

## 15. Grounding and factuality analysis

Grounding uses:

- lexical support against the supplied context;
- `FacebookAI/roberta-large-mnli` for entailment/neutral/contradiction probabilities;
- `neutral + contradiction` as an **automatic hallucination proxy**, not as factual ground truth.

With hybrid-RRF top-1 retrieved evidence:

| Metric | Base + RRF | Self-supervised + RRF |
|---|---:|---:|
| NLI entailment | 0.5976 | **0.6729** |
| NLI contradiction | 0.1875 | **0.1258** |
| Hallucination proxy | 0.4024 | **0.3271** |
| Supported but exact-match wrong | 0.5948 | 0.6717 |
| Supported with positive F1 | **0.2614** | 0.2577 |

The self-supervised model becomes more consistent with retrieved evidence while becoming less correct on the downstream QA task. The project therefore separates **grounding/support** from **task correctness** rather than treating them as interchangeable.

## 16. Serving application

The FastAPI backend loads the deployment stack during application lifespan startup.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service/component health |
| `GET /ready` | Model/retriever/NLI readiness |
| `GET /model-info` | Runtime model and retrieval metadata |
| `POST /retrieve` | BM25 + E5 + RRF retrieval |
| `POST /answer` | retrieval -> generation -> grounding |
| `GET /metrics` | live inference/GPU observability |
| `GET /benchmark` | frozen validated offline benchmark |

The generation system instruction is:

```text
Answer the question using only the provided context.
Return only the answer supported by the context.
If the context does not contain the answer, respond exactly: I don't know.
```

Serving uses deterministic generation with `do_sample = false`, `repetition_penalty = 1.0`, and `max_new_tokens = 64`.

## 17. Run the backend

On a CUDA GPU node with the environment active:

```bash
PYTHONPATH=. uvicorn backend.main:app --host 127.0.0.1 --port 8010
```

Swagger/OpenAPI:

```text
http://127.0.0.1:8010/docs
```

## 18. Run the frontend

The monitoring dashboard is a static HTML/CSS/JavaScript application:

```bash
cd frontend
python -m http.server 8050 --bind 127.0.0.1
```

After tunneling/port forwarding:

```text
http://127.0.0.1:8050
```

The UI separates two categories deliberately:

- **live production-style telemetry**: service status, inference request count, errors, requests/minute, component latency, end-to-end latency, GPU memory/utilization, history;
- **validated offline benchmark**: retrieval recall/MRR, QA metrics, NLI entailment, contradiction, hallucination proxy.

Per-request inference also exposes generated output, retrieved evidence, grounding label/probabilities, hallucination-proxy flag, and component trace latency.

## 19. Remote GPU workflow

The validated setup uses a local browser, a remote login node, and a SLURM GPU node. Hostnames and account-specific paths are intentionally omitted from the public repository.

Conceptually:

```text
local browser
    |
VS Code port forwarding
    |
remote login node
    |
SSH local tunnel
    |
SLURM GPU node
    +-- FastAPI :8010
    +-- frontend :8050
```

The exact cluster allocation command is site-specific and should be adapted to the target SLURM environment.

## 20. Tests and coverage

Run:

```bash
PYTHONPATH=. pytest -q
```

Validated result:

```text
38 passed
```

Coverage command:

```bash
PYTHONPATH=. pytest -q --cov=backend --cov-report=term-missing
```

Validated backend statement coverage:

| Module | Coverage |
|---|---:|
| `backend/benchmark.py` | 100% |
| `backend/main.py` | 92% |
| `backend/schemas.py` | 100% |
| `backend/service.py` | 66% |
| **Total backend** | **78%** |

The untested portion of `service.py` is concentrated primarily in real model loading, dense-model execution, generation, NLI inference, and CUDA-specific branches. These paths were also exercised through live GPU API/dashboard smoke tests rather than being inflated with extensive mocks solely to maximize the coverage percentage.

## 21. Live observability

The backend maintains a rolling inference window and exposes:

- inference request counts;
- retrieval and answer request counts;
- error count/rate;
- requests per minute;
- last/average/P50/P95 retrieval latency;
- generation latency;
- grounding latency;
- end-to-end latency;
- GPU device, allocated/reserved/total memory, memory ratio, and utilization;
- recent history for dashboard plotting.

One validated live snapshot recorded:

```text
requests_total        3
errors_total          0
error_rate            0.0
retrieval avg         102.10 ms
retrieval P95         217.76 ms
generation            648.82 ms
grounding               45.85 ms
end-to-end             740.50 ms
GPU                    NVIDIA A30
allocated VRAM           3.34 GiB
total VRAM              23.60 GiB
```

These are runtime observations from a small live smoke session, not throughput/load-test claims.

## 22. Reproducibility

Reproducibility is supported by:

- tracked `configs/training.yaml`;
- deterministic corpus/split scripts and fixed seeds;
- explicit training loop and SLURM wrappers;
- Optuna study configuration and exported trial/confirmation artifacts;
- frozen evaluation scripts and result JSONs;
- deployment-index build script;
- `requirements-lock.txt`;
- pytest suite and coverage measurement;
- separation of live telemetry from offline benchmark metrics.

Fixed seeds and configuration do not imply bitwise-identical GPU results across hardware/software environments.

## 23. Known limitations

- Self-supervised continued pretraining improves causal-LM validation loss/perplexity but does not improve downstream SQuAD QA accuracy in this experiment.
- Neither base nor adapted model learns reliable unanswerable-question abstention from the self-supervised objective.
- NLI non-entailment is only an automatic hallucination proxy; it is not ground-truth factuality verification.
- A response can be well supported by retrieved context and still be irrelevant or incorrect for the question.
- Retrieval target correctness is based on the associated SQuAD context, which is a controlled benchmark definition rather than a universal relevance judgment.
- Dense retrieval uses a compact E5 model rather than an exhaustive retrieval-model study.
- The browser service is a portfolio deployment demonstration, not a 24/7 commercial service.
- No concurrency/stress benchmark, autoscaling layer, authentication system, or CI/CD production pipeline is claimed.
- Large checkpoints, embeddings, datasets, local MLflow stores, and Optuna databases should remain outside ordinary Git history.

## 24. Technical report

A separate illustrated report documents the experimental design, training evidence, LM results, retrieval benchmark, downstream QA, grounding analysis, deployment architecture, live observability, tests, limitations, and conclusions:

```text
technical_report.pdf
```

The README intentionally contains no screenshots. Visual runtime evidence and narrative analysis are kept in the technical report.
