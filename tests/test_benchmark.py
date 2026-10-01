import json

import backend.benchmark as benchmark


def test_load_benchmark(tmp_path, monkeypatch):
    rrf_path = tmp_path / "rrf.json"
    qa_path = tmp_path / "qa.json"
    grounding_path = tmp_path / "grounding.json"

    rrf_path.write_text(
        json.dumps(
            {
                "summary": {
                    "overall": {
                        "recall_at_1": 0.72,
                        "recall_at_3": 0.86,
                        "recall_at_5": 0.90,
                        "recall_at_10": 0.94,
                        "mrr_at_10": 0.80,
                    }
                }
            }
        )
    )

    qa_model = {
        "summary": {
            "overall": {
                "task_exact_match": 0.01,
                "task_f1": 0.10,
                "answerable_exact_match": 0.02,
                "answerable_f1": 0.20,
                "unanswerable_abstention_accuracy": 0.0,
                "unanswerable_false_answer_rate": 1.0,
            }
        }
    }

    qa_path.write_text(
        json.dumps(
            {
                "base": qa_model,
                "self_supervised": qa_model,
            }
        )
    )

    grounding_model = {
        "summary": {
            "overall": {
                "overall": {
                    "nli_entailment_rate": 0.60,
                    "nli_neutral_rate": 0.20,
                    "nli_contradiction_rate": 0.20,
                    "nli_hallucination_proxy_rate": 0.40,
                    "supported_but_exact_match_wrong_rate": 0.59,
                    "supported_with_positive_f1_rate": 0.26,
                }
            }
        }
    }

    grounding_path.write_text(
        json.dumps(
            {
                "base": grounding_model,
                "self_supervised": grounding_model,
            }
        )
    )

    monkeypatch.setattr(benchmark, "RRF_PATH", rrf_path)
    monkeypatch.setattr(benchmark, "QA_PATH", qa_path)
    monkeypatch.setattr(
        benchmark,
        "GROUNDING_PATH",
        grounding_path,
    )

    result = benchmark.load_benchmark()

    assert result["test_examples"] == 5940
    assert result["retrieval"]["recall_at_1"] == 0.72
    assert result["retrieval"]["mrr_at_10"] == 0.80
    assert result["base"]["qa"]["task_f1"] == 0.10
    assert (
        result["self_supervised"]["grounding"]
        ["nli_entailment_rate"]
        == 0.60
    )


def test_benchmark_notes_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "load_benchmark",
        lambda: {
            "notes": [
                "NLI non-entailment is an automatic hallucination proxy, "
                "not factual ground truth."
            ]
        },
    )

    result = benchmark.load_benchmark()

    assert "automatic hallucination proxy" in result["notes"][0]
    assert "not factual ground truth" in result["notes"][0]