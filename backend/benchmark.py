import json
from pathlib import Path


RRF_PATH = Path(
    "results/evaluation/retrieval_hybrid_rrf.json"
)

QA_PATH = Path(
    "results/evaluation/retrieval_conditioned_qa_refined_full.json"
)

GROUNDING_PATH = Path(
    "results/evaluation/retrieval_conditioned_grounding_full.json"
)


def _load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_benchmark():
    rrf = _load_json(
        RRF_PATH
    )

    qa = _load_json(
        QA_PATH
    )

    grounding = _load_json(
        GROUNDING_PATH
    )

    retrieval = rrf[
        "summary"
    ]["overall"]

    base_qa = qa[
        "base"
    ]["summary"]["overall"]

    ssl_qa = qa[
        "self_supervised"
    ]["summary"]["overall"]

    base_grounding = grounding[
        "base"
    ]["summary"]["overall"]["overall"]

    ssl_grounding = grounding[
        "self_supervised"
    ]["summary"]["overall"]["overall"]

    return {
        "benchmark_name":
            (
                "RRF retrieval-conditioned "
                "QA and grounding benchmark"
            ),

        "test_examples":
            5940,

        "retrieval": {
            "recall_at_1":
                retrieval[
                    "recall_at_1"
                ],

            "recall_at_3":
                retrieval[
                    "recall_at_3"
                ],

            "recall_at_5":
                retrieval[
                    "recall_at_5"
                ],

            "recall_at_10":
                retrieval[
                    "recall_at_10"
                ],

            "mrr_at_10":
                retrieval[
                    "mrr_at_10"
                ],
        },

        "base": {
            "qa": {
                "task_exact_match":
                    base_qa[
                        "task_exact_match"
                    ],

                "task_f1":
                    base_qa[
                        "task_f1"
                    ],

                "answerable_exact_match":
                    base_qa[
                        "answerable_exact_match"
                    ],

                "answerable_f1":
                    base_qa[
                        "answerable_f1"
                    ],

                "unanswerable_abstention_accuracy":
                    base_qa[
                        "unanswerable_abstention_accuracy"
                    ],

                "unanswerable_false_answer_rate":
                    base_qa[
                        "unanswerable_false_answer_rate"
                    ],
            },

            "grounding": {
                "nli_entailment_rate":
                    base_grounding[
                        "nli_entailment_rate"
                    ],

                "nli_neutral_rate":
                    base_grounding[
                        "nli_neutral_rate"
                    ],

                "nli_contradiction_rate":
                    base_grounding[
                        "nli_contradiction_rate"
                    ],

                "nli_hallucination_proxy_rate":
                    base_grounding[
                        "nli_hallucination_proxy_rate"
                    ],

                "supported_but_exact_match_wrong_rate":
                    base_grounding[
                        "supported_but_exact_match_wrong_rate"
                    ],

                "supported_with_positive_f1_rate":
                    base_grounding[
                        "supported_with_positive_f1_rate"
                    ],
            },
        },

        "self_supervised": {
            "qa": {
                "task_exact_match":
                    ssl_qa[
                        "task_exact_match"
                    ],

                "task_f1":
                    ssl_qa[
                        "task_f1"
                    ],

                "answerable_exact_match":
                    ssl_qa[
                        "answerable_exact_match"
                    ],

                "answerable_f1":
                    ssl_qa[
                        "answerable_f1"
                    ],

                "unanswerable_abstention_accuracy":
                    ssl_qa[
                        "unanswerable_abstention_accuracy"
                    ],

                "unanswerable_false_answer_rate":
                    ssl_qa[
                        "unanswerable_false_answer_rate"
                    ],
            },

            "grounding": {
                "nli_entailment_rate":
                    ssl_grounding[
                        "nli_entailment_rate"
                    ],

                "nli_neutral_rate":
                    ssl_grounding[
                        "nli_neutral_rate"
                    ],

                "nli_contradiction_rate":
                    ssl_grounding[
                        "nli_contradiction_rate"
                    ],

                "nli_hallucination_proxy_rate":
                    ssl_grounding[
                        "nli_hallucination_proxy_rate"
                    ],

                "supported_but_exact_match_wrong_rate":
                    ssl_grounding[
                        "supported_but_exact_match_wrong_rate"
                    ],

                "supported_with_positive_f1_rate":
                    ssl_grounding[
                        "supported_with_positive_f1_rate"
                    ],
            },
        },

        "notes": [
            (
                "QA and grounding metrics use "
                "Hybrid RRF top-1 retrieved context."
            ),
            (
                "NLI non-entailment is an automatic "
                "hallucination proxy, not factual ground truth."
            ),
            (
                "Grounded or entailed output does not "
                "necessarily imply task-correct output."
            ),
        ],
    }