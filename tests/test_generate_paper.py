"""Tests for data-driven paper generation (no invented numbers)."""

import json

from src.generate_paper import generate_paper


def test_missing_results_produce_na_and_no_crash(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    out = tmp_path / "paper" / "paper.md"

    generate_paper(results_dir=str(results_dir), output_path=str(out))

    text = out.read_text(encoding="utf-8")
    assert "n/a" in text  # missing data is visible, not silently filled


def test_numbers_are_taken_from_result_files(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    (results_dir / "reranker_eval.json").write_text(
        json.dumps(
            {
                "cv": {
                    "n_folds": 5,
                    "total_pairs": 100,
                    "metrics": {
                        "mlp": {"hits_at_1": {"mean": 0.1, "std": 0.01}},
                        "reranker": {
                            "hits_at_1": {"mean": 0.42, "std": 0.02},
                            "hits_at_5": {"mean": 0.5, "std": 0.02},
                            "hits_at_10": {"mean": 0.55, "std": 0.02},
                        },
                    },
                },
                "holdout": {"mlp": {"1": 0.2}, "reranker": {"1": 0.33}},
                "feature_weights": {"freq_ratio": -1.5, "ica_cosine": -1.0},
            }
        )
    )
    (results_dir / "antonym_retrieval.json").write_text(
        json.dumps(
            {
                "n_folds": 5,
                "total_valid_pairs": 100,
                "metrics": {"hits_at_1": {"mean": 0.25, "std": 0.01}},
            }
        )
    )

    out = tmp_path / "paper" / "paper.md"
    generate_paper(results_dir=str(results_dir), output_path=str(out))
    text = out.read_text(encoding="utf-8")

    assert "0.420±0.020" in text  # reranker CV row
    assert "0.250±0.010" in text  # oracle row
    assert "42.0%" in text  # abstract hits@1
    assert "n/a" not in text.split("## Method")[0].split("Abstract")[1]


def test_blind_qualitative_table_rendered(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "qualitative_eval.json").write_text(
        json.dumps(
            {
                "blind": {
                    "label": {
                        "total_cases": 2,
                        "evaluated": 2,
                        "metrics": {"hits_at_1": 0.5, "hits_at_5": 1.0, "hits_at_10": 1.0},
                        "cases": [
                            {
                                "word": "king",
                                "expected": "queen",
                                "axis_label": "gender",
                                "axes": [0],
                                "rank": 1,
                                "neighbors": [{"word": "queen", "sim": 0.9}],
                            }
                        ],
                    },
                    "auto": {
                        "total_cases": 2,
                        "evaluated": 2,
                        "metrics": {"hits_at_1": 0.0, "hits_at_5": 0.0, "hits_at_10": 0.5},
                        "cases": [],
                    },
                },
                "oracle": {
                    "total_cases": 2,
                    "evaluated": 2,
                    "metrics": {"hits_at_1": 1.0, "hits_at_5": 1.0, "hits_at_10": 1.0},
                    "cases": [],
                },
            }
        )
    )

    out = tmp_path / "paper" / "paper.md"
    generate_paper(results_dir=str(results_dir), output_path=str(out))
    text = out.read_text(encoding="utf-8")

    assert "Blind (declared label group)" in text
    assert "Blind (auto, source top-1 axis)" in text
    assert "Oracle axis selection" in text
    assert "| king | gender | queen | queen | 1 |" in text
