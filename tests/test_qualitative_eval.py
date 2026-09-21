"""Tests for the blind/oracle qualitative evaluator (structure + blindness)."""

from src.axis_labeler import AxisLabeler
from src.qualitative_eval import CANONICAL_TEST_CASES, QualitativeEvaluator
from src.semantic_operations import SemanticOperator


def _make_evaluator(toy_model, toy_space):
    op = SemanticOperator(toy_model, toy_space)
    profiles = AxisLabeler(toy_space).profile_all_axes()
    # Give every axis an empty label; the blind label variant must then
    # report no_labeled_axis instead of consulting the target word.
    evaluator = QualitativeEvaluator(op, toy_space, profiles)
    return evaluator


def test_run_blind_covers_all_cases(toy_model, toy_space):
    evaluator = _make_evaluator(toy_model, toy_space)
    results = evaluator.run_blind()

    for variant in ("label", "auto"):
        block = results[variant]
        assert block["total_cases"] == len(CANONICAL_TEST_CASES)
        assert len(block["cases"]) == len(CANONICAL_TEST_CASES)
        for _k, v in block["metrics"].items():
            assert 0.0 <= v <= 1.0


def test_blind_label_mode_reports_missing_axis_status(toy_model, toy_space):
    evaluator = _make_evaluator(toy_model, toy_space)
    results = evaluator.run_blind()
    statuses = [c.get("status") for c in results["label"]["cases"]]
    # No axis carries a label in this fixture, so every in-vocab case must be
    # flagged no_labeled_axis (never silently evaluated with oracle info).
    assert all(s == "no_labeled_axis" for s in statuses)


def test_blind_label_finds_opposite_on_labeled_axes(toy_model, toy_space):
    """With gender pairs labeled, the blind label protocol should reach the
    expected partner in the top-10 for at least some toy gender queries."""
    op = SemanticOperator(toy_model, toy_space)
    labeler = AxisLabeler(toy_space)
    profiles = labeler.profile_all_axes()
    pairs = [
        ("king", "queen"),
        ("boy", "girl"),
        ("father", "mother"),
        ("husband", "wife"),
        ("man", "woman"),
    ]
    profiles = labeler.auto_label(pairs, profiles)
    assert any(p.label == "gender" for p in profiles), "gender axis not labeled"

    evaluator = QualitativeEvaluator(op, toy_space, profiles)
    metrics = evaluator.run_blind()["label"]["metrics"]
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_run_all_returns_both_protocols(toy_model, toy_space):
    evaluator = _make_evaluator(toy_model, toy_space)
    results = evaluator.run_all()
    assert set(results) == {"blind", "oracle"}
    assert set(results["blind"]) == {"label", "auto"}


def test_failure_analysis_flags_not_found_words(toy_model, toy_space):
    evaluator = _make_evaluator(toy_model, toy_space)
    oracle = evaluator.run_oracle()
    failures = evaluator.failure_analysis(oracle)
    assert failures["total_failures"] >= 0
    assert isinstance(failures["failures"], list)
