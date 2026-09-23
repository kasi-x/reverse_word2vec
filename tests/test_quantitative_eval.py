"""Tests for oracle antonym-retrieval evaluation."""

import numpy as np

from src.quantitative_eval import AntonymRetrievalEval
from src.semantic_operations import SemanticOperator


def _eval_for(toy_space, toy_model) -> AntonymRetrievalEval:
    return AntonymRetrievalEval(SemanticOperator(toy_model, toy_space), toy_space)


def test_select_best_axis_uses_std_normalised_diff(toy_space, toy_model):
    ev = _eval_for(toy_space, toy_model)
    s1, s2 = toy_space.score("king"), toy_space.score("queen")
    expected = int(np.argmax(np.abs(s1 - s2) / ev._axis_std))
    assert ev.select_best_axis("king", "queen") == expected


def test_select_best_axis_unknown_word_returns_none(toy_space, toy_model):
    ev = _eval_for(toy_space, toy_model)
    assert ev.select_best_axis("king", "not_a_word_zzz") is None


def test_evaluate_pairs_runs_on_toy_vocab(toy_space, toy_model):
    ev = _eval_for(toy_space, toy_model)
    out = ev.evaluate_pairs([("king", "queen"), ("up", "down")], top_n=5)
    assert out["evaluated"] == 2
    assert set(out["metrics"]) == {"hits_at_1", "hits_at_5", "hits_at_10"}
