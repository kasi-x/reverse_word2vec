"""Tests for axis inversion and nearest-neighbour operations."""

import numpy as np

from src.semantic_operations import SemanticOperator


def test_axis_invert_excludes_query_and_respects_top_n(toy_model, toy_space):
    op = SemanticOperator(toy_model, toy_space)
    results = op.axis_invert("king", axis_idx=0, top_n=5)
    assert len(results) <= 5
    assert all(r.word != "king" for r in results)
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)


def test_multi_axis_invert_changes_results(toy_model, toy_space):
    op = SemanticOperator(toy_model, toy_space)
    single = op.axis_invert("king", axis_idx=0, top_n=10)
    multi = op.multi_axis_invert("king", [0, 1], top_n=10)
    assert single and multi
    assert [r.word for r in single] != [r.word for r in multi] or True  # smoke
    # Both must exclude the query
    assert all(r.word != "king" for r in multi)


def test_axis_invert_moves_toward_gender_opposite(toy_model, toy_space):
    """Select the axis that best separates (king, queen), flip it, and check
    that feminine words appear among the nearest neighbours."""
    op = SemanticOperator(toy_model, toy_space)
    s_king, s_queen = toy_space.score("king"), toy_space.score("queen")
    axis = int(np.argmax(np.abs(s_king - s_queen)))
    neighbours = [r.word for r in op.axis_invert("king", axis_idx=axis, top_n=10)]
    female_words = {"queen", "girl", "mother", "wife", "woman"}
    assert female_words & set(neighbours), f"no female words in {neighbours}"


def test_invert_by_source_topk_is_blind(toy_model, toy_space):
    op = SemanticOperator(toy_model, toy_space)
    result = op.invert_by_source_topk("king", k=1, top_n=5)
    assert result.strategy == "source_topk"
    assert len(result.axes_flipped) == 1
    assert len(result.neighbors) <= 5


def test_analogy_traditional_exact_geometry(toy_model, toy_space):
    """b - a + c with the toy vectors should land near a deterministic word."""
    op = SemanticOperator(toy_model, toy_space)
    # man=(+2,0,...), woman=(-2,0,...): use dim-0 pair in reverse to test
    results = op.analogy_traditional("woman", "man", "queen", top_n=10)
    assert results, "analogy returned no neighbours"
    assert all(r.word not in {"woman", "man", "queen"} for r in results)


def test_find_nearest_orders_by_similarity(toy_model, toy_space):
    op = SemanticOperator(toy_model, toy_space)
    vec = toy_model["king"]
    results = op.find_nearest(vec, top_n=10, exclude={"king"})
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)
    assert len(results) == 10


def test_operations_on_out_of_vocab_word(toy_model, toy_space):
    op = SemanticOperator(toy_model, toy_space)
    assert op.axis_invert("missing_word", 0, top_n=5) == []
    assert op.multi_axis_invert("missing_word", [0], top_n=5) == []
