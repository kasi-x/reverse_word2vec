"""Tests for shared eval helpers."""

import numpy as np

from src.eval_utils import compute_hits, topn_excluding, unit_rows


def test_unit_rows_normalises():
    m = np.array([[3.0, 4.0], [0.0, 0.0]])
    u = unit_rows(m)
    assert np.isclose(np.linalg.norm(u[0]), 1.0)
    assert np.allclose(u[1], 0.0)  # zero row does not blow up


def test_topn_excluding_skips_query():
    words = ["a", "b", "c", "d"]
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    assert topn_excluding(scores, words, query="a", top_n=3) == ["b", "c", "d"]
    assert topn_excluding(scores, words, query="b", top_n=2) == ["c", "d"]


def test_compute_hits_best_rank_over_directions():
    def retrieve_fn(src):
        # foo->bar finds the target at rank 2; the reverse misses it entirely
        return {"foo": ["x", "bar"], "bar": ["x", "y", "z", "baz"]}[src]

    hits, total = compute_hits([("foo", "bar")], retrieve_fn, top_n=3)
    assert total == 1
    assert hits[1] == 0.0
    assert hits[5] == 1.0  # best rank 2 counts at @5


def test_compute_counts_misses():
    def retrieve_fn(_src):
        return ["x", "y", "z"]  # target never present

    hits, total = compute_hits([("foo", "bar")], retrieve_fn, top_n=10)
    assert total == 1
    assert hits[10] == 0.0


def test_compute_hits_respects_top_n():
    def retrieve_fn(_src):
        return ["x", "y", "z", "bar"]  # target at rank 4

    hits, _ = compute_hits([("foo", "bar")], retrieve_fn, top_n=3)
    assert hits[5] == 0.0
    hits, _ = compute_hits([("foo", "bar")], retrieve_fn, top_n=10)
    assert hits[10] == 1.0
