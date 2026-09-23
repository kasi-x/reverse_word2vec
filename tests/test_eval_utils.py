"""Tests for shared eval helpers."""

import numpy as np

from src.eval_utils import compute_hits, make_folds, topn_excluding, train_test_pairs, unit_rows


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


def test_make_folds_deterministic_and_covering():
    f1 = make_folds(11, 5, seed=42)
    f2 = make_folds(11, 5, seed=42)
    assert [list(f) for f in f1] == [list(f) for f in f2]
    assert sorted(int(i) for f in f1 for i in f) == list(range(11))


def test_train_test_pairs_split_without_overlap():
    pairs = [(f"a{i}", f"b{i}") for i in range(10)]
    folds = make_folds(len(pairs), 5, seed=42)
    train, test = train_test_pairs(pairs, 0, folds)
    assert len(train) + len(test) == len(pairs)
    assert not set(map(str, train)) & set(map(str, test))
