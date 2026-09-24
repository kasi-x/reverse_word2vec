"""Tests for the logistic reranker over MLP candidates."""

import numpy as np
import pytest

from src.antonym_classifier import AntonymClassifier
from src.reranker import Reranker


@pytest.fixture(scope="module")
def fitted_reranker(toy_space, toy_model):
    pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("father", "mother"),
        ("husband", "wife"),
        ("up", "down"),
    ]
    clf = AntonymClassifier(toy_space, "logistic")
    clf.fit(pairs, neg_ratio=2.0, rng=np.random.RandomState(42))
    reranker = Reranker(toy_space, toy_model)
    reranker.fit(pairs, clf, top_n=10)
    return reranker, clf


def test_fit_exposes_thirteen_feature_weights(fitted_reranker):
    reranker, _ = fitted_reranker
    weights = reranker.feature_weights()
    assert set(weights) == {
        "mlp_score",
        "ica_cosine",
        "mlp_rank",
        "freq_ratio",
        "interaction",
        "inv_rank",
        "glove_cos",
        "morph_sim",
        "max_zdiff",
        "in_mlp",
        "in_axis",
        "in_proc",
        "in_ica_map",
    }


def test_rerank_returns_candidates(fitted_reranker):
    reranker, clf = fitted_reranker
    candidates = clf.retrieve("king", top_n=10)
    reranked = reranker.rerank("king", candidates, top_n=10)
    assert 0 < len(reranked) <= 10
    # Same candidate set, possibly reordered
    assert {w for w, _ in reranked} <= {w for w, _ in candidates}
    scores = [s for _, s in reranked]
    assert scores == sorted(scores, reverse=True)


def test_rerank_empty_candidates(fitted_reranker):
    reranker, _ = fitted_reranker
    assert reranker.rerank("king", [], top_n=10) == []


def test_unfitted_reranker_raises(toy_space, toy_model):
    reranker = Reranker(toy_space, toy_model)
    with pytest.raises(RuntimeError):
        reranker.rerank("king", [("queen", 0.9)], top_n=10)
    assert reranker.feature_weights() == {}


def test_feature_vector_unknown_words_are_zeros(toy_space, toy_model):
    reranker = Reranker(toy_space, toy_model)
    feat = reranker._features("missing_word", "queen", 1, 0.9)
    assert np.allclose(feat, 0.0)
    assert feat.shape == (Reranker.N_FEATURES,)


def test_cross_validate_smoke(toy_space, toy_model):
    pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("father", "mother"),
        ("husband", "wife"),
        ("up", "down"),
        ("happy", "sad"),
        ("good", "bad"),
    ]
    reranker = Reranker(toy_space, toy_model)
    out = reranker.cross_validate(pairs, n_folds=2, top_n=5, mlp_top_n=5, neg_ratio=1.0)
    assert out["n_folds"] == 2
    assert out["total_pairs"] == len(pairs)
    for method in ("mlp", "reranker"):
        for k in ("hits_at_1", "hits_at_5", "hits_at_10"):
            assert 0.0 <= out["metrics"][method][k]["mean"] <= 1.0
