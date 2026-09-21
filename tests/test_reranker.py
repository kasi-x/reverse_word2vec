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


def test_fit_exposes_six_feature_weights(fitted_reranker):
    reranker, _ = fitted_reranker
    weights = reranker.feature_weights()
    assert set(weights) == {
        "mlp_score",
        "ica_cosine",
        "mlp_rank",
        "freq_ratio",
        "interaction",
        "inv_rank",
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
