"""Tests for counter-fitting (antonym repel + space preservation)."""

import numpy as np

from src.counter_fitting import CounterFitConfig, CounterFitter


def _cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_antonym_cosine_decreases(toy_model):
    pairs = [("king", "queen"), ("man", "woman"), ("up", "down")]
    before = np.mean([_cosine(toy_model[a], toy_model[b]) for a, b in pairs])

    fitter = CounterFitter(CounterFitConfig(n_iter=50, target_sim=-0.3, verbose=False))
    fitted = fitter.fit(toy_model, pairs)

    after = np.mean([_cosine(fitted[a], fitted[b]) for a, b in pairs])
    assert after < before


def test_unconstrained_words_are_unchanged(toy_model):
    pairs = [("king", "queen"), ("man", "woman")]
    fitter = CounterFitter(CounterFitConfig(n_iter=30, verbose=False))
    fitted = fitter.fit(toy_model, pairs)

    for w in ["hot", "cold", "w000", "w050"]:  # not involved in any constraint
        assert np.allclose(fitted[w], toy_model[w], atol=1e-5)


def test_evaluate_antonym_separation_stats(toy_model):
    fitter = CounterFitter(CounterFitConfig(verbose=False))
    stats = fitter.evaluate_antonym_separation(toy_model, [("king", "queen"), ("man", "woman")])
    assert stats["n_pairs"] == 2
    assert -1.0 <= stats["mean"] <= 1.0
    assert 0.0 <= stats["pct_positive"] <= 1.0


def test_fit_is_deterministic(toy_model):
    cfg = CounterFitConfig(n_iter=10, verbose=False)
    pairs = [("king", "queen"), ("up", "down")]
    m1 = CounterFitter(cfg).fit(toy_model, pairs)
    m2 = CounterFitter(cfg).fit(toy_model, pairs)
    assert np.allclose(m1.vectors, m2.vectors)
