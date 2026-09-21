"""Tests for the ICA-feature antonym classifier."""

import numpy as np
import pytest

from src.antonym_classifier import AntonymClassifier


@pytest.fixture(scope="module")
def fitted_logistic(toy_space):
    pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("father", "mother"),
        ("husband", "wife"),
    ]
    clf = AntonymClassifier(toy_space, "logistic")
    clf.fit(pairs, neg_ratio=3.0, rng=np.random.RandomState(42))
    return clf


def test_features_have_expected_shape(toy_space):
    clf = AntonymClassifier(toy_space, "logistic")
    s1, s2 = toy_space.score("king"), toy_space.score("queen")
    feat = clf._features(s1, s2)
    assert feat.shape == (2 * toy_space.n_components,)


def test_features_are_symmetric_in_product_absolute_diff(toy_space):
    clf = AntonymClassifier(toy_space, "logistic")
    s1, s2 = toy_space.score("king"), toy_space.score("queen")
    assert np.allclose(clf._features(s1, s2), clf._features(s2, s1))


def test_dataset_build_is_deterministic(toy_space):
    pairs = [("king", "queen"), ("man", "woman"), ("up", "down")]
    x1, y1 = AntonymClassifier(toy_space)._build_dataset(
        pairs, neg_ratio=2.0, rng=np.random.RandomState(42)
    )
    x2, y2 = AntonymClassifier(toy_space)._build_dataset(
        pairs, neg_ratio=2.0, rng=np.random.RandomState(42)
    )
    assert x1.shape == x2.shape
    assert np.allclose(x1, x2)
    assert set(np.unique(y1)) == {0, 1}


def test_fit_and_pair_score(fitted_logistic):
    # Gender pairs sit far apart on axis 0 of the toy space → high P(antonym)
    gender_score = fitted_logistic.pair_score("king", "queen")
    filler_score = fitted_logistic.pair_score("w000", "w001")
    assert gender_score > filler_score


def test_retrieve_excludes_query(fitted_logistic):
    results = fitted_logistic.retrieve("king", top_n=10)
    assert 0 < len(results) <= 10
    assert all(w != "king" for w, _ in results)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_unknown_model_type_raises(toy_space):
    with pytest.raises(ValueError):
        AntonymClassifier(toy_space, "svm")


def test_unfitted_raises(toy_space):
    clf = AntonymClassifier(toy_space, "logistic")
    with pytest.raises(RuntimeError):
        clf.pair_score("king", "queen")


def test_top_antonym_axes_logistic(fitted_logistic):
    axes = fitted_logistic.top_antonym_axes(n=3)
    assert len(axes) == 3
    assert all(isinstance(k, int) for k, _ in axes)
