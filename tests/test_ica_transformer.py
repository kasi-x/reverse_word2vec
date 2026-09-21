"""Tests for ICA transform / reconstruct / save-load."""

import numpy as np

from src.ica_transformer import (
    ICASpace,
    ICATransformer,
    is_valid_english_word,
    load_ica_space,
    save_ica_space,
)


def test_reconstruction_algebra_with_exact_inverse():
    """With an orthogonal unmixing W and mixing = W^-1, roundtrip is exact."""
    rng = np.random.RandomState(0)
    d = 6
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))  # orthogonal
    x = rng.normal(size=(10, d))
    mean = x.mean(axis=0)
    s = (x - mean) @ q  # S = Xc @ W.T with unmixing W = q.T
    space = ICASpace(
        words=[f"w{i}" for i in range(10)],
        word_to_idx={f"w{i}": i for i in range(10)},
        S=s,
        mixing_matrix=q,  # A = inv(W) = q (q orthogonal: q^-1 = q.T, W = q.T)
        unmixing_matrix=q.T,
        mean_vector=mean,
        n_components=d,
    )
    transformer = ICATransformer()
    reconstructed = transformer.reconstruct(s, space)
    assert np.allclose(reconstructed, x, atol=1e-10)


def test_score_unknown_word_returns_none(toy_space):
    assert toy_space.score("not_a_word_zzz") is None
    assert toy_space.score("king") is not None


def test_top_words_returns_words_on_both_poles(toy_space):
    words = toy_space.top_words(0, n=5)
    assert len(words) == 5
    neg = toy_space.top_words(0, n=5, positive=False)
    assert set(words).isdisjoint(neg)


def test_save_load_roundtrip(toy_space, tmp_path):
    path = str(tmp_path / "space")
    save_ica_space(toy_space, path, meta={"model": "toy", "counter_fitted": False})
    loaded = load_ica_space(path)

    assert loaded.words == toy_space.words
    assert loaded.n_components == toy_space.n_components
    assert np.allclose(loaded.S, toy_space.S)
    assert np.allclose(loaded.mixing_matrix, toy_space.mixing_matrix)
    assert loaded.meta["model"] == "toy"
    assert loaded.meta["counter_fitted"] is False
    assert "created_at" in loaded.meta


def test_is_valid_english_word():
    assert is_valid_english_word("hello")
    assert is_valid_english_word("don't")
    assert is_valid_english_word("mother-in-law")
    assert not is_valid_english_word("hello123")
    assert not is_valid_english_word("")
    assert not is_valid_english_word("--x")
    assert not is_valid_english_word("Won't")  # uppercase


def test_fit_respects_vocab_limit(toy_model):
    space = ICATransformer().fit(toy_model, n_components=4, vocab_limit=10)
    assert space.n_components == 4
    assert len(space.words) <= 10
