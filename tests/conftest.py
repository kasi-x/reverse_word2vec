"""Shared fixtures: a tiny synthetic embedding space (no model downloads)."""

import numpy as np
import pytest
from gensim.models import KeyedVectors

from src.ica_transformer import ICASpace, ICATransformer

DIMS = 8

# Words used by the canonical qualitative test cases must be in the toy vocab
CANONICAL_WORDS = [
    "king",
    "queen",
    "boy",
    "girl",
    "father",
    "mother",
    "husband",
    "wife",
    "happy",
    "sad",
    "good",
    "bad",
    "love",
    "hate",
    "hot",
    "cold",
    "warm",
    "cool",
    "big",
    "small",
    "huge",
    "tiny",
    "up",
    "down",
    "high",
    "low",
    "alive",
    "dead",
    "begin",
    "end",
]

FILLER_WORDS = [f"w{i:03d}" for i in range(120)]


def build_toy_model() -> KeyedVectors:
    """Synthetic embeddings with a strong 'gender' dimension (dim 0) and
    a strong 'sentiment' dimension (dim 1), plus random filler words."""
    rng = np.random.RandomState(42)

    keys: list[str] = []
    vecs: list[np.ndarray] = []

    def add(word: str, vec: np.ndarray) -> None:
        keys.append(word)
        vecs.append(vec)

    # Gender pairs live on dim 0, sentiment pairs on dim 1
    gender_pairs = [
        ("king", "queen"),
        ("boy", "girl"),
        ("father", "mother"),
        ("husband", "wife"),
        ("man", "woman"),
    ]
    for m, f in gender_pairs:
        add(m, np.r_[2.0, 0.0, rng.normal(0, 0.1, DIMS - 2)])
        add(f, np.r_[-2.0, 0.0, rng.normal(0, 0.1, DIMS - 2)])

    sentiment_pairs = [("happy", "sad"), ("good", "bad"), ("love", "hate")]
    for pos, neg in sentiment_pairs:
        add(pos, np.r_[0.0, 2.0, rng.normal(0, 0.1, DIMS - 2)])
        add(neg, np.r_[0.0, -2.0, rng.normal(0, 0.1, DIMS - 2)])

    # Temperature pairs share a direction on dim 2
    for pos, neg in [("hot", "cold"), ("warm", "cool")]:
        add(pos, np.r_[0.0, 0.0, 1.5, rng.normal(0, 0.1, DIMS - 3)])
        add(neg, np.r_[0.0, 0.0, -1.5, rng.normal(0, 0.1, DIMS - 3)])

    for w in CANONICAL_WORDS:
        if w not in keys:
            add(w, rng.normal(0, 0.5, DIMS))

    for w in FILLER_WORDS:
        add(w, rng.normal(0, 0.5, DIMS))

    kv = KeyedVectors(vector_size=DIMS)
    kv.add_vectors(keys, np.array(vecs, dtype=np.float32))
    return kv


@pytest.fixture(scope="session")
def toy_model() -> KeyedVectors:
    return build_toy_model()


@pytest.fixture(scope="session")
def toy_space(toy_model) -> ICASpace:
    return ICATransformer().fit(toy_model, n_components=DIMS, vocab_limit=1000)
