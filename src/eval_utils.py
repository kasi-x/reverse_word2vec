"""
Shared helpers for evaluation scripts.

Extracted from scripts/eval_*.py, which previously each carried their own
copy of these functions.
"""

from __future__ import annotations

import numpy as np


def make_folds(n: int, n_folds: int = 5, seed: int = 42) -> list[np.ndarray]:
    """Shuffled index folds shared by every CV loop (seed 42)."""
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    return list(np.array_split(indices, n_folds))


def train_test_pairs(pairs: list, fold_idx: int, folds: list[np.ndarray]) -> tuple[list, list]:
    """Split `pairs` into (train, test) for one fold of `make_folds` output."""
    test_idx = set(folds[fold_idx].tolist())
    train = [pairs[i] for i in range(len(pairs)) if i not in test_idx]
    test = [pairs[i] for i in folds[fold_idx]]
    return train, test


def unit_rows(M: np.ndarray) -> np.ndarray:
    """Return M with each row scaled to unit L2 norm."""
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-10)


def topn_excluding(scores: np.ndarray, words: list[str], query: str, top_n: int) -> list[str]:
    """Indices of the top-n scores, skipping the query word."""
    n = scores.size
    k = min(top_n + 1, n)  # +1 covers the skipped query word
    pool = np.argpartition(scores, n - k)[n - k :]
    pool = pool[np.argsort(scores[pool])[::-1]]
    return [words[i] for i in pool if words[i] != query][:top_n]


def compute_hits(
    pairs: list[tuple[str, str]],
    retrieve_fn,
    top_n: int = 10,
) -> tuple[dict[int, float], int]:
    """
    Hits@{1,5,10} for a retrieval function over antonym pairs.

    Standard protocol: for each pair, the target must appear in the top-n
    list for at least one direction (w1->w2 or w2->w1); the best rank over
    both directions counts.
    """
    hits = {1: 0, 5: 0, 10: 0}
    total = 0
    for w1, w2 in pairs:
        total += 1
        best_rank = None
        for src, tgt in [(w1, w2), (w2, w1)]:
            results = retrieve_fn(src)
            for j, word in enumerate(results[:top_n]):
                if word == tgt:
                    rank = j + 1
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                    break
        if best_rank is not None:
            for k in [1, 5, 10]:
                if best_rank <= k:
                    hits[k] += 1
    return {k: hits[k] / total for k in [1, 5, 10]}, total
