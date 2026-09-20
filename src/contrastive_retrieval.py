"""
Contrastive Retrieval in ICA Space.

Avoids the reconstruction step entirely. Instead of:
  s → flip axis k → reconstruct → cosine NN in original space

We search directly in ICA space for words that are:
  - Similar to the query on all OTHER axes (same semantic neighborhood)
  - Opposite to the query on the TARGET axis (opposite pole)

score(c) = cos(s_w[other], s_c[other]) - λ · s_w[k] · s_c[k] / std[k]²
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ica_transformer import ICASpace


@dataclass
class ContrastiveResult:
    word: str
    score: float
    cos_other: float   # similarity on non-target axes
    axis_align: float  # alignment on target axis (negative = good)
    axis_idx: int
    label: str = ""


class ContrastiveRetriever:
    """
    Direct search in ICA space: similar everywhere except on one axis.

    Related to SemAxis (An et al. 2018) but fully data-driven via ICA.
    """

    def __init__(self, space: ICASpace, lam: float = 1.0):
        self.space = space
        self.lam = lam
        self._axis_std = np.std(space.S, axis=0)
        self._axis_std = np.maximum(self._axis_std, 1e-8)

        # Precompute row norms of S for fast cosine similarity
        self._S = space.S  # (n_words, n_components)
        norms = np.linalg.norm(self._S, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._S_normed = self._S / norms  # normalized rows

    def search(
        self,
        word: str,
        axis_idx: int,
        top_n: int = 10,
        exclude: set[str] | None = None,
        lam: float | None = None,
    ) -> list[ContrastiveResult]:
        """
        Find words that are semantically close on all axes except axis_idx,
        where they should be on the opposite pole.

        Args:
            word: Query word.
            axis_idx: The semantic axis to invert.
            top_n: Number of results to return.
            exclude: Words to exclude from results.
            lam: Override instance lambda. Higher = stronger opposition penalty.

        Returns:
            List of ContrastiveResult sorted by score descending.
        """
        lam = lam if lam is not None else self.lam
        idx = self.space.word_to_idx.get(word)
        if idx is None:
            return []

        s_w = self._S[idx]  # (n_components,)
        excl = {word} | (exclude or set())

        # Axes other than target
        other_axes = [i for i in range(self.space.n_components) if i != axis_idx]
        other_axes = np.array(other_axes)

        # Cosine similarity on other axes
        s_w_other = s_w[other_axes]
        norm_w_other = np.linalg.norm(s_w_other)
        norm_w_other = max(norm_w_other, 1e-10)

        S_other = self._S[:, other_axes]  # (n_words, n_components-1)
        norms_other = np.linalg.norm(S_other, axis=1)
        norms_other = np.maximum(norms_other, 1e-10)
        cos_other = (S_other @ s_w_other) / (norm_w_other * norms_other)  # (n_words,)

        # Alignment on target axis: positive means same direction (bad), negative = opposite (good)
        # Normalize by axis variance to make λ scale-invariant
        axis_align = (self._S[:, axis_idx] * s_w[axis_idx]) / (self._axis_std[axis_idx] ** 2)

        # Combined score: similar elsewhere, opposite on target axis
        scores = cos_other - lam * axis_align

        # Sort and collect results
        ranked = np.argsort(scores)[::-1]
        results = []
        for i in ranked:
            if len(results) >= top_n:
                break
            w = self.space.words[i]
            if w in excl:
                continue
            results.append(ContrastiveResult(
                word=w,
                score=float(scores[i]),
                cos_other=float(cos_other[i]),
                axis_align=float(axis_align[i]),
                axis_idx=axis_idx,
            ))
        return results

    def search_label_group(
        self,
        word: str,
        axes: list[int],
        top_n: int = 10,
        exclude: set[str] | None = None,
        lam: float | None = None,
    ) -> list[ContrastiveResult]:
        """
        Contrastive search inverting a GROUP of axes simultaneously.

        Score: similar on all non-group axes, opposite on all group axes.
        """
        lam = lam if lam is not None else self.lam
        idx = self.space.word_to_idx.get(word)
        if idx is None:
            return []

        s_w = self._S[idx]
        excl = {word} | (exclude or set())
        axes_set = set(axes)
        other_axes = np.array([i for i in range(self.space.n_components) if i not in axes_set])

        # Cosine on other axes
        s_w_other = s_w[other_axes]
        norm_w_other = max(np.linalg.norm(s_w_other), 1e-10)
        S_other = self._S[:, other_axes]
        norms_other = np.maximum(np.linalg.norm(S_other, axis=1), 1e-10)
        cos_other = (S_other @ s_w_other) / (norm_w_other * norms_other)

        # Sum of alignment penalties across all group axes
        group_align = np.zeros(len(self.space.words))
        for k in axes:
            group_align += (self._S[:, k] * s_w[k]) / (self._axis_std[k] ** 2)

        scores = cos_other - lam * group_align
        ranked = np.argsort(scores)[::-1]

        results = []
        for i in ranked:
            if len(results) >= top_n:
                break
            w = self.space.words[i]
            if w in excl:
                continue
            results.append(ContrastiveResult(
                word=w,
                score=float(scores[i]),
                cos_other=float(cos_other[i]),
                axis_align=float(group_align[i]),
                axis_idx=-1,
            ))
        return results

    def tune_lambda(
        self,
        pairs: list[tuple[str, str]],
        axis_selector,
        lam_values: list[float] | None = None,
        top_n: int = 10,
    ) -> float:
        """
        Find the best λ on a set of antonym pairs using Hits@10.

        Args:
            pairs: (word1, word2) antonym pairs for tuning.
            axis_selector: callable(w1, w2) -> axis_idx
            lam_values: λ values to try.

        Returns:
            Best λ value.
        """
        if lam_values is None:
            lam_values = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]

        best_lam = 1.0
        best_hits = 0

        for lam in lam_values:
            hits = 0
            total = 0
            for w1, w2 in pairs:
                axis = axis_selector(w1, w2)
                if axis is None:
                    continue
                total += 1
                for src, tgt in [(w1, w2), (w2, w1)]:
                    results = self.search(src, axis, top_n=top_n, lam=lam)
                    if any(r.word == tgt for r in results):
                        hits += 1
                        break
            if total > 0 and hits > best_hits:
                best_hits = hits
                best_lam = lam

        return best_lam
