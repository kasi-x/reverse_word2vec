"""
Semantic operations on ICA-decomposed embedding spaces.

Provides axis-wise inversion, sliding, and nearest-neighbor search
for discovering antonyms and performing semantic manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gensim.models import KeyedVectors

from src.axis_labeler import AxisProfile
from src.ica_transformer import ICASpace, ICATransformer


@dataclass
class NeighborResult:
    """A nearest-neighbor search result."""

    word: str
    similarity: float


@dataclass
class InversionResult:
    """Result of a semantic inversion with metadata."""

    neighbors: list[NeighborResult]
    axes_flipped: list[int]
    strategy: str
    label: str = ""


class SemanticOperator:
    """Operations on ICA-decomposed word vectors."""

    def __init__(
        self,
        model: KeyedVectors,
        space: ICASpace,
        profiles: list[AxisProfile] | None = None,
    ):
        self.model = model
        self.space = space
        self.transformer = ICATransformer()
        self.profiles = profiles or []
        # Label -> list of axis indices
        self._label_to_axes: dict[str, list[int]] = {}
        for p in self.profiles:
            if p.label:
                self._label_to_axes.setdefault(p.label, []).append(p.axis_idx)
        # Precompute axis std for z-score calculations
        self._axis_std = np.std(space.S, axis=0)
        self._axis_std = np.maximum(self._axis_std, 1e-8)
        # Build search index restricted to ICA vocabulary (valid English words only)
        self._search_words = space.words
        self._search_indices = [model.key_to_index[w] for w in space.words]
        search_vecs = model.vectors[self._search_indices].astype(np.float32)
        norms = np.linalg.norm(search_vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._normalized = search_vecs / norms

    def axis_invert(
        self,
        word: str,
        axis_idx: int,
        top_n: int = 10,
        exclude: set[str] | None = None,
    ) -> list[NeighborResult]:
        """
        Invert a word's score on a single axis and find nearest neighbors.

        Args:
            word: Input word.
            axis_idx: ICA axis to invert.
            top_n: Number of neighbors to return.
            exclude: Words to exclude from results.

        Returns:
            List of NeighborResult ordered by similarity.
        """
        scores = self.transformer.transform_word(word, self.model, self.space)
        if scores is None:
            return []
        modified = scores.copy()
        modified[axis_idx] = -modified[axis_idx]
        vec = self.transformer.reconstruct(modified, self.space)
        excl = {word} | (exclude or set())
        return self.find_nearest(vec, top_n, excl)

    def multi_axis_invert(
        self,
        word: str,
        axis_indices: list[int],
        top_n: int = 10,
        exclude: set[str] | None = None,
    ) -> list[NeighborResult]:
        """Invert a word's score on multiple axes simultaneously."""
        scores = self.transformer.transform_word(word, self.model, self.space)
        if scores is None:
            return []
        modified = scores.copy()
        for idx in axis_indices:
            modified[idx] = -modified[idx]
        vec = self.transformer.reconstruct(modified, self.space)
        excl = {word} | (exclude or set())
        return self.find_nearest(vec, top_n, excl)

    # ── Practical inversion strategies (no oracle) ──────────────

    def invert_by_source_topk(
        self,
        word: str,
        k: int = 5,
        top_n: int = 10,
        exclude: set[str] | None = None,
    ) -> InversionResult:
        """
        Strategy 1: Flip the k axes where the SOURCE word has the highest |z-score|.

        No knowledge of target word needed. Based on the idea that a word's
        most extreme axes are its defining semantic dimensions.
        """
        scores = self.transformer.transform_word(word, self.model, self.space)
        if scores is None:
            return InversionResult([], [], "source_topk")
        z = np.abs(scores) / self._axis_std
        top_axes = list(np.argsort(z)[-k:][::-1])
        neighbors = self.multi_axis_invert(word, top_axes, top_n, exclude)
        return InversionResult(neighbors, top_axes, "source_topk")

    def invert_by_label(
        self,
        word: str,
        label: str,
        top_n: int = 10,
        exclude: set[str] | None = None,
    ) -> InversionResult:
        """
        Strategy 2: Flip ALL axes that share a semantic label.

        User specifies the dimension (e.g., "gender", "temperature").
        Multiple axes may carry the same semantic concept; flipping all
        of them together should give a stronger inversion than a single axis.
        """
        axes = self._label_to_axes.get(label, [])
        if not axes:
            return InversionResult([], [], "label_group", label)
        neighbors = self.multi_axis_invert(word, axes, top_n, exclude)
        return InversionResult(neighbors, axes, "label_group", label)

    def invert_by_scan(
        self,
        word: str,
        top_n: int = 10,
        exclude: set[str] | None = None,
        candidate_axes: list[int] | None = None,
    ) -> list[InversionResult]:
        """
        Strategy 3: Try each axis (or label group) individually, return
        all results ranked by how different the top neighbor is from the source.

        Returns multiple InversionResults so the caller can see which
        semantic dimension produces the most meaningful inversion.
        """
        scores = self.transformer.transform_word(word, self.model, self.space)
        if scores is None:
            return []
        original_vec = self.model[word].astype(np.float32)
        original_norm = original_vec / max(np.linalg.norm(original_vec), 1e-10)

        results = []

        # Try each label group
        if self._label_to_axes:
            for label, axes in self._label_to_axes.items():
                neighbors = self.multi_axis_invert(word, axes, top_n, exclude)
                if not neighbors:
                    continue
                # Score by dissimilarity from source (1 - cos_sim)
                top_vec = self.model[neighbors[0].word].astype(np.float32)
                top_norm = top_vec / max(np.linalg.norm(top_vec), 1e-10)
                dissim = 1.0 - float(original_norm @ top_norm)
                results.append((dissim, InversionResult(neighbors, axes, "scan_label", label)))

        # Also try individual high-kurtosis axes without labels
        if candidate_axes is None:
            candidate_axes = list(range(self.space.n_components))
        for k in candidate_axes:
            if any(k in axes for axes in self._label_to_axes.values()):
                continue  # already covered by label groups
            neighbors = self.axis_invert(word, k, top_n, exclude)
            if not neighbors:
                continue
            top_vec = self.model[neighbors[0].word].astype(np.float32)
            top_norm = top_vec / max(np.linalg.norm(top_vec), 1e-10)
            dissim = 1.0 - float(original_norm @ top_norm)
            label = self._label_to_axes_reverse(k)
            results.append((dissim, InversionResult(neighbors, [k], "scan_single", label)))

        results.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in results]

    def _label_to_axes_reverse(self, axis_idx: int) -> str:
        """Get label for an axis index."""
        for label, axes in self._label_to_axes.items():
            if axis_idx in axes:
                return label
        return ""

    def invert_auto(
        self,
        word: str,
        top_n: int = 10,
        max_results: int = 5,
        exclude: set[str] | None = None,
    ) -> list[InversionResult]:
        """
        Automatic inversion: try all label groups, return the top results
        ranked by semantic distance from the source word.

        This is the "give me all possible opposites" mode.
        """
        scan = self.invert_by_scan(word, top_n, exclude)
        return scan[:max_results]

    def invert_by_attributes(
        self,
        word: str,
        z_threshold: float = 2.0,
        top_n: int = 10,
        exclude: set[str] | None = None,
    ) -> list[InversionResult]:
        """
        Attribute-based multi-sense inversion.

        Identifies the axes where the source word has strong engagement
        (|z| >= threshold), then inverts on each axis INDIVIDUALLY.
        Returns one InversionResult per strong axis, sorted by z-score.

        This surfaces "many kinds of opposite" for a word:
          light → {weight: heavy, brightness: dark, mood: serious}

        Args:
            word: Query word.
            z_threshold: Minimum |z-score| to consider an axis an "attribute".
            top_n: Neighbors per axis.
            exclude: Words to exclude.

        Returns:
            List of InversionResult, one per strong axis, z-score descending.
        """
        scores = self.transformer.transform_word(word, self.model, self.space)
        if scores is None:
            return []

        z = np.abs(scores) / self._axis_std
        strong_axes = np.where(z >= z_threshold)[0]
        strong_axes = strong_axes[np.argsort(z[strong_axes])[::-1]]  # sort by z desc

        results = []
        for k in strong_axes:
            neighbors = self.axis_invert(word, int(k), top_n, exclude)
            if neighbors:
                label = self._label_to_axes_reverse(int(k))
                results.append(
                    InversionResult(
                        neighbors=neighbors,
                        axes_flipped=[int(k)],
                        strategy="attribute",
                        label=label,
                    )
                )
        return results

    def axis_slide(
        self,
        word: str,
        axis_idx: int,
        target_score: float,
        top_n: int = 10,
        exclude: set[str] | None = None,
    ) -> list[NeighborResult]:
        """
        Set a word's score on an axis to a specific value.

        Args:
            word: Input word.
            axis_idx: ICA axis to modify.
            target_score: Desired score on that axis.
            top_n: Number of neighbors to return.
            exclude: Words to exclude.

        Returns:
            List of NeighborResult.
        """
        scores = self.transformer.transform_word(word, self.model, self.space)
        if scores is None:
            return []
        modified = scores.copy()
        modified[axis_idx] = target_score
        vec = self.transformer.reconstruct(modified, self.space)
        excl = {word} | (exclude or set())
        return self.find_nearest(vec, top_n, excl)

    def find_nearest(
        self,
        vector: np.ndarray,
        top_n: int = 10,
        exclude: set[str] | None = None,
    ) -> list[NeighborResult]:
        """Find nearest words to a vector by cosine similarity (ICA vocab only)."""
        vec = vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return []
        vec_norm = vec / norm

        sims = self._normalized @ vec_norm
        exclude = exclude or set()

        # Only need the top (top_n + |exclude|) candidates; the pool always
        # contains at least top_n non-excluded words when k <= vocab.
        n = sims.size
        k = min(top_n + len(exclude), n)
        pool = np.argpartition(sims, n - k)[n - k :]
        pool = pool[np.argsort(sims[pool])[::-1]]

        results = []
        for idx in pool:
            if len(results) >= top_n:
                break
            w = self._search_words[idx]
            if w in exclude:
                continue
            results.append(NeighborResult(word=w, similarity=float(sims[idx])))
        return results

    def compare_words(self, word1: str, word2: str) -> dict:
        """
        Compare two words axis-by-axis, showing the largest score differences.

        Returns:
            Dict with 'diffs' (sorted axis differences) and 'cosine_sim'.
        """
        s1 = self.transformer.transform_word(word1, self.model, self.space)
        s2 = self.transformer.transform_word(word2, self.model, self.space)
        if s1 is None or s2 is None:
            return {"error": f"Word not found: {word1 if s1 is None else word2}"}

        diff = s1 - s2
        abs_diff = np.abs(diff)
        ranked = np.argsort(abs_diff)[::-1]

        diffs = []
        for axis in ranked[:10]:
            diffs.append(
                {
                    "axis": int(axis),
                    "score_w1": float(s1[axis]),
                    "score_w2": float(s2[axis]),
                    "diff": float(diff[axis]),
                }
            )

        v1 = self.model[word1]
        v2 = self.model[word2]
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        return {"word1": word1, "word2": word2, "cosine_sim": cos, "top_diffs": diffs}

    def analogy_traditional(
        self,
        a: str,
        b: str,
        c: str,
        top_n: int = 10,
    ) -> list[NeighborResult]:
        """
        Traditional vector analogy: a is to b as c is to ?
        Computes b - a + c and finds nearest neighbors.
        """
        for w in [a, b, c]:
            if w not in self.model:
                return []
        vec = (
            self.model[b].astype(np.float64)
            - self.model[a].astype(np.float64)
            + self.model[c].astype(np.float64)
        )
        return self.find_nearest(vec.astype(np.float32), top_n, exclude={a, b, c})
