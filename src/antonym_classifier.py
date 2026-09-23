"""
ICA Features + Antonym Classifier.

Uses ICA axis scores as interpretable features to train a lightweight
classifier that predicts antonymy between word pairs.

Features for pair (w1, w2):
  - |s1 - s2|        (100d) — axis-wise absolute difference
  - s1 * s2          (100d) — axis-wise product (negative = opposite poles)

Inspired by: Nguyen et al. (2016) "Distinguishing antonyms and synonyms
in a word embedding space" but using ICA features for interpretability.

5-fold CV evaluation included.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.eval_utils import make_folds, train_test_pairs
from src.ica_transformer import ICASpace


class AntonymClassifier:
    """
    Predicts whether two words are antonyms using ICA-based features.

    Can be used in two modes:
      - pair_score(w1, w2): probability that (w1, w2) are antonyms
      - retrieve(word, top_n): rank all vocabulary words as antonym candidates
    """

    # Number of unexcluded top candidates memoized per query word.
    # Callers use top_n <= ~20, so this window is ample.
    _RETRIEVE_CACHE_K = 128

    def __init__(self, space: ICASpace, model_type: str = "logistic"):
        self.space = space
        self.model_type = model_type
        self.scaler = StandardScaler()
        self._axis_std = np.std(space.S, axis=0)
        self._axis_std = np.maximum(self._axis_std, 1e-8)
        # Cached once: retrieve() is called thousands of times during CV.
        self._S32 = space.S.astype(np.float32)
        # word -> top-K (word, prob) ranking; cleared on fit (model changes).
        self._candidate_cache: dict[str, list[tuple[str, float]]] = {}

        if model_type == "logistic":
            self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        elif model_type == "mlp":
            self.clf = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=200,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self._fitted = False

    def _features(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """Compute features for a pair of ICA score vectors."""
        abs_diff = np.abs(s1 - s2)
        product = s1 * s2
        return np.concatenate([abs_diff, product])  # (200,)

    def _build_dataset(
        self,
        antonym_pairs: list[tuple[str, str]],
        neg_ratio: float = 3.0,
        rng: np.random.RandomState | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build a balanced dataset of antonym (pos) and random (neg) pairs.

        Args:
            antonym_pairs: Known antonym pairs.
            neg_ratio: Number of negative samples per positive.
            rng: Random state for reproducibility.

        Returns:
            (X, y) feature matrix and labels.
        """
        if rng is None:
            rng = np.random.RandomState(42)

        X_pos, X_neg = [], []

        # Positive samples: antonym pairs
        valid_pairs = []
        for w1, w2 in antonym_pairs:
            s1 = self.space.score(w1)
            s2 = self.space.score(w2)
            if s1 is not None and s2 is not None:
                # Features (|s1-s2|, s1*s2) are symmetric in the two directions;
                # the pair is appended twice to keep the positive weighting.
                feat = self._features(s1, s2)
                X_pos.append(feat)
                X_pos.append(feat)
                valid_pairs.append((w1, w2))

        # Negative samples: random pairs
        n_neg = int(len(X_pos) * neg_ratio)
        words = self.space.words
        antonym_set = set(antonym_pairs) | {(w2, w1) for w1, w2 in antonym_pairs}

        attempts = 0
        while len(X_neg) < n_neg and attempts < n_neg * 10:
            attempts += 1
            i, j = rng.randint(0, len(words), size=2)
            w1, w2 = words[i], words[j]
            if w1 == w2 or (w1, w2) in antonym_set:
                continue
            s1 = self.space.score(w1)
            s2 = self.space.score(w2)
            if s1 is None or s2 is None:
                continue
            X_neg.append(self._features(s1, s2))

        X = np.array(X_pos + X_neg, dtype=np.float32)
        y = np.array([1] * len(X_pos) + [0] * len(X_neg), dtype=np.int32)
        return X, y

    def fit(
        self,
        antonym_pairs: list[tuple[str, str]],
        neg_ratio: float = 3.0,
        rng: np.random.RandomState | None = None,
    ) -> AntonymClassifier:
        """Train the classifier on antonym pairs."""
        X, y = self._build_dataset(antonym_pairs, neg_ratio, rng)
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self._fitted = True
        self._candidate_cache.clear()
        return self

    def pair_score(self, w1: str, w2: str) -> float:
        """Return P(antonym) for a word pair."""
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        s1 = self.space.score(w1)
        s2 = self.space.score(w2)
        if s1 is None or s2 is None:
            return 0.0
        feat = self._features(s1, s2).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        return float(self.clf.predict_proba(feat_scaled)[0, 1])

    def retrieve(
        self,
        word: str,
        top_n: int = 10,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Rank all vocabulary words as antonym candidates for the query.

        Returns:
            List of (word, antonymy_score) sorted by score descending.
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        s_w = self.space.score(word)
        if s_w is None:
            return []

        excl = {word} | (exclude or set())

        cand = self._candidate_cache.get(word)
        if cand is None:
            cand = self._rank_candidates(s_w, self._RETRIEVE_CACHE_K)
            self._candidate_cache[word] = cand

        results = [(w, p) for w, p in cand if w not in excl][:top_n]
        if len(results) < top_n and len(cand) < len(self.space.words):
            # Exclusions ate into the cached window; rank the full vocabulary.
            cand = self._rank_candidates(s_w, len(self.space.words))
            results = [(w, p) for w, p in cand if w not in excl][:top_n]
        return results

    def _rank_candidates(self, s_w: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Score the whole vocabulary; return the top-k (word, prob) ranking."""
        # Vectorised pair features for the whole vocabulary at once:
        # X = [|s_w - S|, s_w * S] with shape (n_vocab, 2 * n_components)
        X = np.concatenate([np.abs(self._S32 - s_w), self._S32 * s_w], axis=1).astype(np.float32)
        probs = self.clf.predict_proba(self.scaler.transform(X))[:, 1]

        n = len(probs)
        k = min(k, n)
        pool = np.argpartition(probs, n - k)[n - k :]
        pool = pool[np.argsort(probs[pool])[::-1]]
        return [(self.space.words[i], float(probs[i])) for i in pool]

    def cross_validate(
        self,
        antonym_pairs: list[tuple[str, str]],
        n_folds: int = 5,
        top_n: int = 10,
        neg_ratio: float = 3.0,
    ) -> dict:
        """
        5-fold cross-validation on antonym retrieval.

        For each fold: train on 4/5 pairs, evaluate retrieval on 1/5 pairs.
        """
        valid_pairs = [
            (w1, w2)
            for w1, w2 in antonym_pairs
            if self.space.score(w1) is not None and self.space.score(w2) is not None
        ]
        rng = np.random.RandomState(42)
        folds = make_folds(len(valid_pairs), n_folds, seed=42)

        fold_results = []
        for fold_idx in range(n_folds):
            train_pairs, test_pairs = train_test_pairs(valid_pairs, fold_idx, folds)

            # Train on this fold
            fold_clf = AntonymClassifier(self.space, self.model_type)
            fold_clf.fit(train_pairs, neg_ratio, rng)

            # Evaluate on test pairs
            hits = {1: 0, 5: 0, 10: 0}
            evaluated = 0
            for w1, w2 in test_pairs:
                best_rank = None
                for src, tgt in [(w1, w2), (w2, w1)]:
                    results = fold_clf.retrieve(src, top_n=top_n)
                    for j, (cand, _) in enumerate(results):
                        if cand == tgt:
                            rank = j + 1
                            if best_rank is None or rank < best_rank:
                                best_rank = rank
                            break
                evaluated += 1
                if best_rank is not None:
                    for k in [1, 5, 10]:
                        if best_rank <= k:
                            hits[k] += 1

            metrics = {f"hits_at_{k}": hits[k] / evaluated for k in [1, 5, 10]}
            fold_results.append(metrics)
            print(
                f"  Fold {fold_idx + 1}/{n_folds}: "
                f"Hits@1={metrics['hits_at_1']:.3f}  "
                f"Hits@5={metrics['hits_at_5']:.3f}  "
                f"Hits@10={metrics['hits_at_10']:.3f}  "
                f"({evaluated} pairs)"
            )

        # Aggregate
        agg = {}
        for k_str in ["hits_at_1", "hits_at_5", "hits_at_10"]:
            vals = [r[k_str] for r in fold_results]
            agg[k_str] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        return {"n_folds": n_folds, "total_pairs": len(valid_pairs), "metrics": agg}

    def top_antonym_axes(self, n: int = 10) -> list[tuple[int, float]]:
        """
        Return the axes most important for antonym prediction.

        For logistic regression: use coefficient magnitudes from abs_diff features.
        """
        if not self._fitted or self.model_type != "logistic":
            return []
        # Coefficients for the positive class
        coefs = self.clf.coef_[0]  # (200,)
        abs_diff_coefs = coefs[: self.space.n_components]  # first 100 = |s1-s2|
        ranked = np.argsort(np.abs(abs_diff_coefs))[::-1][:n]
        return [(int(k), float(abs_diff_coefs[k])) for k in ranked]
