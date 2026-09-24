"""
Antonym Reranker.

Takes MLP classifier's top-N candidates and re-ranks them using
additional signals that filter out noise (rare/irrelevant words).

Features per candidate:
  1. mlp_score   - MLP P(antonym) probability
  2. ica_cosine  - cosine similarity in ICA score space
  3. mlp_rank    - position in MLP ranking
  4. freq_ratio  - candidate_vocab_rank / query_vocab_rank
                   (large → candidate is rarer than query → likely noise)
  5. interaction - mlp_score * (-ica_cosine)  (high when prob high AND cosine negative)
  6. inv_rank    - 1 / mlp_rank
  7. glove_cos   - raw GloVe cosine(query, cand)
  8. morph_sim   - difflib string similarity (stem-sharing antonyms like
                   unhappy score high; inflectional confusables too)
  9. max_zdiff   - max_k |s1[k]-s2[k]| / std_k (dominant-axis signature)

Insight: in GloVe-100 + ICA, noise words like "householder" and "passerine"
achieve high MLP scores and high ICA dissimilarity, but they are rare and
their ICA cosine with the query is near-random. The freq_ratio signal
effectively filters them out.
"""

from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.antonym_classifier import AntonymClassifier
from src.eval_utils import make_folds, train_test_pairs
from src.ica_transformer import ICASpace


def _rank_of(candidates: list[tuple[str, float]], target: str) -> int | None:
    """1-based rank of `target` in a (word, score) list, or None."""
    for j, (w, _) in enumerate(candidates):
        if w == target:
            return j + 1
    return None


class Reranker:
    """
    Learns to rerank MLP antonym candidates using secondary features.

    Training procedure:
      For each training pair (w1, w2):
        - Get MLP's top-10 candidates for w1 (and w2)
        - Label each candidate as 1 (== correct antonym) or 0
        - Fit a logistic regression over the 9-dimensional feature vector

    Inference:
      - Get MLP top-N candidates
      - Score each with the learned linear model
      - Return candidates sorted by reranker score
    """

    N_FEATURES = 9

    def __init__(self, space: ICASpace, model: KeyedVectors):
        self.space = space
        self.model = model
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self._fitted = False
        # Set when the training set has a single class (MLP never surfaced the
        # target): no logistic fit is possible, so rerank() passes candidates
        # through in MLP order.
        self._passthrough = False

        # Precompute ICA-normalised row vectors for fast batch cosine
        S = space.S.astype(np.float32)
        norms = np.linalg.norm(S, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._S_norm = S / norms  # (vocab, n_components)

        # Vocab-rank map: lower index == more frequent (GloVe ordering)
        self._vocab_rank: dict[str, int] = model.key_to_index
        self._oov_rank = len(model.key_to_index)

        self._axis_std = np.maximum(np.std(space.S, axis=0), 1e-8)
        # Lazily filled unit-norm GloVe vectors for the glove_cos feature
        self._glove_unit: dict[str, np.ndarray | None] = {}

    # ------------------------------------------------------------------ #
    # Feature extraction                                                   #
    # ------------------------------------------------------------------ #

    def _features(
        self,
        query: str,
        candidate: str,
        mlp_rank: int,
        mlp_score: float,
    ) -> np.ndarray:
        """Return 9-d feature vector for a (query, candidate) pair."""
        q_idx = self.space.word_to_idx.get(query)
        c_idx = self.space.word_to_idx.get(candidate)
        if q_idx is None or c_idx is None:
            return np.zeros(self.N_FEATURES, dtype=np.float32)

        # ICA cosine
        ica_cosine = float(self._S_norm[q_idx] @ self._S_norm[c_idx])

        # Frequency ratio proxy: vocab index in GloVe (lower = more frequent)
        q_rank = self._vocab_rank.get(query, self._oov_rank) + 1
        c_rank = self._vocab_rank.get(candidate, self._oov_rank) + 1
        freq_ratio = float(c_rank / q_rank)

        inv_rank = 1.0 / mlp_rank
        interaction = mlp_score * (-ica_cosine)

        # Extended signals
        s1 = self.space.S[q_idx]
        s2 = self.space.S[c_idx]
        max_zdiff = float(np.max(np.abs(s1 - s2) / self._axis_std))
        g1 = self._glove_vec(query)
        g2 = self._glove_vec(candidate)
        glove_cos = float(g1 @ g2) if g1 is not None and g2 is not None else 0.0
        morph = SequenceMatcher(None, query, candidate).ratio()

        return np.array(
            [
                mlp_score,
                ica_cosine,
                float(mlp_rank),
                freq_ratio,
                interaction,
                inv_rank,
                glove_cos,
                morph,
                max_zdiff,
            ],
            dtype=np.float32,
        )

    def _glove_vec(self, word: str) -> np.ndarray | None:
        """Unit-norm raw GloVe vector, cached; None for OOV."""
        if word not in self._glove_unit:
            if word not in self.model:
                self._glove_unit[word] = None
            else:
                v = self.model[word].astype(np.float64)
                n = np.linalg.norm(v)
                self._glove_unit[word] = v / n if n > 0 else None
        return self._glove_unit[word]

    def _build_dataset(
        self,
        antonym_pairs: list[tuple[str, str]],
        classifier: AntonymClassifier,
        top_n: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build (X, y) reranker training set from MLP predictions.

        For each pair (w1, w2), queries MLP for top-N candidates for both
        directions and labels the correct antonym as positive.
        """
        X, y = [], []
        for w1, w2 in antonym_pairs:
            for query, target in [(w1, w2), (w2, w1)]:
                candidates = classifier.retrieve(query, top_n=top_n)
                if not candidates:
                    continue
                for rank_0, (cand, score) in enumerate(candidates):
                    feat = self._features(query, cand, rank_0 + 1, score)
                    X.append(feat)
                    y.append(1 if cand == target else 0)

        if not X:
            raise ValueError("No training examples generated.")
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Fit / rerank                                                         #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        antonym_pairs: list[tuple[str, str]],
        classifier: AntonymClassifier,
        top_n: int = 10,
    ) -> Reranker:
        """Train the reranker on MLP predictions for antonym_pairs."""
        X, y = self._build_dataset(antonym_pairs, classifier, top_n)
        if len(np.unique(y)) < 2:
            self._passthrough = True
            self._fitted = True
            return self
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self._fitted = True
        return self

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Re-rank MLP candidates for a query.

        Args:
            query: Query word.
            candidates: [(word, mlp_score), ...] from AntonymClassifier.retrieve().
            top_n: How many to return.

        Returns:
            Re-ranked [(word, reranker_score), ...].
        """
        if not self._fitted:
            raise RuntimeError("Reranker not fitted. Call fit() first.")
        if not candidates:
            return []
        if self._passthrough:
            return candidates[:top_n]

        feats = np.array(
            [
                self._features(query, cand, rank + 1, score)
                for rank, (cand, score) in enumerate(candidates)
            ],
            dtype=np.float32,
        )
        feats_scaled = self.scaler.transform(feats)
        scores = self.clf.predict_proba(feats_scaled)[:, 1]

        ranked = np.argsort(scores)[::-1]
        result = []
        for idx in ranked[:top_n]:
            result.append((candidates[idx][0], float(scores[idx])))
        return result

    def feature_weights(self) -> dict:
        """Return feature name -> coefficient mapping."""
        if not self._fitted or self._passthrough:
            return {}
        names = [
            "mlp_score",
            "ica_cosine",
            "mlp_rank",
            "freq_ratio",
            "interaction",
            "inv_rank",
            "glove_cos",
            "morph_sim",
            "max_zdiff",
        ]
        coefs = self.clf.coef_[0]
        return dict(zip(names, coefs.tolist(), strict=False))

    # ------------------------------------------------------------------ #
    # Cross-validation                                                     #
    # ------------------------------------------------------------------ #

    def cross_validate(
        self,
        antonym_pairs: list[tuple[str, str]],
        n_folds: int = 5,
        top_n: int = 10,
        mlp_top_n: int = 10,
        neg_ratio: float = 3.0,
    ) -> dict:
        """
        Full pipeline 5-fold CV: train MLP + reranker on 4/5, eval on 1/5.

        This avoids leakage: the reranker never sees test examples during
        training (MLP or reranker).
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

            # Train MLP on fold's training set
            fold_mlp = AntonymClassifier(self.space, "mlp")
            fold_mlp.fit(train_pairs, neg_ratio, rng)

            # Train reranker on same training set using fold's MLP
            fold_reranker = Reranker(self.space, self.model)
            fold_reranker.fit(train_pairs, fold_mlp, mlp_top_n)

            # Evaluate on test pairs: MLP only vs MLP + reranker
            mlp_hits = {1: 0, 5: 0, 10: 0}
            rer_hits = {1: 0, 5: 0, 10: 0}
            evaluated = 0
            cand_n = max(top_n, mlp_top_n)

            for w1, w2 in test_pairs:
                evaluated += 1
                best = {"mlp": None, "rer": None}
                for src, tgt in [(w1, w2), (w2, w1)]:
                    # One retrieval per direction feeds both evaluations
                    cands = fold_mlp.retrieve(src, top_n=cand_n)
                    rank = _rank_of(cands[:top_n], tgt)
                    if rank is not None and (best["mlp"] is None or rank < best["mlp"]):
                        best["mlp"] = rank
                    reranked = fold_reranker.rerank(src, cands[:mlp_top_n], top_n=top_n)
                    rank = _rank_of(reranked, tgt)
                    if rank is not None and (best["rer"] is None or rank < best["rer"]):
                        best["rer"] = rank
                for k in [1, 5, 10]:
                    if best["mlp"] is not None and best["mlp"] <= k:
                        mlp_hits[k] += 1
                    if best["rer"] is not None and best["rer"] <= k:
                        rer_hits[k] += 1

            mlp_metrics = {f"hits_at_{k}": mlp_hits[k] / evaluated for k in [1, 5, 10]}
            rer_metrics = {f"hits_at_{k}": rer_hits[k] / evaluated for k in [1, 5, 10]}
            fold_results.append({"mlp": mlp_metrics, "reranker": rer_metrics})

            print(
                f"  Fold {fold_idx + 1}/{n_folds}: "
                f"MLP @1={mlp_metrics['hits_at_1']:.3f}  "
                f"Reranker @1={rer_metrics['hits_at_1']:.3f}  "
                f"({evaluated} pairs)"
            )

        # Aggregate
        agg = {"mlp": {}, "reranker": {}}
        for method in ["mlp", "reranker"]:
            for k_str in ["hits_at_1", "hits_at_5", "hits_at_10"]:
                vals = [r[method][k_str] for r in fold_results]
                agg[method][k_str] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        return {
            "n_folds": n_folds,
            "total_pairs": len(valid_pairs),
            "metrics": agg,
        }
