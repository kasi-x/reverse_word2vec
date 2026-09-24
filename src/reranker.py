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
 10. in_mlp      - candidate came from the MLP pool
 11. in_axis     - candidate came from k-NN predicted-axis inversion
 12. in_proc     - candidate came from the procrustes translation map

CandidateSources builds the union pool (MLP top-N ∪ axis top-M ∪
procrustes top-M); the reranker scores every member of the union.

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
from src.eval_utils import make_folds, train_test_pairs, unit_rows
from src.ica_transformer import ICASpace
from src.semantic_operations import SemanticOperator


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

    N_FEATURES = 14

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

    FEATURE_NAMES = [
        "mlp_score",
        "ica_cosine",
        "mlp_rank",
        "freq_ratio",
        "interaction",
        "inv_rank",
        "glove_cos",
        "morph_sim",
        "max_zdiff",
        "in_mlp",
        "in_axis",
        "in_proc",
        "in_ica_map",
        "in_morph",
    ]

    # ------------------------------------------------------------------ #
    # Feature extraction                                                   #
    # ------------------------------------------------------------------ #

    def _features(
        self,
        query: str,
        candidate: str,
        mlp_rank: int,
        mlp_score: float,
        in_mlp: int = 1,
        in_axis: int = 0,
        in_proc: int = 0,
        in_ica_map: int = 0,
        in_morph: int = 0,
    ) -> np.ndarray:
        """Return 14-d feature vector for a (query, candidate) pair."""
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
                float(in_mlp),
                float(in_axis),
                float(in_proc),
                float(in_ica_map),
                float(in_morph),
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
        sources: CandidateSources | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build (X, y) reranker training set from candidate pools.

        For each pair (w1, w2), builds the candidate pool for both
        directions and labels the correct antonym as positive. With
        `sources`, the pool is the union of MLP/axis/procrustes
        candidates; otherwise it is MLP top-N only.
        """
        X, y = [], []
        for w1, w2 in antonym_pairs:
            for query, target in [(w1, w2), (w2, w1)]:
                if sources is not None:
                    pool = sources.pool(query)
                else:
                    pool = [
                        (cand, rank_0 + 1, score, 1, 0, 0, 0, 0)
                        for rank_0, (cand, score) in enumerate(
                            classifier.retrieve(query, top_n=top_n)
                        )
                    ]
                for cand, mlp_rank, mlp_score, im, ia, ip, ic, imo in pool:
                    feat = self._features(query, cand, mlp_rank, mlp_score, im, ia, ip, ic, imo)
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
        sources: CandidateSources | None = None,
    ) -> Reranker:
        """Train the reranker on candidate pools for antonym_pairs."""
        X, y = self._build_dataset(antonym_pairs, classifier, top_n, sources)
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
        candidates: list[tuple],
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Re-rank candidates for a query.

        Args:
            query: Query word.
            candidates: either [(word, mlp_score), ...] from
                AntonymClassifier.retrieve(), or pool rows
                (word, mlp_rank, mlp_score, in_mlp, in_axis, in_proc, in_ica_map, in_morph)
                from CandidateSources.pool().
            top_n: How many to return.

        Returns:
            Re-ranked [(word, reranker_score), ...].
        """
        if not self._fitted:
            raise RuntimeError("Reranker not fitted. Call fit() first.")
        if not candidates:
            return []
        rows = []
        for rank, item in enumerate(candidates):
            if len(item) == 2:
                cand, score = item
                rows.append((cand, rank + 1, score, 1, 0, 0, 0, 0))
            else:
                rows.append(item)
        if self._passthrough:
            return [(r[0], r[2]) for r in rows[:top_n]]

        feats = np.array(
            [self._features(query, *row) for row in rows],
            dtype=np.float32,
        )
        feats_scaled = self.scaler.transform(feats)
        scores = self.clf.predict_proba(feats_scaled)[:, 1]

        ranked = np.argsort(scores)[::-1]
        result = []
        for idx in ranked[:top_n]:
            result.append((rows[idx][0], float(scores[idx])))
        return result

    def feature_weights(self) -> dict:
        """Return feature name -> coefficient mapping."""
        if not self._fitted or self._passthrough:
            return {}
        coefs = self.clf.coef_[0]
        return dict(zip(self.FEATURE_NAMES, coefs.tolist(), strict=True))

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
        aux_top_n: int = 20,
        extra_pairs: list[tuple[str, str]] | None = None,
    ) -> dict:
        """
        Full pipeline 5-fold CV: train MLP + reranker on 4/5, eval on 1/5.

        This avoids leakage: the reranker never sees test examples during
        training (MLP or reranker). `extra_pairs` (e.g. ConceptNet) are
        appended to the reranker training set only — the MLP and the
        candidate sources stay on `antonym_pairs` (extra data is noisier
        and hurts them). Callers must remove any overlap with the
        evaluation pairs.
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
            train_aug = train_pairs + extra_pairs if extra_pairs else train_pairs

            # Train MLP on fold's WordNet training set only
            fold_mlp = AntonymClassifier(self.space, "mlp")
            fold_mlp.fit(train_pairs, neg_ratio, rng)

            # Train reranker on the augmented set; sources stay on the
            # fold's WordNet pairs (CN noise degrades them too).
            fold_sources = CandidateSources(self.space, self.model)
            fold_sources.fit(train_pairs, fold_mlp, mlp_top_n, aux_top_n)
            fold_reranker = Reranker(self.space, self.model)
            fold_reranker.fit(train_aug, fold_mlp, mlp_top_n, fold_sources)

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
                    pool = fold_sources.pool(src)
                    reranked = fold_reranker.rerank(src, pool, top_n=top_n)
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


class CandidateSources:
    """
    Builds the union candidate pool for a query:

      MLP top-N  ∪  k-NN predicted-axis inversion top-M  ∪  procrustes top-M
      ∪  ICA-score translation map top-M  ∪  morphological transforms

    - Axis predictor: the query's nearest neighbour among TRAIN source
      words in ICA space votes for the axis (deployable — no target needed).
    - Procrustes map: W fit by least squares on train pairs maps a source
      GloVe vector toward its antonym's vector; nearest neighbours of
      W·v(query) are candidates.
    - ICA map: W_ica fit by least squares on train pairs maps a source
      ICA score vector toward its antonym's; nearest neighbours of
      s(query)·W_ica in score space are candidates.

    - Morphological: prefix transforms (un-/in-/dis-/...) generate
      vocabulary candidates directly — no model needed.
    """

    def __init__(self, space: ICASpace, model: KeyedVectors):
        self.space = space
        self.model = model
        self._operator = SemanticOperator(model, space)
        self._axis_std = np.maximum(np.std(space.S, axis=0), 1e-8)
        S = space.S.astype(np.float64)
        self._S_norm = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-10)
        self._fitted = False

    def fit(
        self,
        train_pairs: list[tuple[str, str]],
        classifier: AntonymClassifier,
        mlp_top_n: int = 100,
        aux_top_n: int = 20,
    ) -> CandidateSources:
        """Fit axis predictor + procrustes map on training pairs."""
        self.classifier = classifier
        self.mlp_top_n = mlp_top_n
        self.aux_top_n = aux_top_n

        # --- k-NN axis predictor (k=1 vote over train source words)
        src_words, src_axes = [], []
        for a, b in train_pairs:
            for src, tgt in [(a, b), (b, a)]:
                s1, s2 = self.space.score(src), self.space.score(tgt)
                if s1 is None or s2 is None:
                    continue
                src_words.append(src)
                src_axes.append(int(np.argmax(np.abs(s1 - s2) / self._axis_std)))
        self._src_axes = src_axes
        self._src_vecs = self._S_norm[np.array([self.space.word_to_idx[w] for w in src_words])]

        # --- Procrustes map on raw GloVe (both directions)
        src_v, tgt_v = [], []
        for a, b in train_pairs:
            if a in self.model and b in self.model:
                src_v.append(self.model[a])
                tgt_v.append(self.model[b])
                src_v.append(self.model[b])
                tgt_v.append(self.model[a])
        X = unit_rows(np.array(src_v, dtype=np.float64))
        Y = unit_rows(np.array(tgt_v, dtype=np.float64))
        self._W, *_ = np.linalg.lstsq(X.T @ X, X.T @ Y, rcond=None)

        # --- ICA-score translation map (both directions)
        ica_src, ica_tgt = [], []
        for a, b in train_pairs:
            s_a, s_b = self.space.score(a), self.space.score(b)
            if s_a is not None and s_b is not None:
                ica_src.append(s_a)
                ica_tgt.append(s_b)
                ica_src.append(s_b)
                ica_tgt.append(s_a)
        Xi = np.array(ica_src, dtype=np.float64)
        Yi = np.array(ica_tgt, dtype=np.float64)
        self._W_ica, *_ = np.linalg.lstsq(Xi.T @ Xi, Xi.T @ Yi, rcond=None)

        # Candidate vocabulary: GloVe top-50k ∩ ICA vocab
        self._glove_pool = [
            w for w in self.model.index_to_key[:50000] if w in self.space.word_to_idx
        ]
        self._G = unit_rows(np.array([self.model[w] for w in self._glove_pool], dtype=np.float64))
        self._fitted = True
        return self

    def _predict_axis(self, query: str) -> int:
        q_idx = self.space.word_to_idx[query]
        sims = self._src_vecs @ self._S_norm[q_idx]
        return self._src_axes[int(np.argmax(sims))]

    def axis_candidates(self, query: str) -> list[str]:
        """Top-M words from inverting the predicted axis."""
        ax = self._predict_axis(query)
        return [nb.word for nb in self._operator.axis_invert(query, ax, top_n=self.aux_top_n)]

    def proc_candidates(self, query: str) -> list[str]:
        """Top-M words nearest to W·v(query) in raw GloVe."""
        if query not in self.model:
            return []
        q = self.model[query].astype(np.float64)
        q = q / max(np.linalg.norm(q), 1e-10)
        pred = q @ self._W
        pred = pred / max(np.linalg.norm(pred), 1e-10)
        sims = self._G @ pred
        out = []
        for i in np.argsort(sims)[::-1]:
            w = self._glove_pool[i]
            if w != query:
                out.append(w)
            if len(out) >= self.aux_top_n:
                break
        return out

    def ica_map_candidates(self, query: str) -> list[str]:
        """Top-M words nearest to s(query)·W_ica in ICA score space."""
        q = self.space.score(query)
        if q is None:
            return []
        pred = q.astype(np.float64) @ self._W_ica
        pred = pred / max(np.linalg.norm(pred), 1e-10)
        sims = self._S_norm @ pred
        out = []
        for i in np.argsort(sims)[::-1]:
            w = self.space.words[i]
            if w != query:
                out.append(w)
            if len(out) >= self.aux_top_n:
                break
        return out

    # Antonym-forming prefixes; "re-" excluded (re-do ≠ undo antonym).
    _MORPH_PREFIXES = (
        "un",
        "in",
        "im",
        "il",
        "ir",
        "dis",
        "non",
        "anti",
        "de",
        "mis",
        "over",
        "under",
        "counter",
    )

    def morph_candidates(self, query: str) -> list[str]:
        """Vocabulary words reachable by adding/stripping an antonym prefix."""
        out: list[str] = []
        for p in self._MORPH_PREFIXES:
            if query.startswith(p) and len(query) > len(p) + 2:
                stem = query[len(p) :]
                if stem in self.space.word_to_idx:
                    out.append(stem)
            cand = p + query
            if cand in self.space.word_to_idx:
                out.append(cand)
        return out

    def pool(self, query: str) -> list[tuple[str, int, float, int, int, int, int, int]]:
        """
        Union pool rows: (word, mlp_rank, mlp_score, in_mlp, in_axis,
        in_proc, in_ica_map, in_morph).

        Non-MLP members get mlp_rank = mlp_top_n + 1 and mlp_score = 0.
        """
        if not self._fitted:
            raise RuntimeError("CandidateSources not fitted. Call fit() first.")
        pool: dict[str, list] = {}
        for r, (w, s) in enumerate(self.classifier.retrieve(query, top_n=self.mlp_top_n)):
            pool[w] = [r + 1, s, 1, 0, 0, 0, 0]
        for w in self.axis_candidates(query):
            if w in pool:
                pool[w][3] = 1
            else:
                pool[w] = [self.mlp_top_n + 1, 0.0, 0, 1, 0, 0, 0]
        for w in self.proc_candidates(query):
            if w in pool:
                pool[w][4] = 1
            else:
                pool[w] = [self.mlp_top_n + 1, 0.0, 0, 0, 1, 0, 0]
        for w in self.ica_map_candidates(query):
            if w in pool:
                pool[w][5] = 1
            else:
                pool[w] = [self.mlp_top_n + 1, 0.0, 0, 0, 0, 1, 0]
        for w in self.morph_candidates(query):
            if w in pool:
                pool[w][6] = 1
            else:
                pool[w] = [self.mlp_top_n + 1, 0.0, 0, 0, 0, 0, 1]
        return [(w, *v) for w, v in pool.items()]
