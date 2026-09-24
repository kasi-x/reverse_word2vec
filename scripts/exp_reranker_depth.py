"""
Experiment: does a deeper MLP candidate pool + extra reranker features
improve antonym retrieval?

Baseline pipeline caps the reranker at the MLP's top-10 candidates, so
Hits@10 = MLP Hits@10 by construction (~36.6%). This script ablates:

  pool_n  in {10, 50, 100}   — how deep the MLP ranking the reranker sees
  feats   in {base, ext}     — 6 baseline signals vs +3 new ones:
      glove_cos   raw GloVe cosine(query, cand)
      morph_sim   difflib similarity (inflectional confusables penalty)
      max_zdiff   max_k |s1[k]-s2[k]| / std_k  (dominant-axis signature)

Same 70/15/15 split and seed as scripts/eval_reranker.py.

Usage:
    pixi run python scripts/exp_reranker_depth.py
"""

import json
import sys
import time
from difflib import SequenceMatcher

sys.path.insert(0, ".")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.antonym_classifier import AntonymClassifier
from src.antonym_loader import extract_antonym_pairs
from src.eval_utils import compute_hits
from src.ica_transformer import load_ica_space
from src.word2vec_loader import Word2VecLoader

POOL_DEPTHS = [10, 50, 100]
FEATURE_SETS = ["base", "ext"]


def morph_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class ExpReranker:
    """Self-contained reranker with configurable pool depth + features."""

    def __init__(self, space, model, feat_set: str):
        self.space = space
        self.model = model
        self.feat_set = feat_set
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

        S = space.S.astype(np.float32)
        norms = np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-10)
        self._S_norm = S / norms
        self._axis_std = np.maximum(np.std(space.S, axis=0), 1e-8)

        # Raw GloVe unit vectors for cosine
        self._vocab_rank = model.key_to_index
        self._oov_rank = len(model.key_to_index)
        self._glove_cache: dict[str, np.ndarray] = {}

    def _glove_unit(self, w: str) -> np.ndarray | None:
        v = self._glove_cache.get(w)
        if v is None:
            if w not in self.model:
                self._glove_cache[w] = None
                return None
            raw = self.model[w].astype(np.float64)
            n = np.linalg.norm(raw)
            v = raw / n if n > 0 else None
            self._glove_cache[w] = v
        return v

    def features(self, query, cand, mlp_rank, mlp_score) -> np.ndarray:
        q_idx = self.space.word_to_idx.get(query)
        c_idx = self.space.word_to_idx.get(cand)
        n = 9 if self.feat_set == "ext" else 6
        if q_idx is None or c_idx is None:
            return np.zeros(n, dtype=np.float32)

        ica_cos = float(self._S_norm[q_idx] @ self._S_norm[c_idx])
        q_rank = self._vocab_rank.get(query, self._oov_rank) + 1
        c_rank = self._vocab_rank.get(cand, self._oov_rank) + 1
        freq_ratio = float(c_rank / q_rank)
        base = [
            mlp_score,
            ica_cos,
            float(mlp_rank),
            freq_ratio,
            mlp_score * (-ica_cos),
            1.0 / mlp_rank,
        ]
        if self.feat_set == "base":
            return np.array(base, dtype=np.float32)

        s1 = self.space.S[q_idx]
        s2 = self.space.S[c_idx]
        max_zdiff = float(np.max(np.abs(s1 - s2) / self._axis_std))
        g1 = self._glove_unit(query)
        g2 = self._glove_unit(cand)
        glove_cos = float(g1 @ g2) if g1 is not None and g2 is not None else 0.0
        return np.array(base + [glove_cos, morph_sim(query, cand), max_zdiff], dtype=np.float32)

    def fit(self, pairs, mlp: AntonymClassifier, pool_n: int):
        X, y = [], []
        for w1, w2 in pairs:
            for q, t in [(w1, w2), (w2, w1)]:
                cands = mlp.retrieve(q, top_n=pool_n)
                for r, (c, s) in enumerate(cands):
                    X.append(self.features(q, c, r + 1, s))
                    y.append(1 if c == t else 0)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        if len(np.unique(y)) < 2:
            self._passthrough = True
            return self
        self._passthrough = False
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def rerank(self, query, cands, top_n=10):
        if self._passthrough:
            return cands[:top_n]
        feats = np.array(
            [self.features(query, c, i + 1, s) for i, (c, s) in enumerate(cands)],
            dtype=np.float32,
        )
        scores = self.clf.predict_proba(self.scaler.transform(feats))[:, 1]
        order = np.argsort(scores)[::-1][:top_n]
        return [(cands[i][0], float(scores[i])) for i in order]


def main():
    t0 = time.time()
    print("Loading model and ICA space...")
    model = Word2VecLoader().load_glove(100)
    space = load_ica_space("results/ica_space")

    pairs = [
        (a, b)
        for a, b in extract_antonym_pairs()
        if space.score(a) is not None and space.score(b) is not None
    ]
    print(f"Valid pairs: {len(pairs)}")

    rng = np.random.RandomState(42)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n = len(pairs)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    train_pairs = [pairs[i] for i in idx[:n_train]]
    val_pairs = [pairs[i] for i in idx[n_train : n_train + n_val]]
    test_pairs = [pairs[i] for i in idx[n_train + n_val :]]
    print(f"Split: {len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}")

    print("Training MLP on train split...")
    mlp = AntonymClassifier(space, "mlp")
    mlp.fit(train_pairs, neg_ratio=3.0, rng=rng)

    # Pool recall ceiling: how often is the target even in the MLP pool?
    print("\nMLP pool recall on test (target present in top-N):")
    pool_recall = {}
    for depth in POOL_DEPTHS:
        hit = 0
        for w1, w2 in test_pairs:
            found = False
            for src, tgt in [(w1, w2), (w2, w1)]:
                cands = [w for w, _ in mlp.retrieve(src, top_n=depth)]
                if tgt in cands:
                    found = True
                    break
            hit += found
        pool_recall[depth] = hit / len(test_pairs)
        print(f"  top-{depth:3d}: {pool_recall[depth]:.1%}")

    results = {}
    for depth in POOL_DEPTHS:
        for fs in FEATURE_SETS:
            print(f"\n=== pool={depth}, features={fs} ===")
            rr = ExpReranker(space, model, fs).fit(val_pairs, mlp, depth)

            def fn(src, _rr=rr, _d=depth):
                cands = mlp.retrieve(src, top_n=_d)
                return [w for w, _ in _rr.rerank(src, cands, top_n=10)]

            hits, _ = compute_hits(test_pairs, fn)
            key = f"pool{depth}_{fs}"
            results[key] = {str(k): v for k, v in hits.items()}
            print(f"  Hits@1={hits[1]:.1%}  @5={hits[5]:.1%}  @10={hits[10]:.1%}")

    # MLP-only reference at top-10
    mlp_hits, _ = compute_hits(test_pairs, lambda s: [w for w, _ in mlp.retrieve(s, top_n=10)])
    results["mlp_only_top10"] = {str(k): v for k, v in mlp_hits.items()}
    print(f"\nMLP only top-10: @1={mlp_hits[1]:.1%} @5={mlp_hits[5]:.1%} @10={mlp_hits[10]:.1%}")

    out = {
        "config": {"pool_depths": POOL_DEPTHS, "feature_sets": FEATURE_SETS},
        "pool_recall": {str(k): v for k, v in pool_recall.items()},
        "results": results,
        "elapsed_s": time.time() - t0,
    }
    with open("results/exp_reranker_depth.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/exp_reranker_depth.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
