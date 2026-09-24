"""
Experiment: union candidate pool for the reranker.

The reranker currently sees only the MLP's top-100. Other retrieval
signals find different antonyms:
  - knn-axis inversion (predicted axis, deployable — exp_axis_prediction)
  - procrustes map W: src -> tgt fit on train pairs (eval_translation)

Union pool = MLP top-100 ∪ axis top-20 ∪ procrustes top-20, reranked by
the 9-feature logistic model extended with per-source presence flags:
  in_mlp, in_axis, in_proc  (1/0 each)
For non-MLP candidates, mlp_score=0 and mlp_rank=POOL_N+1.

Reports per-source recall contribution and reranked Hits@k on the
70/15/15 holdout (seed 42, same split as eval_reranker).

Usage:
    pixi run python scripts/exp_union_pool.py
"""

import json
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher

sys.path.insert(0, ".")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.antonym_classifier import AntonymClassifier
from src.antonym_loader import extract_antonym_pairs
from src.eval_utils import compute_hits, unit_rows
from src.ica_transformer import load_ica_space
from src.semantic_operations import SemanticOperator
from src.word2vec_loader import Word2VecLoader

MLP_POOL = 100
AUX_POOL = 20
N_FEATURES = 12


class UnionReranker:
    """9 base features + 3 source-presence flags."""

    def __init__(self, space, model):
        self.space = space
        self.model = model
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self._passthrough = False

        S = space.S.astype(np.float32)
        norms = np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-10)
        self._S_norm = S / norms
        self._axis_std = np.maximum(np.std(space.S, axis=0), 1e-8)
        self._vocab_rank = model.key_to_index
        self._oov_rank = len(model.key_to_index)
        self._glove_unit: dict[str, np.ndarray | None] = {}

    def _glove_vec(self, w):
        if w not in self._glove_unit:
            if w not in self.model:
                self._glove_unit[w] = None
            else:
                v = self.model[w].astype(np.float64)
                n = np.linalg.norm(v)
                self._glove_unit[w] = v / n if n > 0 else None
        return self._glove_unit[w]

    def features(self, query, cand, mlp_rank, mlp_score, in_mlp, in_axis, in_proc):
        q_idx = self.space.word_to_idx.get(query)
        c_idx = self.space.word_to_idx.get(cand)
        if q_idx is None or c_idx is None:
            return np.zeros(N_FEATURES, dtype=np.float32)

        ica_cos = float(self._S_norm[q_idx] @ self._S_norm[c_idx])
        q_rank = self._vocab_rank.get(query, self._oov_rank) + 1
        c_rank = self._vocab_rank.get(cand, self._oov_rank) + 1
        freq_ratio = float(c_rank / q_rank)
        s1, s2 = self.space.S[q_idx], self.space.S[c_idx]
        max_zdiff = float(np.max(np.abs(s1 - s2) / self._axis_std))
        g1, g2 = self._glove_vec(query), self._glove_vec(cand)
        glove_cos = float(g1 @ g2) if g1 is not None and g2 is not None else 0.0
        morph = SequenceMatcher(None, query, cand).ratio()

        return np.array(
            [
                mlp_score,
                ica_cos,
                float(mlp_rank),
                freq_ratio,
                mlp_score * (-ica_cos),
                1.0 / mlp_rank,
                glove_cos,
                morph,
                max_zdiff,
                float(in_mlp),
                float(in_axis),
                float(in_proc),
            ],
            dtype=np.float32,
        )

    def fit(self, rows):
        """rows: list of (query, cand, mlp_rank, mlp_score, in_mlp, in_axis, in_proc, label)"""
        X = np.array([self.features(*r[:-1]) for r in rows], dtype=np.float32)
        y = np.array([r[-1] for r in rows], dtype=np.int32)
        if len(np.unique(y)) < 2:
            self._passthrough = True
            return self
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def rerank(self, query, cand_rows, top_n=10):
        """cand_rows: list of (cand, mlp_rank, mlp_score, in_mlp, in_axis, in_proc)"""
        if self._passthrough or not cand_rows:
            return [c[0] for c in cand_rows[:top_n]]
        feats = np.array(
            [self.features(query, *r) for r in cand_rows], dtype=np.float32
        )
        scores = self.clf.predict_proba(self.scaler.transform(feats))[:, 1]
        order = np.argsort(scores)[::-1][:top_n]
        return [cand_rows[i][0] for i in order]


def build_pool(query, mlp, axis_cands, proc_cands):
    """Return union pool rows: (cand, mlp_rank, mlp_score, in_mlp, in_axis, in_proc)."""
    mlp_cands = mlp.retrieve(query, top_n=MLP_POOL)
    pool: dict[str, list] = {}
    for r, (w, s) in enumerate(mlp_cands):
        pool[w] = [r + 1, s, 1, 0, 0]
    for w in axis_cands:
        if w in pool:
            pool[w][3] = 1
        else:
            pool[w] = [MLP_POOL + 1, 0.0, 0, 1, 0]
    for w in proc_cands:
        if w in pool:
            pool[w][4] = 1
        else:
            pool[w] = [MLP_POOL + 1, 0.0, 0, 0, 1]
    return [(w, *v) for w, v in pool.items()]


def main():
    t0 = time.time()
    print("Loading model and ICA space...")
    model = Word2VecLoader().load_glove(100)
    space = load_ica_space("results/ica_space")
    operator = SemanticOperator(model, space)
    axis_std = np.maximum(np.std(space.S, axis=0), 1e-8)

    pairs = [
        (a, b)
        for a, b in extract_antonym_pairs()
        if space.score(a) is not None and space.score(b) is not None
    ]
    rng = np.random.RandomState(42)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n = len(pairs)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    train_pairs = [pairs[i] for i in idx[:n_train]]
    val_pairs = [pairs[i] for i in idx[n_train : n_train + n_val]]
    test_pairs = [pairs[i] for i in idx[n_train + n_val :]]
    print(f"Split: {len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}")

    # --- Train MLP
    print("Training MLP...")
    mlp = AntonymClassifier(space, "mlp")
    mlp.fit(train_pairs, neg_ratio=3.0, rng=np.random.RandomState(42))

    # --- Train k-NN axis predictor on train pairs
    print("Building k-NN axis predictor...")
    S64 = space.S.astype(np.float64)
    S_norm = S64 / np.maximum(np.linalg.norm(S64, axis=1, keepdims=True), 1e-10)
    w2i = space.word_to_idx
    src_words, src_axes = [], []
    for a, b in train_pairs:
        for src, tgt in [(a, b), (b, a)]:
            s1, s2 = space.score(src), space.score(tgt)
            src_words.append(src)
            src_axes.append(int(np.argmax(np.abs(s1 - s2) / axis_std)))
    src_vecs = S_norm[np.array([w2i[w] for w in src_words])]

    def predict_axis(query):
        q = S_norm[w2i[query]]
        sims = src_vecs @ q
        j = int(np.argmax(sims))
        return src_axes[j]

    def axis_cands(query, top_n=AUX_POOL):
        ax = predict_axis(query)
        return [nb.word for nb in operator.axis_invert(query, ax, top_n=top_n)]

    # --- Fit procrustes map on train pairs (raw GloVe)
    print("Fitting procrustes map...")
    src_v, tgt_v = [], []
    for a, b in train_pairs:
        if a in model and b in model:
            src_v.append(model[a])
            tgt_v.append(model[b])
            src_v.append(model[b])
            tgt_v.append(model[a])
    X = unit_rows(np.array(src_v, dtype=np.float64))
    Y = unit_rows(np.array(tgt_v, dtype=np.float64))
    W, *_ = np.linalg.lstsq(X.T @ X, X.T @ Y, rcond=None)

    glove_pool = [w for w in model.index_to_key[:50000] if w in w2i]
    G = unit_rows(np.array([model[w] for w in glove_pool], dtype=np.float64))

    def proc_cands(query, top_n=AUX_POOL):
        if query not in model:
            return []
        q = model[query].astype(np.float64)
        q = q / max(np.linalg.norm(q), 1e-10)
        pred = q @ W
        pred = pred / max(np.linalg.norm(pred), 1e-10)
        sims = G @ pred
        order = np.argsort(sims)[::-1]
        out = []
        for i in order:
            w = glove_pool[i]
            if w != query:
                out.append(w)
            if len(out) >= top_n:
                break
        return out

    # --- Union recall on test
    print("\nUnion pool recall on test:")
    src_recall = defaultdict(int)
    union_hit = 0
    for w1, w2 in test_pairs:
        found = False
        for src, tgt in [(w1, w2), (w2, w1)]:
            mset = {w for w, _ in mlp.retrieve(src, top_n=MLP_POOL)}
            aset = set(axis_cands(src))
            pset = set(proc_cands(src))
            if tgt in mset:
                src_recall["mlp"] += 1
            if tgt in aset:
                src_recall["axis"] += 1
            if tgt in pset:
                src_recall["proc"] += 1
            if tgt in mset | aset | pset:
                found = True
        union_hit += found
    n_dir = len(test_pairs) * 2
    for s in ["mlp", "axis", "proc"]:
        print(f"  {s:5s}: {src_recall[s] / n_dir:.1%} of directions")
    print(f"  union: {union_hit / len(test_pairs):.1%} of pairs")

    # --- Train union reranker on val
    print("\nTraining union reranker on val set...")
    rows = []
    for w1, w2 in val_pairs:
        for q, t in [(w1, w2), (w2, w1)]:
            for cand, mr, ms, im, ia, ip in build_pool(q, mlp, axis_cands(q), proc_cands(q)):
                rows.append((q, cand, mr, ms, im, ia, ip, 1 if cand == t else 0))
    rr = UnionReranker(space, model).fit(rows)
    names = [
        "mlp_score", "ica_cosine", "mlp_rank", "freq_ratio", "interaction",
        "inv_rank", "glove_cos", "morph_sim", "max_zdiff",
        "in_mlp", "in_axis", "in_proc",
    ]
    print("  Feature weights:")
    for nm, c in sorted(zip(names, rr.clf.coef_[0], strict=True), key=lambda x: -abs(x[1])):
        print(f"    {nm:12s} {c:+.3f}")

    def fn(src):
        pool = build_pool(src, mlp, axis_cands(src), proc_cands(src))
        return rr.rerank(src, pool, top_n=10)

    hits, _ = compute_hits(test_pairs, fn)
    print(f"\nUnion reranker: @1={hits[1]:.1%} @5={hits[5]:.1%} @10={hits[10]:.1%}")

    out = {
        "per_source_recall": {s: src_recall[s] / n_dir for s in ["mlp", "axis", "proc"]},
        "union_recall": union_hit / len(test_pairs),
        "hits": {str(k): v for k, v in hits.items()},
        "feature_weights": dict(zip(names, rr.clf.coef_[0].tolist(), strict=True)),
        "elapsed_s": time.time() - t0,
    }
    with open("results/exp_union_pool.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results/exp_union_pool.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
