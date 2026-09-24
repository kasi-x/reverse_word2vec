"""
Experiment: 4-source union pool — add the ICA-score translation map.

exp_ica_translation showed W_ica (least squares on ICA scores) retrieves
@1=26.6% standalone — stronger than the MLP. This script adds it as a
4th candidate source to the union pool and re-reranks.

Sources: MLP top-100 ∪ knn-axis top-20 ∪ procrustes top-20 ∪ ica-map top-20
Features: 12 union features + in_ica_map flag (13 total).

Usage:
    pixi run python scripts/exp_union4.py
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
from src.reranker import CandidateSources
from src.word2vec_loader import Word2VecLoader

MLP_POOL = 100
AUX_POOL = 20
N_FEATURES = 13


class Union4Reranker:
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

    def features(self, query, cand, mlp_rank, mlp_score, im, ia, ip, ic):
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
                mlp_score, ica_cos, float(mlp_rank), freq_ratio,
                mlp_score * (-ica_cos), 1.0 / mlp_rank,
                glove_cos, morph, max_zdiff,
                float(im), float(ia), float(ip), float(ic),
            ],
            dtype=np.float32,
        )

    def fit(self, rows):
        X = np.array([self.features(*r[:-1]) for r in rows], dtype=np.float32)
        y = np.array([r[-1] for r in rows], dtype=np.int32)
        if len(np.unique(y)) < 2:
            self._passthrough = True
            return self
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def rerank(self, query, cand_rows, top_n=10):
        if self._passthrough or not cand_rows:
            return [c[0] for c in cand_rows[:top_n]]
        feats = np.array([self.features(query, *r) for r in cand_rows], dtype=np.float32)
        scores = self.clf.predict_proba(self.scaler.transform(feats))[:, 1]
        order = np.argsort(scores)[::-1][:top_n]
        return [cand_rows[i][0] for i in order]


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
    rng = np.random.RandomState(42)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n = len(pairs)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    train_pairs = [pairs[i] for i in idx[:n_train]]
    val_pairs = [pairs[i] for i in idx[n_train : n_train + n_val]]
    test_pairs = [pairs[i] for i in idx[n_train + n_val :]]
    print(f"Split: {len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}")

    print("Training MLP + candidate sources...")
    mlp = AntonymClassifier(space, "mlp")
    mlp.fit(train_pairs, neg_ratio=3.0, rng=np.random.RandomState(42))
    sources = CandidateSources(space, model)
    sources.fit(train_pairs, mlp, mlp_top_n=MLP_POOL, aux_top_n=AUX_POOL)

    # --- ICA translation map (4th source)
    src_v, tgt_v = [], []
    for a, b in train_pairs:
        src_v.append(space.score(a))
        tgt_v.append(space.score(b))
        src_v.append(space.score(b))
        tgt_v.append(space.score(a))
    X = np.array(src_v, dtype=np.float64)
    Y = np.array(tgt_v, dtype=np.float64)
    W_ica, *_ = np.linalg.lstsq(X.T @ X, X.T @ Y, rcond=None)

    S64 = space.S.astype(np.float64)
    S_norm = S64 / np.maximum(np.linalg.norm(S64, axis=1, keepdims=True), 1e-10)

    def ica_map_cands(query, top_n=AUX_POOL):
        q = space.score(query)
        if q is None:
            return []
        pred = q @ W_ica
        pred = pred / max(np.linalg.norm(pred), 1e-10)
        sims = S_norm @ pred
        out = []
        for i in np.argsort(sims)[::-1]:
            w = space.words[i]
            if w != query:
                out.append(w)
            if len(out) >= top_n:
                break
        return out

    def pool4(query):
        """Union rows: (cand, mlp_rank, mlp_score, in_mlp, in_axis, in_proc, in_ica)."""
        pool: dict[str, list] = {}
        for w, mr, ms, im, ia, ip in sources.pool(query):
            pool[w] = [mr, ms, im, ia, ip, 0]
        for w in ica_map_cands(query):
            if w in pool:
                pool[w][5] = 1
            else:
                pool[w] = [MLP_POOL + 1, 0.0, 0, 0, 0, 1]
        return [(w, *v) for w, v in pool.items()]

    # Union-4 recall
    rec = 0
    for w1, w2 in test_pairs:
        if any(
            t in {c[0] for c in pool4(s)} for s, t in [(w1, w2), (w2, w1)]
        ):
            rec += 1
    print(f"\nUnion-4 pool recall: {rec / len(test_pairs):.1%}")

    print("Training union-4 reranker on val set...")
    rows = []
    for w1, w2 in val_pairs:
        for q, t in [(w1, w2), (w2, w1)]:
            for cand, mr, ms, im, ia, ip, ic in pool4(q):
                rows.append((q, cand, mr, ms, im, ia, ip, ic, 1 if cand == t else 0))
    rr = Union4Reranker(space, model).fit(rows)

    names = [
        "mlp_score", "ica_cosine", "mlp_rank", "freq_ratio", "interaction",
        "inv_rank", "glove_cos", "morph_sim", "max_zdiff",
        "in_mlp", "in_axis", "in_proc", "in_ica_map",
    ]
    print("  Feature weights:")
    for nm, c in sorted(zip(names, rr.clf.coef_[0], strict=True), key=lambda x: -abs(x[1])):
        print(f"    {nm:12s} {c:+.3f}")

    def fn(src):
        return rr.rerank(src, pool4(src), top_n=10)

    hits, _ = compute_hits(test_pairs, fn)
    print(f"\nUnion-4 reranker: @1={hits[1]:.1%} @5={hits[5]:.1%} @10={hits[10]:.1%}")

    out = {
        "union4_recall": rec / len(test_pairs),
        "hits": {str(k): v for k, v in hits.items()},
        "feature_weights": dict(zip(names, rr.clf.coef_[0].tolist(), strict=True)),
        "elapsed_s": time.time() - t0,
    }
    with open("results/exp_union4.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results/exp_union4.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
