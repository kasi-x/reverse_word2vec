"""
Experiment: non-linear reranker (HistGradientBoosting vs logistic).

The logistic reranker is linear over 13 features. Feature interactions
(e.g. morph_sim matters only when in_mlp=0) may be non-linear. This
script swaps the scorer for HistGradientBoostingClassifier on the same
union pool and compares Hits@k on the holdout.

Usage:
    pixi run python scripts/exp_gbdt_reranker.py
"""

import json
import sys
import time
from difflib import SequenceMatcher

sys.path.insert(0, ".")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.antonym_classifier import AntonymClassifier
from src.antonym_loader import extract_antonym_pairs
from src.eval_utils import compute_hits
from src.ica_transformer import load_ica_space
from src.reranker import CandidateSources
from src.word2vec_loader import Word2VecLoader

POOL_N = 100
AUX_N = 20
N_FEATURES = 13


def build_features(space, model, vocab_rank, oov_rank, S_norm, axis_std, glove_cache, query, cand, mr, ms, im, ia, ip, ic):
    q_idx = space.word_to_idx.get(query)
    c_idx = space.word_to_idx.get(cand)
    if q_idx is None or c_idx is None:
        return np.zeros(N_FEATURES, dtype=np.float32)
    ica_cos = float(S_norm[q_idx] @ S_norm[c_idx])
    q_rank = vocab_rank.get(query, oov_rank) + 1
    c_rank = vocab_rank.get(cand, oov_rank) + 1
    freq_ratio = float(c_rank / q_rank)
    s1, s2 = space.S[q_idx], space.S[c_idx]
    max_zdiff = float(np.max(np.abs(s1 - s2) / axis_std))
    g1 = glove_cache(query)
    g2 = glove_cache(cand)
    glove_cos = float(g1 @ g2) if g1 is not None and g2 is not None else 0.0
    morph = SequenceMatcher(None, query, cand).ratio()
    return np.array(
        [ms, ica_cos, float(mr), freq_ratio, ms * (-ica_cos), 1.0 / mr,
         glove_cos, morph, max_zdiff, float(im), float(ia), float(ip), float(ic)],
        dtype=np.float32,
    )


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

    mlp = AntonymClassifier(space, "mlp")
    mlp.fit(train_pairs, neg_ratio=3.0, rng=np.random.RandomState(42))
    sources = CandidateSources(space, model)
    sources.fit(train_pairs, mlp, mlp_top_n=POOL_N, aux_top_n=AUX_N)

    S = space.S.astype(np.float32)
    S_norm = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-10)
    axis_std = np.maximum(np.std(space.S, axis=0), 1e-8)
    vocab_rank = model.key_to_index
    oov_rank = len(model.key_to_index)
    _cache = {}

    def glove_vec(w):
        if w not in _cache:
            if w not in model:
                _cache[w] = None
            else:
                v = model[w].astype(np.float64)
                nn = np.linalg.norm(v)
                _cache[w] = v / nn if nn > 0 else None
        return _cache[w]

    def feats(q, c, mr, ms, im, ia, ip, ic):
        return build_features(space, model, vocab_rank, oov_rank, S_norm, axis_std, glove_vec, q, c, mr, ms, im, ia, ip, ic)

    print("Building reranker training set from val pool...")
    X, y = [], []
    for w1, w2 in val_pairs:
        for q, t in [(w1, w2), (w2, w1)]:
            for cand, mr, ms, im, ia, ip, ic in sources.pool(q):
                X.append(feats(q, cand, mr, ms, im, ia, ip, ic))
                y.append(1 if cand == t else 0)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"  {len(X)} examples, {y.sum()} positives")

    scorers = {
        "logistic": (LogisticRegression(C=1.0, max_iter=1000, random_state=42), True),
        "gbdt": (HistGradientBoostingClassifier(random_state=42), False),
    }
    results = {}
    for name, (clf, scale) in scorers.items():
        Xtr = StandardScaler().fit_transform(X) if scale else X
        clf.fit(Xtr, y)
        scaler = StandardScaler().fit(X) if scale else None

        def fn(src, _c=clf, _s=scaler):
            pool = sources.pool(src)
            F = np.array([feats(src, *r) for r in pool], dtype=np.float32)
            if _s is not None:
                F = _s.transform(F)
            sc = _c.predict_proba(F)[:, 1]
            order = np.argsort(sc)[::-1][:10]
            return [pool[i][0] for i in order]

        hits, _ = compute_hits(test_pairs, fn)
        results[name] = {str(k): v for k, v in hits.items()}
        print(f"{name:10s} @1={hits[1]:.1%} @5={hits[5]:.1%} @10={hits[10]:.1%}")

    out = {"results": results, "elapsed_s": time.time() - t0}
    with open("results/exp_gbdt_reranker.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results/exp_gbdt_reranker.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
