"""
Experiment: learned linear map in ICA score space.

Procrustes on raw GloVe already contributes candidates (in_proc is the
#2 reranker signal). This tests the same idea on ICA scores:

  W_ica = argmin ||s_src·W - s_tgt||²   (least squares on train pairs)

Retrieval: nearest neighbours of s(src)·W_ica in ICA score space.
Because W_ica acts on interpretable axes, its diagonal/weights show
which axes the map amplifies or flips — a learned analogue of manual
axis inversion.

Also reports union-pool recall gain if added as a 4th source.

Usage:
    pixi run python scripts/exp_ica_translation.py
"""

import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from src.antonym_loader import extract_antonym_pairs
from src.eval_utils import compute_hits
from src.ica_transformer import load_ica_space


def main():
    t0 = time.time()
    print("Loading ICA space...")
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
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_pairs = [pairs[i] for i in idx[:n_train]]
    test_pairs = [pairs[i] for i in idx[n_train + n_val :]]
    print(f"Split: {len(train_pairs)} train / {len(test_pairs)} test")

    # --- Fit W_ica on train pairs (both directions)
    src_v, tgt_v = [], []
    for a, b in train_pairs:
        src_v.append(space.score(a))
        tgt_v.append(space.score(b))
        src_v.append(space.score(b))
        tgt_v.append(space.score(a))
    X = np.array(src_v, dtype=np.float64)
    Y = np.array(tgt_v, dtype=np.float64)
    W, *_ = np.linalg.lstsq(X.T @ X, X.T @ Y, rcond=None)
    print(f"W_ica: {W.shape}, ||W||_F={np.linalg.norm(W):.3f}")

    # --- Retrieve in ICA score space
    S = space.S.astype(np.float64)
    S_norm = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-10)

    def ica_map_cands(query, top_n=20):
        q = space.score(query)
        if q is None:
            return []
        pred = q @ W
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

    # Hits@k standalone
    hits, _ = compute_hits(test_pairs, lambda s: ica_map_cands(s, top_n=10))
    print(f"\nICA-map retrieval: @1={hits[1]:.1%} @5={hits[5]:.1%} @10={hits[10]:.1%}")

    # Recall contribution vs other sources (reuse exp_union_pool numbers)
    rec = 0
    for w1, w2 in test_pairs:
        if any(
            tgt in ica_map_cands(src, top_n=20)
            for src, tgt in [(w1, w2), (w2, w1)]
        ):
            rec += 1
    print(f"ICA-map top-20 recall (pair-level): {rec / len(test_pairs):.1%}")

    # --- Interpretability: which axes does W_ica emphasise?
    diag = np.diag(W)
    top_axes = np.argsort(np.abs(diag))[::-1][:10]
    print("\nTop |diag(W)| axes (self-mapping strength):")
    for k in top_axes:
        print(f"  axis {k:3d}: W[k,k]={diag[k]:+.3f}")

    # Off-diagonal: which axis pairs interact most
    off = W.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(np.argsort(np.abs(off).ravel())[::-1][:10], off.shape)
    print("\nTop off-diagonal interactions (axis i -> axis j):")
    for a, b in zip(i, j, strict=True):
        print(f"  {a:3d} -> {b:3d}: {off[a, b]:+.3f}")

    out = {
        "hits": {str(k): v for k, v in hits.items()},
        "recall_top20": rec / len(test_pairs),
        "diag_top": [
            {"axis": int(k), "w": float(diag[k])} for k in top_axes
        ],
        "offdiag_top": [
            {"from": int(a), "to": int(b), "w": float(off[a, b])}
            for a, b in zip(i, j, strict=True)
        ],
        "elapsed_s": time.time() - t0,
    }
    with open("results/exp_ica_translation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/exp_ica_translation.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
