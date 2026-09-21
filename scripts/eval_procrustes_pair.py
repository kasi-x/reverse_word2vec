"""Pair-level (best-of-2-directions) scoring for the Procrustes antonym map.

Same folds/pool as eval_translation.py (seed 42), but scores each test PAIR by
the better rank of its two directed queries — the `compute_hits` protocol used
by eval_leakfree.py and the legacy scripts. Enables apples-to-apples comparison
with oracle_1ax (0.245 @1) and linear_raw (0.218 @1).

Usage:
    pixi run python scripts/eval_procrustes_pair.py
"""

import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from src.antonym_loader import extract_antonym_pairs
from src.eval_utils import unit_rows
from src.ica_transformer import is_valid_english_word
from src.word2vec_loader import Word2VecLoader


def main():
    t0 = time.time()
    model = Word2VecLoader().load_glove(100)
    all_pairs = [(a, b) for a, b in extract_antonym_pairs() if a in model and b in model]
    pool = [w for w in model.index_to_key[:50000] if is_valid_english_word(w)]
    U = unit_rows(np.array([model[w] for w in pool], dtype=np.float64))
    w2i = {w: i for i, w in enumerate(pool)}
    print(f"pairs: {len(all_pairs)}, pool: {len(pool)}")

    rng = np.random.RandomState(42)
    idx = np.arange(len(all_pairs))
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)

    fold_metrics = []
    for fi in range(5):
        test_idx = set(folds[fi].tolist())
        train = [all_pairs[i] for i in range(len(all_pairs)) if i not in test_idx]
        test = [
            all_pairs[i] for i in folds[fi] if all_pairs[i][0] in w2i and all_pairs[i][1] in w2i
        ]

        Xtr, Ytr = [], []
        for w1, w2 in train:
            if w1 not in w2i or w2 not in w2i:
                continue
            v1, v2 = model[w1].astype(np.float64), model[w2].astype(np.float64)
            Xtr += [v1, v2]
            Ytr += [v2, v1]
        Xtr, Ytr = np.array(Xtr), np.array(Ytr)
        Uu, _, Vt = np.linalg.svd(Xtr.T @ Ytr)
        W = Uu @ Vt

        hits = {1: 0, 5: 0, 10: 0}
        for w1, w2 in test:
            best = None
            for src, tgt in ((w1, w2), (w2, w1)):
                v = model[src].astype(np.float64)
                pred = v @ W
                pred /= max(np.linalg.norm(pred), 1e-10)
                sims = U @ pred
                order = np.argsort(sims)[::-1][:11]
                rank = 0
                for ci in order:
                    w = pool[ci]
                    if w == src:
                        continue
                    rank += 1
                    if rank > 10:
                        break
                    if w == tgt:
                        if best is None or rank < best:
                            best = rank
                        break
            if best is not None:
                for k in (1, 5, 10):
                    if best <= k:
                        hits[k] += 1
        m = {k: hits[k] / len(test) for k in (1, 5, 10)}
        fold_metrics.append(m)
        print(f"Fold {fi + 1}/5 ({len(test)} test): @1={m[1]:.3f} @5={m[5]:.3f} @10={m[10]:.3f}")

    summary = {
        str(k): {
            "mean": float(np.mean([m[k] for m in fold_metrics])),
            "std": float(np.std([m[k] for m in fold_metrics])),
        }
        for k in (1, 5, 10)
    }
    print(
        f"\nprocrustes/pair-level: @1={summary['1']['mean']:.3f}±{summary['1']['std']:.3f}  "
        f"@5={summary['5']['mean']:.3f}±{summary['5']['std']:.3f}  "
        f"@10={summary['10']['mean']:.3f}±{summary['10']['std']:.3f}"
    )
    with open("results/procrustes_pair.json", "w") as f:
        json.dump({"summary": summary, "elapsed_s": time.time() - t0}, f, indent=2)
    print(f"Saved results/procrustes_pair.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
