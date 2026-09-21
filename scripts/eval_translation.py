"""Translation-baseline refinements for antonym retrieval (raw GloVe, no CF/ICA).

Same 5-fold split as eval_leakfree.py (seed 42 over pairs with both words in
GloVe-100), same candidate pool (top-50k valid-English words). Compares:

  mapping:
    - lstsq      : least-squares W: v(src) -> v(tgt)
    - procrustes : orthogonal W via SVD (rotation-only, cross-lingual style)
    - meandiff   : single translation b = mean(tgt - src), predict v(src) + b
    - negcos     : nearest neighbour to -v(src) (raw space)
  pool:
    - full       : all pool words (~46k)
    - freq20k    : top-20k GloVe ranks only (frequency prior)
    - nostop     : full pool minus sklearn English stop words

Usage:
    pixi run python scripts/eval_translation.py
"""

import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.antonym_loader import extract_antonym_pairs
from src.eval_utils import unit_rows
from src.ica_transformer import is_valid_english_word
from src.word2vec_loader import Word2VecLoader


def batched_hits(
    preds_u: np.ndarray,
    cand_u: np.ndarray,
    cand_words: list[str],
    queries: list[tuple[str, str, str]],
    top_n: int = 10,
) -> dict:
    """preds_u: (n_q, d) unit predictions. queries: (query, target, _) per row."""
    sims = preds_u @ cand_u.T  # (n_q, n_cand)
    order = np.argsort(sims, axis=1)[:, ::-1][:, : top_n + 1]
    hits = {1: 0, 5: 0, 10: 0}
    for qi, (q, tgt, _) in enumerate(queries):
        rank = 0
        for ci in order[qi]:
            w = cand_words[ci]
            if w == q:
                continue
            rank += 1
            if rank > top_n:
                break
            if w == tgt:
                for k in (1, 5, 10):
                    if rank <= k:
                        hits[k] += 1
                break
    n = len(queries)
    return {k: hits[k] / n for k in (1, 5, 10)}


def main():
    t0 = time.time()
    model = Word2VecLoader().load_glove(100)
    all_pairs = [(a, b) for a, b in extract_antonym_pairs() if a in model and b in model]

    pool = [w for w in model.index_to_key[:50000] if is_valid_english_word(w)]
    print(f"pairs: {len(all_pairs)}, pool: {len(pool)}")
    V = np.array([model[w] for w in pool], dtype=np.float64)
    U = unit_rows(V)
    w2i = {w: i for i, w in enumerate(pool)}

    rng = np.random.RandomState(42)
    idx = np.arange(len(all_pairs))
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)

    pools = {
        "full": (pool, U),
        "freq20k": ([w for w in pool if model.key_to_index[w] < 20000], None),  # filled below
        "nostop": ([w for w in pool if w not in ENGLISH_STOP_WORDS], None),
    }
    for key in ("freq20k", "nostop"):
        ws, _ = pools[key]
        pools[key] = (ws, unit_rows(np.array([model[w] for w in ws])))

    methods = ["lstsq", "procrustes", "meandiff", "negcos"]
    results: dict[str, dict[str, list[dict]]] = {m: {p: [] for p in pools} for m in methods}

    for fi in range(5):
        test_idx = set(folds[fi].tolist())
        train = [all_pairs[i] for i in range(len(all_pairs)) if i not in test_idx]
        test = [all_pairs[i] for i in folds[fi]]
        test = [p for p in test if p[0] in w2i and p[1] in w2i]
        print(f"Fold {fi + 1}/5: {len(train)} train / {len(test)} test")

        Xtr, Ytr = [], []
        for w1, w2 in train:
            if w1 not in w2i or w2 not in w2i:
                continue
            v1, v2 = model[w1].astype(np.float64), model[w2].astype(np.float64)
            Xtr += [v1, v2]
            Ytr += [v2, v1]
        Xtr = np.array(Xtr)
        Ytr = np.array(Ytr)

        W_lstsq, _, _, _ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
        M = Xtr.T @ Ytr
        Uu, _, Vt = np.linalg.svd(M)
        W_proc = Uu @ Vt
        b_diff = (Ytr - Xtr).mean(axis=0)

        # directed queries (both directions = separate rows)
        qvecs, qnames = [], []
        for w1, w2 in test:
            for src, tgt in ((w1, w2), (w2, w1)):
                qvecs.append(model[src].astype(np.float64))
                qnames.append((src, tgt, ""))
        Q = np.array(qvecs)

        preds = {
            "lstsq": unit_rows(Q @ W_lstsq),
            "procrustes": unit_rows(Q @ W_proc),
            "meandiff": unit_rows(Q + b_diff),
            "negcos": unit_rows(-Q),
        }
        for m in methods:
            for pname, (ws, UU) in pools.items():
                h = batched_hits(preds[m], UU, ws, qnames)
                results[m][pname].append(h)
                print(f"  {m:10s}/{pname:7s} @1={h[1]:.3f} @5={h[5]:.3f} @10={h[10]:.3f}")

    print(f"\n{'Method':10s}  {'Pool':7s}  {'Hits@1':>13s}  {'Hits@5':>13s}  {'Hits@10':>13s}")
    summary = {}
    for m in methods:
        summary[m] = {}
        for pname in pools:
            ms = results[m][pname]
            row = {
                str(k): {
                    "mean": float(np.mean([x[k] for x in ms])),
                    "std": float(np.std([x[k] for x in ms])),
                }
                for k in (1, 5, 10)
            }
            summary[m][pname] = row
            print(
                f"{m:10s}  {pname:7s}  "
                f"{row['1']['mean']:.3f}±{row['1']['std']:.3f}  "
                f"{row['5']['mean']:.3f}±{row['5']['std']:.3f}  "
                f"{row['10']['mean']:.3f}±{row['10']['std']:.3f}"
            )

    with open("results/translation_comparison.json", "w") as f:
        json.dump({"summary": summary, "elapsed_s": time.time() - t0}, f, indent=2)
    print(f"\nSaved results/translation_comparison.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
