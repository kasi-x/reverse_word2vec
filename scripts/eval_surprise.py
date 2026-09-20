"""Surprise-scored novel antonym mining.

Procrustes W on WordNet pairs (raw GloVe-100). For each common query word,
score its predicted top-1 candidate c by SURPRISE: how much MORE similar c
is to the prediction W·v(q) than to the query v(q) itself —

    surprise = cos(W·v(q), c) − cos(v(q), c)

High surprise = the map points somewhere the neighbourhood alone would not.
Filters: c not in either dictionary (either direction), not a stopword,
in top-20k ranks, no morph-negation variants, prediction confidence ≥ 0.45.

Also reports CN-only recall as a function of surprise threshold, to show
that surprise selects the directed (non-NN) hits.

Usage:
    pixi run python scripts/eval_surprise.py [--top 100]
"""
import argparse
import csv
import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.antonym_loader import extract_antonym_pairs
from src.ica_transformer import is_valid_english_word
from src.word2vec_loader import Word2VecLoader
from scripts.eval_translation import unit_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()
    t0 = time.time()

    model = Word2VecLoader().load_glove(100)
    wn = set()
    for a, b in extract_antonym_pairs():
        if a in model and b in model:
            wn.add(tuple(sorted((a, b))))
    cn = set()
    with open("data/conceptnet_antonyms.tsv") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2 and row[0].strip() != row[1].strip():
                a, b = row[0].strip(), row[1].strip()
                if a in model and b in model:
                    cn.add(tuple(sorted((a, b))))
    known_dir = (wn | cn) | {(b, a) for a, b in (wn | cn)}
    print(f"known union: {len(wn | cn)}")

    pool = [w for w in model.index_to_key[:50000] if is_valid_english_word(w)]
    U = unit_rows(np.array([model[w] for w in pool], dtype=np.float64))
    w2i = {w: i for i, w in enumerate(pool)}
    rank = model.key_to_index

    Xtr, Ytr = [], []
    for w1, w2 in wn:
        if w1 not in w2i or w2 not in w2i:
            continue
        v1, v2 = model[w1].astype(np.float64), model[w2].astype(np.float64)
        Xtr += [v1, v2]
        Ytr += [v2, v1]
    Xtr, Ytr = np.array(Xtr), np.array(Ytr)
    Uu, _, Vt = np.linalg.svd(Xtr.T @ Ytr)
    W = Uu @ Vt

    queries = [w for w in pool
               if rank[w] < 10000 and w not in ENGLISH_STOP_WORDS and len(w) >= 3]
    Q = np.array([model[q].astype(np.float64) for q in queries])
    Qu = unit_rows(Q)
    P = unit_rows(Q @ W)

    # batched: top-3 by prediction + surprise of argmax
    S_pred = P @ U.T
    S_self = Qu @ U.T
    order = np.argsort(S_pred, axis=1)[:, ::-1][:, :4]
    cands = []
    for qi, q in enumerate(queries):
        top = []
        for ci in order[qi]:
            w = pool[ci]
            if w == q:
                continue
            top.append((w, float(S_pred[qi][ci]), float(S_self[qi][ci])))
            if len(top) == 3:
                break
        (c1, sp1, ss1), (c2, sp2, _), _ = top
        if (q, c1) in known_dir:
            continue
        if c1 in ENGLISH_STOP_WORDS or rank.get(c1, 10**9) >= 20000:
            continue
        if sp1 < 0.45:
            continue
        mt = morph(q, c1)
        if mt:
            continue
        cands.append({
            "query": q, "top1": c1,
            "pred_cos": round(sp1, 4), "self_cos": round(ss1, 4),
            "surprise": round(sp1 - ss1, 4),
            "margin": round(sp1 - sp2, 4),
            "top3": [w for w, _, _ in top],
        })
    cands.sort(key=lambda d: (-d["surprise"], -d["pred_cos"]))
    print(f"novel candidates (conf>=0.45, no morph): {len(cands)}")
    for d in cands[:args.top]:
        print(f"  {d['query']:16s} -> {d['top1']:16s} "
              f"surp={d['surprise']:+.3f} pred={d['pred_cos']:.3f} "
              f"self={d['self_cos']:.3f} m={d['margin']:.3f}  [{', '.join(d['top3'][1:])}]")

    # surprise vs CN-only recall curve (directed queries)
    cn_only = [(a, b) for a, b in cn
               if tuple(sorted((a, b))) not in wn and a in w2i and b in w2i]
    qs = []
    for w1, w2 in cn_only:
        qs += [(w1, w2), (w2, w1)]
    Qq = np.array([model[q].astype(np.float64) for q, _ in qs])
    Qqu = unit_rows(Qq)
    Pp = unit_rows(Qq @ W)
    Sp, Ss = Pp @ U.T, Qqu @ U.T
    op = np.argsort(Sp, axis=1)[:, ::-1][:, :11]
    print("\nsurprise threshold -> CN-only hit@10 (directed):")
    for thr in (-1.0, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3):
        hit = tot = 0
        for qi, (q, tgt) in enumerate(qs):
            top = [pool[ci] for ci in op[qi] if pool[ci] != q][:10]
            if not top:
                continue
            surp = float(Sp[qi][w2i[top[0]]] - Ss[qi][w2i[top[0]]])
            if surp < thr:
                continue
            tot += 1
            if tgt in top:
                hit += 1
        print(f"  thr={thr:+.1f}: {hit}/{tot} = {hit / tot:.4f}" if tot else
              f"  thr={thr:+.1f}: n/a")

    with open("results/surprise_candidates.json", "w") as f:
        json.dump({"candidates": cands[:args.top], "n_novel": len(cands),
                   "elapsed_s": time.time() - t0}, f, indent=2)
    print(f"\nSaved results/surprise_candidates.json ({time.time() - t0:.0f}s)")


def morph(q, c):
    for p in ("un", "in", "im", "il", "ir", "dis", "non", "anti", "de",
              "mis", "over", "under", "counter", "a"):
        if c == p + q or q == p + c:
            return p
    if c.endswith("less") and c[:-4] == q:
        return "-less"
    return ""


if __name__ == "__main__":
    main()
