"""Novel antonym candidate mining with the Procrustes map.

Trains Procrustes W on WordNet pairs (raw GloVe-100), predicts antonyms for
common query words, and keeps queries whose top-1 prediction appears in
NEITHER WordNet nor ConceptNet. Ranks candidates by prediction confidence
(cosine of predicted vector to top-1 candidate + margin over top-2).

Usage:
    pixi run python scripts/eval_novel.py [--n-query 10000 --top 80]
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


def unit_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-10)


NEG_PREFIXES = ("un", "in", "im", "il", "ir", "dis", "non", "anti", "de",
                "mis", "over", "under", "counter", "a")


def morph_variant(q, c):
    for p in NEG_PREFIXES:
        if c == p + q or q == p + c:
            return p
    if c.endswith("less") and c[:-4] == q:
        return "-less"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-query", type=int, default=10000)
    ap.add_argument("--top", type=int, default=80)
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
    known_pair = wn | cn
    known_dir = set(known_pair) | {(b, a) for a, b in known_pair}
    print(f"known pairs: WN={len(wn)} CN={len(cn)} union={len(known_pair)}")

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
               if rank[w] < args.n_query and w not in ENGLISH_STOP_WORDS
               and len(w) >= 3]
    print(f"queries: {len(queries)}")
    Q = np.array([model[q].astype(np.float64) for q in queries])
    P = unit_rows(Q @ W)
    S = P @ U.T  # (nq, npool)

    cands = []
    known_top1 = 0
    for qi, q in enumerate(queries):
        order = np.argsort(S[qi])[::-1]
        top = []
        for ci in order:
            w = pool[ci]
            if w == q:
                continue
            top.append((w, float(S[qi][ci])))
            if len(top) == 3:
                break
        (c1, s1), (_, s2), _ = top
        if (q, c1) in known_dir:
            known_top1 += 1
            continue
        if c1 in ENGLISH_STOP_WORDS or rank.get(c1, 10**9) >= 20000:
            continue
        cands.append({
            "query": q, "q_rank": rank[q],
            "top1": c1, "cos1": round(s1, 4),
            "margin": round(s1 - s2, 4),
            "morph": morph_variant(q, c1),
            "top3": [{"w": w, "cos": round(s, 4)} for w, s in top],
        })
    print(f"queries with known top-1: {known_top1}/{len(queries)} "
          f"({known_top1 / len(queries):.1%})")
    print(f"novel top-1 candidates: {len(cands)}")

    cands.sort(key=lambda d: (-d["cos1"], -d["margin"]))
    show = cands[:args.top]
    for d in show:
        print(f"  {d['query']:16s} -> {d['top1']:16s} "
              f"cos={d['cos1']:.3f} m={d['margin']:.3f} {d['morph']}")

    with open("results/novel_candidates.json", "w") as f:
        json.dump({"candidates": show, "n_novel": len(cands),
                   "elapsed_s": time.time() - t0}, f, indent=2)
    print(f"\nSaved results/novel_candidates.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
