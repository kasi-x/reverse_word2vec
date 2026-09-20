"""Control: is Procrustes just nearest-neighbour in disguise?

On CN-only directed queries (same set/protocol as eval_discovery.py):
  1. NN baseline: rank by cosine to the QUERY vector itself (raw GloVe).
  2. Overlap: P(procrustes top-1 == NN top-1); hits shared vs unique.
  3. Asymmetry check: fraction of queries where procrustes(src->tgt)
     differs from NN — the directed value-add.

Usage:
    pixi run python scripts/eval_nn_control.py
"""
import csv
import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from src.antonym_loader import extract_antonym_pairs
from src.ica_transformer import is_valid_english_word
from src.word2vec_loader import Word2VecLoader
from scripts.eval_translation import unit_rows


def ranks_for(sims_row, cand_words, query, target, top_n=10):
    order = np.argsort(sims_row)[::-1][: top_n + 1]
    rank = 0
    for ci in order:
        w = cand_words[ci]
        if w == query:
            continue
        rank += 1
        if rank > top_n:
            break
        if w == target:
            return rank
    return None


def main():
    t0 = time.time()
    model = Word2VecLoader().load_glove(100)
    wn = set()
    for a, b in extract_antonym_pairs():
        if a in model and b in model:
            wn.add(tuple(sorted((a, b))))

    pool = [w for w in model.index_to_key[:50000] if is_valid_english_word(w)]
    U = unit_rows(np.array([model[w] for w in pool], dtype=np.float64))
    w2i = {w: i for i, w in enumerate(pool)}

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

    cn_only = []
    with open("data/conceptnet_antonyms.tsv") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            a, b = row[0].strip(), row[1].strip()
            if a and b and tuple(sorted((a, b))) not in wn \
                    and a in w2i and b in w2i:
                cn_only.append((a, b))
    queries = []
    for w1, w2 in cn_only:
        queries += [(w1, w2), (w2, w1)]
    print(f"queries: {len(queries)}")

    Q = np.array([model[q].astype(np.float64) for q, _ in queries])
    Qu = unit_rows(Q)
    P = unit_rows(Q @ W)
    S_nn = Qu @ U.T
    S_pr = P @ U.T

    nn_hits = {1: 0, 5: 0, 10: 0}
    pr_hits = {1: 0, 5: 0, 10: 0}
    both = {1: 0, 5: 0, 10: 0}
    pr_only = 0   # pr hit@10 where nn misses@10
    nn_only = 0
    same_top1 = 0
    pr_top1_nn_rank = []  # where does pr-top1 sit in NN ranking?
    for qi, (q, tgt) in enumerate(queries):
        r_nn = ranks_for(S_nn[qi], pool, q, tgt)
        r_pr = ranks_for(S_pr[qi], pool, q, tgt)
        for k in (1, 5, 10):
            if r_nn is not None and r_nn <= k:
                nn_hits[k] += 1
            if r_pr is not None and r_pr <= k:
                pr_hits[k] += 1
            if r_nn is not None and r_nn <= k and r_pr is not None and r_pr <= k:
                both[k] += 1
        if r_pr is not None and r_pr <= 10 and (r_nn is None or r_nn > 10):
            pr_only += 1
        if r_nn is not None and r_nn <= 10 and (r_pr is None or r_pr > 10):
            nn_only += 1
        # top-1 agreement
        top_pr = top1_word(S_pr[qi], pool, q)
        top_nn = top1_word(S_nn[qi], pool, q)
        if top_pr == top_nn:
            same_top1 += 1

    n = len(queries)
    print(f"\nNN         @1={nn_hits[1]/n:.4f} @5={nn_hits[5]/n:.4f} @10={nn_hits[10]/n:.4f}")
    print(f"Procrustes @1={pr_hits[1]/n:.4f} @5={pr_hits[5]/n:.4f} @10={pr_hits[10]/n:.4f}")
    print(f"both@{ {k: both[k]/n for k in (1,5,10)} }")
    print(f"pr-only@10: {pr_only} ({pr_only/n:.3f}), nn-only@10: {nn_only} ({nn_only/n:.3f})")
    print(f"same top-1: {same_top1}/{n} = {same_top1/n:.1%}")

    with open("results/nn_control.json", "w") as f:
        json.dump({
            "nn": {str(k): nn_hits[k]/n for k in (1, 5, 10)},
            "procrustes": {str(k): pr_hits[k]/n for k in (1, 5, 10)},
            "both": {str(k): both[k]/n for k in (1, 5, 10)},
            "pr_only_10": pr_only / n,
            "nn_only_10": nn_only / n,
            "same_top1": same_top1 / n,
            "elapsed_s": time.time() - t0,
        }, f, indent=2)
    print(f"\nSaved results/nn_control.json ({time.time() - t0:.0f}s)")


def top1_word(sims_row, cand_words, query):
    for ci in np.argsort(sims_row)[::-1]:
        if cand_words[ci] != query:
            return cand_words[ci]
    return None


if __name__ == "__main__":
    main()
