"""Cross-dictionary antonym discovery: train on WordNet, test on ConceptNet.

Protocol (leak-free):
  - TRAIN pairs: WordNet pairs, both words in GloVe-100 (2353 pairs).
  - TEST queries: ConceptNet-only pairs (in neither WordNet direction), both
    words in the retrieval pool (top-50k valid-English).
  - Methods share the candidate pool. Procrustes W is fit once on WordNet.
    The ICA+MLP path reuses the legacy pipeline space (results/ica_space),
    trained on WordNet pairs — the legacy space was counter-fit on WordNet
    pairs too, which is train-side information only here, so no leak.
  - Metrics on directed queries (same as translation eval) AND pair-level
    best-of-2 protocol. Detailed rank dump for showcase mining.

Also evaluates WordNet-train/WordNet-test 5-fold Procrustes as reference.

Usage:
    pixi run python scripts/eval_discovery.py
"""
import csv
import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.antonym_classifier import AntonymClassifier
from src.antonym_loader import extract_antonym_pairs
from src.ica_transformer import is_valid_english_word, load_ica_space
from src.word2vec_loader import Word2VecLoader


def unit_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-10)


def load_conceptnet(path="data/conceptnet_antonyms.tsv"):
    pairs = set()
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            a, b = row[0].strip(), row[1].strip()
            if a and b and a != b:
                pairs.add(tuple(sorted((a, b))))
    return pairs


def directed_metrics(preds_u, cand_u, cand_words, queries, top_n=10):
    """queries: list of (src, tgt). Returns hits dict + per-query ranks."""
    sims = preds_u @ cand_u.T
    order = np.argsort(sims, axis=1)[:, ::-1][:, : top_n + 1]
    hits = {1: 0, 5: 0, 10: 0}
    ranks = []
    for qi, (q, tgt) in enumerate(queries):
        rank = 0
        found = None
        for ci in order[qi]:
            w = cand_words[ci]
            if w == q:
                continue
            rank += 1
            if rank > top_n:
                break
            if w == tgt:
                found = rank
                for k in (1, 5, 10):
                    if rank <= k:
                        hits[k] += 1
                break
        ranks.append(found)  # None = miss
    n = len(queries)
    return {k: hits[k] / n for k in (1, 5, 10)}, ranks


def main():
    t0 = time.time()
    model = Word2VecLoader().load_glove(100)

    wn = set()
    for a, b in extract_antonym_pairs():
        if a in model and b in model:
            wn.add(tuple(sorted((a, b))))
    cn_all = {p for p in load_conceptnet() if p[0] in model and p[1] in model}
    cn_only = [p for p in cn_all if p not in wn]
    print(f"WordNet: {len(wn)}, ConceptNet: {len(cn_all)}, CN-only: {len(cn_only)}")

    pool = [w for w in model.index_to_key[:50000] if is_valid_english_word(w)]
    U = unit_rows(np.array([model[w] for w in pool], dtype=np.float64))
    w2i = {w: i for i, w in enumerate(pool)}

    # --- Procrustes trained on ALL WordNet pairs (in-pool) ---
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
    print(f"Procrustes fit on {len(Xtr) // 2} WordNet pairs")

    # --- Test queries: CN-only pairs, both words in pool, common-word focus ---
    test_pairs = [p for p in cn_only if p[0] in w2i and p[1] in w2i]
    queries = []
    for w1, w2 in test_pairs:
        queries += [(w1, w2), (w2, w1)]
    Q = np.array([model[q].astype(np.float64) for q, _ in queries])
    preds = unit_rows(Q @ W)
    hits, ranks = directed_metrics(preds, U, pool, queries)

    # negatives control: random pool words as pseudo-targets (chance rate)
    rng = np.random.RandomState(0)
    neg_queries = [(q, pool[rng.randint(len(pool))]) for q, _ in queries]
    neg_hits, _ = directed_metrics(preds, U, pool, neg_queries)

    print(f"\nCN-only directed queries: {len(queries)} "
          f"({len(test_pairs)} pairs)")
    for k in (1, 5, 10):
        print(f"  procrustes @{k}: {hits[k]:.4f}   (random-target control: "
              f"{neg_hits[k]:.4f})")

    # --- Legacy ICA+MLP on same queries (reference; train-side only info) ---
    try:
        space = load_ica_space("results/ica_space")
        mlp = AntonymClassifier(space, "mlp")
        mlp.fit([p for p in wn
                 if p[0] in space.word_to_idx and p[1] in space.word_to_idx],
                rng=np.random.RandomState(42))
        mlp_rows, mlp_q = [], []
        for q, tgt in queries:
            if q not in space.word_to_idx:
                continue
            mlp_rows.append(q)
            mlp_q.append((q, tgt))
        # score via retrieve() top-10 per query (slow but one-off)
        mlp_hits = {1: 0, 5: 0, 10: 0}
        for q, tgt in mlp_q:
            for j, (w, _) in enumerate(mlp.retrieve(q, top_n=10)):
                if w == tgt:
                    for k in (1, 5, 10):
                        if j + 1 <= k:
                            mlp_hits[k] += 1
                    break
        n = len(mlp_q)
        print("  mlp_ica      : " + "  ".join(
            f"@{k}: {mlp_hits[k] / n:.4f}" for k in (1, 5, 10)))
    except Exception as e:
        print(f"  mlp_ica skipped: {e}")

    # --- Rank dump for showcase mining: hits (rank<=10) + near-miss sample ---
    dump = []
    for (q, tgt), r in zip(queries, ranks):
        if r is not None and r <= 10:
            dump.append({"query": q, "target": tgt, "rank": r})
    dump.sort(key=lambda d: (d["rank"], d["query"]))
    print(f"\nCN-only hits@10: {len(dump)}/{len(queries)}")
    for d in dump[:40]:
        print(f"  @{d['rank']:2d}  {d['query']:18s} -> {d['target']}")

    out = {
        "n_wordnet": len(wn),
        "n_conceptnet": len(cn_all),
        "n_cn_only": len(cn_only),
        "n_test_pairs": len(test_pairs),
        "n_queries": len(queries),
        "procrustes": {str(k): v for k, v in hits.items()},
        "random_control": {str(k): v for k, v in neg_hits.items()},
        "hits_dump": dump,
        "elapsed_s": time.time() - t0,
    }
    with open("results/discovery_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/discovery_eval.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
