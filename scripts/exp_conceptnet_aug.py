"""
Experiment: augment training data with ConceptNet antonym pairs.

WordNet gives 1121 usable pairs; ConceptNet adds ~11k more (noisier).
Question: does training the MLP + reranker on WordNet ∪ ConceptNet
beat WordNet-only on the same WordNet test split?

Variants:
  wn_only      : train on WordNet train split (current pipeline)
  wn_plus_cn   : train on WordNet train + ConceptNet pairs (in-vocab)
  cn_only      : train on ConceptNet only (ablation)

Evaluation is always on the WordNet 70/15/15 holdout (seed 42) so
numbers compare directly with reranker_eval.json.

Usage:
    pixi run python scripts/exp_conceptnet_aug.py
"""

import csv
import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from src.antonym_classifier import AntonymClassifier
from src.antonym_loader import extract_antonym_pairs
from src.eval_utils import compute_hits
from src.ica_transformer import load_ica_space
from src.reranker import CandidateSources, Reranker
from src.word2vec_loader import Word2VecLoader

POOL_N = 100
AUX_N = 20


def load_cn_pairs(space) -> list[tuple[str, str]]:
    pairs = []
    with open("data/conceptnet_antonyms.tsv") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                a, b = row[0].strip(), row[1].strip()
                if a != b and space.score(a) is not None and space.score(b) is not None:
                    pairs.append((a, b))
    # dedupe (order-preserving)
    seen = set()
    out = []
    for p in pairs:
        key = tuple(sorted(p))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def main():
    t0 = time.time()
    print("Loading model and ICA space...")
    model = Word2VecLoader().load_glove(100)
    space = load_ica_space("results/ica_space")

    wn_pairs = [
        (a, b)
        for a, b in extract_antonym_pairs()
        if space.score(a) is not None and space.score(b) is not None
    ]
    cn_pairs = load_cn_pairs(space)
    print(f"WordNet pairs: {len(wn_pairs)}, ConceptNet pairs (in-vocab): {len(cn_pairs)}")

    rng = np.random.RandomState(42)
    idx = np.arange(len(wn_pairs))
    rng.shuffle(idx)
    n = len(wn_pairs)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    wn_train = [wn_pairs[i] for i in idx[:n_train]]
    wn_val = [wn_pairs[i] for i in idx[n_train : n_train + n_val]]
    wn_test = [wn_pairs[i] for i in idx[n_train + n_val :]]
    print(f"WordNet split: {len(wn_train)}/{len(wn_val)}/{len(wn_test)}")

    # Exclude CN pairs that overlap the WN test set (leak guard)
    test_set = {tuple(sorted(p)) for p in wn_test}
    cn_clean = [p for p in cn_pairs if tuple(sorted(p)) not in test_set]
    print(f"ConceptNet after test-overlap removal: {len(cn_clean)}")

    variants = {
        "wn_only": (wn_train, wn_val),
        "wn_plus_cn": (wn_train + cn_clean, wn_val + cn_clean[:500]),
        "cn_only": (cn_clean, cn_clean[:500]),
    }

    results = {}
    for name, (train_p, val_p) in variants.items():
        print(f"\n=== {name}: train={len(train_p)} val={len(val_p)} ===")
        mlp = AntonymClassifier(space, "mlp")
        mlp.fit(train_p, neg_ratio=3.0, rng=np.random.RandomState(42))

        sources = CandidateSources(space, model)
        sources.fit(train_p, mlp, mlp_top_n=POOL_N, aux_top_n=AUX_N)

        rr = Reranker(space, model)
        rr.fit(val_p, mlp, top_n=POOL_N, sources=sources)

        mlp_hits, _ = compute_hits(
            wn_test, lambda s, _m=mlp: [w for w, _ in _m.retrieve(s, top_n=10)]
        )
        rr_hits, _ = compute_hits(
            wn_test,
            lambda s, _r=rr, _src=sources: [
                w for w, _ in _r.rerank(s, _src.pool(s), top_n=10)
            ],
        )
        results[name] = {
            "mlp": {str(k): v for k, v in mlp_hits.items()},
            "reranker": {str(k): v for k, v in rr_hits.items()},
        }
        print(
            f"  MLP      @1={mlp_hits[1]:.1%} @5={mlp_hits[5]:.1%} @10={mlp_hits[10]:.1%}"
        )
        print(
            f"  Reranker @1={rr_hits[1]:.1%} @5={rr_hits[5]:.1%} @10={rr_hits[10]:.1%}"
        )

    out = {"results": results, "elapsed_s": time.time() - t0}
    with open("results/exp_conceptnet_aug.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/exp_conceptnet_aug.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
