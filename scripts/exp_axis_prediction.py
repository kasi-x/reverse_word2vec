"""
Experiment: can we predict WHICH axis to invert without seeing the target?

Blind axis inversion currently flips the source word's highest-|z| axis
(qualitative "auto" mode: 6.7% Hits@1 on 15 canonical cases). The oracle
axis — argmax_k |s1[k]-s2[k]|/std_k — needs the target. This script tests
a deployable middle ground:

  knn-axis: for query q, find its nearest neighbours among TRAIN source
    words in ICA space; predict the oracle axis by similarity-weighted
    vote over their oracle axes.

Baselines:
  auto:   source top-|z| axis (current blind mode)
  oracle: true argmax axis (upper bound, not deployable)

5-fold CV over WordNet pairs; Hits@{1,5,10} of axis-inversion retrieval.

Usage:
    pixi run python scripts/exp_axis_prediction.py
"""

import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")

import numpy as np

from src.antonym_loader import extract_antonym_pairs
from src.eval_utils import make_folds, train_test_pairs
from src.ica_transformer import load_ica_space
from src.semantic_operations import SemanticOperator
from src.word2vec_loader import Word2VecLoader

K_NEIGHBORS = [1, 3, 5, 10]


def oracle_axis(space, axis_std, w1, w2) -> int:
    s1, s2 = space.score(w1), space.score(w2)
    return int(np.argmax(np.abs(s1 - s2) / axis_std))


def main():
    t0 = time.time()
    print("Loading model and ICA space...")
    model = Word2VecLoader().load_glove(100)
    space = load_ica_space("results/ica_space")
    operator = SemanticOperator(model, space)

    axis_std = np.maximum(np.std(space.S, axis=0), 1e-8)

    pairs = [
        (a, b)
        for a, b in extract_antonym_pairs()
        if space.score(a) is not None and space.score(b) is not None
    ]
    print(f"Valid pairs: {len(pairs)}")

    # Unit-normalised ICA scores for cosine NN
    S = space.S.astype(np.float64)
    S_norm = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-10)
    w2i = space.word_to_idx

    folds = make_folds(len(pairs), 5, seed=42)
    results = defaultdict(list)

    for fold in range(5):
        train_pairs, test_pairs = train_test_pairs(pairs, fold, folds)

        # Train: oracle axis for every train pair + every direction
        src_words, src_axes = [], []
        for a, b in train_pairs:
            for src, tgt in [(a, b), (b, a)]:
                src_words.append(src)
                src_axes.append(oracle_axis(space, axis_std, src, tgt))
        src_idx = np.array([w2i[w] for w in src_words])
        src_vecs = S_norm[src_idx]  # (n_train_src, 100)

        hits = {
            m: {1: 0, 5: 0, 10: 0} for m in ["auto", "oracle"] + [f"knn{k}" for k in K_NEIGHBORS]
        }
        n_eval = 0

        for a, b in test_pairs:
            n_eval += 1
            for src, tgt in [(a, b), (b, a)]:
                s_src = space.score(src)
                q_idx = w2i[src]
                q_vec = S_norm[q_idx]

                # --- auto baseline: top-|z| axis
                z = np.abs(s_src) / axis_std
                auto_ax = int(np.argmax(z))

                # --- oracle
                or_ax = oracle_axis(space, axis_std, src, tgt)

                # --- knn vote
                sims = src_vecs @ q_vec
                nn = np.argsort(sims)[::-1][: max(K_NEIGHBORS)]

                pred_axes = {"auto": auto_ax, "oracle": or_ax}
                for k in K_NEIGHBORS:
                    votes = defaultdict(float)
                    for j in nn[:k]:
                        votes[src_axes[j]] += float(sims[j])
                    pred_axes[f"knn{k}"] = max(votes.items(), key=lambda kv: kv[1])[0]

                for m, ax in pred_axes.items():
                    nbrs = operator.axis_invert(src, ax, top_n=10)
                    rank = next((i + 1 for i, nb in enumerate(nbrs) if nb.word == tgt), None)
                    if rank is not None:
                        for kk in (1, 5, 10):
                            if rank <= kk:
                                hits[m][kk] += 1

        for m in hits:
            for kk in (1, 5, 10):
                results[m].append(hits[m][kk] / (n_eval * 2))
        print(
            f"Fold {fold + 1}: " + "  ".join(f"{m}@1={hits[m][1] / (n_eval * 2):.3f}" for m in hits)
        )

    summary = {}
    for m, vals in results.items():
        # vals: list of 5 folds × but stored per-k flattened — regroup
        arr = np.array(vals).reshape(5, 3)
        summary[m] = {
            f"hits_at_{k}": {"mean": float(arr[:, i].mean()), "std": float(arr[:, i].std())}
            for i, k in enumerate([1, 5, 10])
        }

    print("\n=== Summary (5-fold CV, both directions) ===")
    for m in ["auto"] + [f"knn{k}" for k in K_NEIGHBORS] + ["oracle"]:
        s = summary[m]
        print(
            f"{m:8s}  @1={s['hits_at_1']['mean']:.3f}±{s['hits_at_1']['std']:.3f}"
            f"  @5={s['hits_at_5']['mean']:.3f}  @10={s['hits_at_10']['mean']:.3f}"
        )

    with open("results/exp_axis_prediction.json", "w") as f:
        json.dump({"summary": summary, "elapsed_s": time.time() - t0}, f, indent=2)
    print(f"\nSaved results/exp_axis_prediction.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
