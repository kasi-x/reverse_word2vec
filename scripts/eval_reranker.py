"""
Reranker evaluation: MLP + learned reranker on antonym retrieval.

Evaluates the full pipeline:
  GloVe → (optional CF) → ICA → MLP classifier → Reranker

Usage:
    pixi run python scripts/eval_reranker.py
"""

import json
import sys

sys.path.insert(0, ".")

import numpy as np

from src.antonym_classifier import AntonymClassifier
from src.antonym_loader import extract_antonym_pairs
from src.axis_labeler import AxisLabeler
from src.eval_utils import compute_hits
from src.ica_transformer import load_ica_space
from src.reranker import CandidateSources, Reranker
from src.word2vec_loader import Word2VecLoader


def main():
    print("Loading model and ICA space...")
    loader = Word2VecLoader()
    model = loader.load_glove(100)
    space = load_ica_space("results/ica_space")
    labeler = AxisLabeler(space)
    labeler.load_profiles("results/axis_profiles.json")  # axis metadata (unused here)
    print(f"  Loaded ICA space: {len(space.words)} words, {space.n_components} components")

    antonym_pairs = extract_antonym_pairs()
    valid_pairs = [
        (w1, w2)
        for w1, w2 in antonym_pairs
        if space.score(w1) is not None and space.score(w2) is not None
    ]
    print(f"  Valid WordNet antonym pairs: {len(valid_pairs)}")

    # Split: 70% train / 15% val (reranker train) / 15% test
    rng = np.random.RandomState(42)
    idx = np.arange(len(valid_pairs))
    rng.shuffle(idx)
    n = len(valid_pairs)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_pairs = [valid_pairs[i] for i in idx[:n_train]]
    val_pairs = [valid_pairs[i] for i in idx[n_train : n_train + n_val]]
    test_pairs = [valid_pairs[i] for i in idx[n_train + n_val :]]
    print(f"  Split: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")

    # ── Train MLP classifier ──────────────────────────────────────────
    print("\nTraining MLP classifier on train set...")
    mlp = AntonymClassifier(space, "mlp")
    mlp.fit(train_pairs, neg_ratio=3.0, rng=rng)
    print("  MLP trained.")

    # ── Candidate sources + reranker ──────────────────────────────────
    # Union pool: MLP top-100 ∪ k-NN predicted-axis top-20 ∪ procrustes
    # top-20. Sources are fit on train pairs only (leak-free).
    POOL_N = 100
    AUX_N = 20
    print(f"\nFitting candidate sources on train set (mlp={POOL_N}, aux={AUX_N})...")
    sources = CandidateSources(space, model)
    sources.fit(train_pairs, mlp, mlp_top_n=POOL_N, aux_top_n=AUX_N)
    print("Training reranker on val set (union pool)...")
    reranker = Reranker(space, model)
    reranker.fit(val_pairs, mlp, top_n=POOL_N, sources=sources)
    print("  Reranker trained.")
    weights = reranker.feature_weights()
    print("  Feature weights:")
    for feat, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat:20s}: {w:+.3f}")

    # ── Evaluate on test set ──────────────────────────────────────────
    TOP_N = 10

    def fn_mlp(src):
        return [w for w, _ in mlp.retrieve(src, top_n=TOP_N)]

    def fn_reranker(src):
        return [w for w, _ in reranker.rerank(src, sources.pool(src), top_n=TOP_N)]

    print(f"\nEvaluating on {len(test_pairs)} test pairs...")
    mlp_hits, _ = compute_hits(test_pairs, fn_mlp)
    rer_hits, _ = compute_hits(test_pairs, fn_reranker)

    print(f"\n{'=' * 50}")
    print(f"{'Method':20s}  {'@1':>7s}  {'@5':>7s}  {'@10':>7s}")
    print(f"{'=' * 50}")
    print(f"{'MLP only':20s}  {mlp_hits[1]:7.1%}  {mlp_hits[5]:7.1%}  {mlp_hits[10]:7.1%}")
    print(f"{'MLP + reranker':20s}  {rer_hits[1]:7.1%}  {rer_hits[5]:7.1%}  {rer_hits[10]:7.1%}")
    print(f"{'=' * 50}")

    # ── Full 5-fold CV of complete pipeline ───────────────────────────
    print("\nRunning 5-fold CV of full pipeline (MLP + reranker)...")
    cv_reranker = Reranker(space, model)
    cv_results = cv_reranker.cross_validate(
        valid_pairs, n_folds=5, top_n=10, mlp_top_n=POOL_N, aux_top_n=AUX_N
    )

    print("\n5-fold CV results:")
    print(f"{'=' * 55}")
    print(f"{'Method':20s}  {'@1':>14s}  {'@5':>14s}  {'@10':>14s}")
    print(f"{'=' * 55}")
    for method in ["mlp", "reranker"]:
        m = cv_results["metrics"][method]
        print(
            f"{method:20s}  "
            f"{m['hits_at_1']['mean']:.3f}±{m['hits_at_1']['std']:.3f}  "
            f"{m['hits_at_5']['mean']:.3f}±{m['hits_at_5']['std']:.3f}  "
            f"{m['hits_at_10']['mean']:.3f}±{m['hits_at_10']['std']:.3f}"
        )
    print(f"{'=' * 55}")
    print(f"  ({cv_results['total_pairs']} valid pairs, {cv_results['n_folds']} folds)")

    # ── Canonical test cases ──────────────────────────────────────────
    canonical = [
        ("king", "queen"),
        ("boy", "girl"),
        ("father", "mother"),
        ("husband", "wife"),
        ("happy", "sad"),
        ("good", "bad"),
        ("love", "hate"),
        ("hot", "cold"),
        ("warm", "cool"),
        ("big", "small"),
        ("alive", "dead"),
        ("up", "down"),
    ]
    print(f"\n{'=' * 55}")
    print("Canonical test cases")
    print(f"{'=' * 55}")
    print(f"{'Pair':16s}  {'MLP top-1':16s}  {'Reranker top-1':16s}")
    print("-" * 55)
    for w1, w2 in canonical:
        mlp_top = mlp.retrieve(w1, top_n=1)
        mlp_pred = mlp_top[0][0] if mlp_top else "?"
        rer_top = fn_reranker(w1)
        rer_pred = rer_top[0] if rer_top else "?"

        def mark(pred, target):
            return ("✓ " if pred == target else "✗ ") + pred[:12]

        print(f"{w1 + '->' + w2:16s}  {mark(mlp_pred, w2):16s}  {mark(rer_pred, w2):16s}")

    # ── Save results ──────────────────────────────────────────────────
    out = {
        "holdout": {
            "mlp": {str(k): v for k, v in mlp_hits.items()},
            "reranker": {str(k): v for k, v in rer_hits.items()},
        },
        "cv": cv_results,
        "feature_weights": weights,
    }
    with open("results/reranker_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/reranker_eval.json")


if __name__ == "__main__":
    main()
