"""
Comprehensive evaluation of all inversion approaches.
"""

import json
import sys

import numpy as np

sys.path.insert(0, ".")

from src.antonym_classifier import AntonymClassifier
from src.antonym_loader import extract_antonym_pairs
from src.axis_labeler import AxisLabeler
from src.contrastive_retrieval import ContrastiveRetriever
from src.eval_utils import compute_hits
from src.ica_transformer import load_ica_space
from src.semantic_operations import SemanticOperator
from src.word2vec_loader import Word2VecLoader


def main():
    print("Loading...")
    loader = Word2VecLoader()
    model = loader.load_glove(100)
    space = load_ica_space("results/ica_space")
    labeler = AxisLabeler(space)
    profiles = labeler.load_profiles("results/axis_profiles.json")
    operator = SemanticOperator(model, space, profiles)
    retriever = ContrastiveRetriever(space, lam=1.0)

    axis_std = np.std(space.S, axis=0)
    axis_std = np.maximum(axis_std, 1e-8)

    antonym_pairs = extract_antonym_pairs()
    valid_pairs = [
        (w1, w2)
        for w1, w2 in antonym_pairs
        if space.score(w1) is not None and space.score(w2) is not None
    ]
    print(f"Valid pairs: {len(valid_pairs)}")

    rng = np.random.RandomState(42)
    # 80/20 split for classifier
    all_idx = np.arange(len(valid_pairs))
    rng.shuffle(all_idx)
    n_train = int(len(valid_pairs) * 0.8)
    train_pairs = [valid_pairs[i] for i in all_idx[:n_train]]
    test_pairs = [valid_pairs[i] for i in all_idx[n_train:]]

    # Sample for oracle evaluation (needs both words)
    sample_idx = rng.choice(len(valid_pairs), size=min(500, len(valid_pairs)), replace=False)
    sample = [valid_pairs[i] for i in sample_idx]

    # Tune lambda
    print("\nTuning contrastive lambda...")

    def oracle_axis_fn(w1, w2):
        s1, s2 = space.score(w1), space.score(w2)
        if s1 is None or s2 is None:
            return None
        return int(np.argmax(np.abs(s1 - s2) / axis_std))

    best_lam = retriever.tune_lambda(train_pairs[:100], oracle_axis_fn)
    print(f"  Best lambda = {best_lam}")
    retriever.lam = best_lam

    # Train classifiers
    print("\nTraining classifiers...")
    clf_lr = AntonymClassifier(space, "logistic")
    clf_lr.fit(train_pairs)
    print("  LR done")

    clf_mlp = AntonymClassifier(space, "mlp")
    clf_mlp.fit(train_pairs)
    print("  MLP done")

    TOP_N = 10

    # Retrieval functions (practical — no oracle)
    def fn_contrastive_src(src):
        s = space.score(src)
        if s is None:
            return []
        k = int(np.argmax(np.abs(s) / axis_std))
        return [r.word for r in retriever.search(src, k, top_n=TOP_N)]

    def fn_attribute(src):
        results = operator.invert_by_attributes(src, z_threshold=2.0, top_n=TOP_N)
        seen = {}
        for res in results:
            for n in res.neighbors:
                if n.word not in seen:
                    seen[n.word] = n.similarity
        return sorted(seen, key=lambda w: seen[w], reverse=True)[:TOP_N]

    def fn_attribute_15(src):
        results = operator.invert_by_attributes(src, z_threshold=1.5, top_n=TOP_N)
        seen = {}
        for res in results:
            for n in res.neighbors:
                if n.word not in seen:
                    seen[n.word] = n.similarity
        return sorted(seen, key=lambda w: seen[w], reverse=True)[:TOP_N]

    def fn_clf_lr(src):
        return [w for w, _ in clf_lr.retrieve(src, top_n=TOP_N)]

    def fn_clf_mlp(src):
        return [w for w, _ in clf_mlp.retrieve(src, top_n=TOP_N)]

    # Oracle baselines on sample
    print("\nComputing oracle baselines on sample...")
    oracle1_hits = {1: 0, 5: 0, 10: 0}
    oracle5_hits = {1: 0, 5: 0, 10: 0}
    cr_oracle_hits = {1: 0, 5: 0, 10: 0}
    total_oracle = 0

    for w1, w2 in sample:
        s1, s2 = space.score(w1), space.score(w2)
        diff = np.abs(s1 - s2) / axis_std
        oracle_axes = list(np.argsort(diff)[::-1])
        best_k = oracle_axes[0]
        total_oracle += 1

        for src, tgt in [(w1, w2), (w2, w1)]:
            for j, n in enumerate(operator.axis_invert(src, best_k, top_n=TOP_N)):
                if n.word == tgt:
                    for k in [1, 5, 10]:
                        if j + 1 <= k:
                            oracle1_hits[k] += 1
                    break
            for j, n in enumerate(operator.multi_axis_invert(src, oracle_axes[:5], top_n=TOP_N)):
                if n.word == tgt:
                    for k in [1, 5, 10]:
                        if j + 1 <= k:
                            oracle5_hits[k] += 1
                    break
            for j, r in enumerate(retriever.search(src, best_k, top_n=TOP_N)):
                if r.word == tgt:
                    for k in [1, 5, 10]:
                        if j + 1 <= k:
                            cr_oracle_hits[k] += 1
                    break

    # Practical approaches on test set
    print(f"Evaluating practical approaches on {len(test_pairs)} test pairs...")
    results_table = {}
    results_table["oracle_reconstruct_1†"] = {k: oracle1_hits[k] / total_oracle for k in [1, 5, 10]}
    results_table["oracle_reconstruct_5†"] = {k: oracle5_hits[k] / total_oracle for k in [1, 5, 10]}
    results_table["oracle_contrastive_1†"] = {
        k: cr_oracle_hits[k] / total_oracle for k in [1, 5, 10]
    }

    m, _ = compute_hits(test_pairs, fn_contrastive_src)
    results_table["contrastive_src"] = m

    m, _ = compute_hits(test_pairs, fn_attribute)
    results_table["attribute_z>=2"] = m

    m, _ = compute_hits(test_pairs, fn_attribute_15)
    results_table["attribute_z>=1.5"] = m

    m, _ = compute_hits(test_pairs, fn_clf_lr)
    results_table["classifier_LR"] = m

    m, _ = compute_hits(test_pairs, fn_clf_mlp)
    results_table["classifier_MLP"] = m

    # Print table
    print(f"\n{'=' * 60}")
    print(f"{'Approach':30s}  {'@1':>7s}  {'@5':>7s}  {'@10':>7s}")
    print(f"{'=' * 60}")
    for name, metrics in results_table.items():
        print(f"{name:30s}  {metrics[1]:7.1%}  {metrics[5]:7.1%}  {metrics[10]:7.1%}")
    print(f"{'=' * 60}")
    print("† oracle (knows both words — upper bound)")

    # Lambda sweep for contrastive (oracle axis) on subset
    print("\nContrastive lambda sweep (oracle axis, n=200):")
    for lam in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
        h = {1: 0, 5: 0, 10: 0}
        n = 0
        for w1, w2 in sample[:200]:
            s1, s2 = space.score(w1), space.score(w2)
            k = int(np.argmax(np.abs(s1 - s2) / axis_std))
            n += 1
            for src, tgt in [(w1, w2), (w2, w1)]:
                for j, r in enumerate(retriever.search(src, k, top_n=TOP_N, lam=lam)):
                    if r.word == tgt:
                        for kk in [1, 5, 10]:
                            if j + 1 <= kk:
                                h[kk] += 1
                        break
        print(f"  lam={lam:.1f}: @1={h[1] / n:.1%}  @5={h[5] / n:.1%}  @10={h[10] / n:.1%}")

    # Classifier: top axes
    axis_to_label = {p.axis_idx: p.label for p in profiles if p.label}
    print("\nTop antonym axes (LR coefficients from |s1-s2| features):")
    for k, coef in clf_lr.top_antonym_axes(10):
        label = axis_to_label.get(k, "")
        print(f"  axis {k:3d} ({label:20s}): coef = {coef:+.3f}")

    # Full 5-fold CV for LR
    print(f"\n5-fold CV — logistic regression (all {len(valid_pairs)} pairs):")
    clf_cv = AntonymClassifier(space, "logistic")
    cv = clf_cv.cross_validate(valid_pairs, n_folds=5)
    for k_str, vals in cv["metrics"].items():
        print(f"  {k_str}: {vals['mean']:.3f} ± {vals['std']:.3f}")

    # Canonical cases
    print(f"\n{'=' * 70}")
    print("Canonical cases")
    print(f"{'=' * 70}")
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
    print(f"{'pair':15s}  {'attribute':>11s}  {'cr_src':>9s}  {'LR':>11s}  {'MLP':>11s}")
    print("-" * 70)
    for w1, w2 in canonical:

        def top1(fn, _w1=w1):
            ws = fn(_w1)
            return ws[0] if ws else "?"

        def m(r, _w2=w2):
            return ("o " if r == _w2 else "x ") + r[:9]

        print(
            f"{w1 + '->' + w2:15s}  {m(top1(fn_attribute)):>11s}  "
            f"{m(top1(fn_contrastive_src)):>9s}  "
            f"{m(top1(fn_clf_lr)):>11s}  {m(top1(fn_clf_mlp)):>11s}"
        )

    # Save
    out = {
        "results": {k: {str(kk): v for kk, v in ms.items()} for k, ms in results_table.items()},
        "best_lambda": best_lam,
    }
    with open("results/approach_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/approach_comparison.json")


if __name__ == "__main__":
    main()
