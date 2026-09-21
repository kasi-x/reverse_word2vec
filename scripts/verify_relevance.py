"""
Verify relevance hypotheses:
1. Does high z-score correlate with inversion success?
2. Does symmetric axis loading predict success?
3. Does multi-axis inversion help asymmetric cases?
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from src.antonym_loader import extract_antonym_pairs
from src.axis_labeler import AxisLabeler
from src.ica_transformer import load_ica_space
from src.relevance import RelevanceScorer
from src.semantic_operations import SemanticOperator
from src.word2vec_loader import Word2VecLoader


def main():
    # Load everything
    print("Loading model and ICA space...")
    loader = Word2VecLoader()
    model = loader.load_glove(100)
    space = load_ica_space("results/ica_space")
    labeler = AxisLabeler(space)
    profiles = labeler.load_profiles("results/axis_profiles.json")
    operator = SemanticOperator(model, space)
    scorer = RelevanceScorer(space, profiles)

    antonym_pairs = extract_antonym_pairs()
    valid_pairs = [
        (w1, w2)
        for w1, w2 in antonym_pairs
        if space.score(w1) is not None and space.score(w2) is not None
    ]
    print(f"Valid antonym pairs: {len(valid_pairs)}")

    axis_std = np.std(space.S, axis=0)
    axis_std = np.maximum(axis_std, 1e-8)

    # ── Experiment 1: z-score vs inversion success ──────────────
    print("\n" + "=" * 60)
    print("EXP 1: Does best-axis z-score predict inversion success?")
    print("=" * 60)

    bins = {"z<2": [], "2<=z<3": [], "3<=z<4": [], "z>=4": []}

    for w1, w2 in valid_pairs:
        s1 = space.score(w1)
        s2 = space.score(w2)
        diff = np.abs(s1 - s2)
        norm_diff = diff / axis_std
        best_axis = int(np.argmax(norm_diff))
        best_z = float(norm_diff[best_axis])

        # Try inversion in both directions
        success = False
        for src, tgt in [(w1, w2), (w2, w1)]:
            neighbors = operator.axis_invert(src, best_axis, top_n=10)
            if any(n.word == tgt for n in neighbors):
                success = True
                break

        if best_z < 2:
            bins["z<2"].append(success)
        elif best_z < 3:
            bins["2<=z<3"].append(success)
        elif best_z < 4:
            bins["3<=z<4"].append(success)
        else:
            bins["z>=4"].append(success)

    for label, results in bins.items():
        if results:
            rate = sum(results) / len(results)
            print(f"  {label:10s}: {sum(results):4d}/{len(results):4d} = {rate:.1%}")

    # ── Experiment 2: Symmetric loading predicts success? ──────
    print("\n" + "=" * 60)
    print("EXP 2: Does symmetric axis loading predict success?")
    print("  (Both words have high |z| on the same axis)")
    print("=" * 60)

    sym_bins = {"both_high": [], "one_high": [], "neither_high": []}

    for w1, w2 in valid_pairs:
        s1 = space.score(w1)
        s2 = space.score(w2)
        diff = np.abs(s1 - s2)
        norm_diff = diff / axis_std
        best_axis = int(np.argmax(norm_diff))

        z1 = abs(float(s1[best_axis])) / float(axis_std[best_axis])
        z2 = abs(float(s2[best_axis])) / float(axis_std[best_axis])

        success = False
        for src, tgt in [(w1, w2), (w2, w1)]:
            neighbors = operator.axis_invert(src, best_axis, top_n=10)
            if any(n.word == tgt for n in neighbors):
                success = True
                break

        if z1 >= 2.0 and z2 >= 2.0:
            sym_bins["both_high"].append(success)
        elif z1 >= 2.0 or z2 >= 2.0:
            sym_bins["one_high"].append(success)
        else:
            sym_bins["neither_high"].append(success)

    for label, results in sym_bins.items():
        if results:
            rate = sum(results) / len(results)
            print(f"  {label:14s}: {sum(results):4d}/{len(results):4d} = {rate:.1%}")

    # ── Experiment 3: Multi-axis inversion ──────────────────────
    print("\n" + "=" * 60)
    print("EXP 3: Multi-axis inversion (top-k axes) vs single axis")
    print("=" * 60)

    results_by_k = {1: 0, 2: 0, 3: 0, 5: 0}
    total = 0
    # Use a random sample for speed
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(valid_pairs), size=min(500, len(valid_pairs)), replace=False)
    sample_pairs = [valid_pairs[i] for i in sample_idx]

    for w1, w2 in sample_pairs:
        s1 = space.score(w1)
        s2 = space.score(w2)
        diff = np.abs(s1 - s2)
        norm_diff = diff / axis_std
        top_axes = list(np.argsort(norm_diff)[::-1])

        total += 1
        for k in results_by_k:
            axes_to_flip = top_axes[:k]
            success = False
            for src, tgt in [(w1, w2), (w2, w1)]:
                if k == 1:
                    neighbors = operator.axis_invert(src, axes_to_flip[0], top_n=10)
                else:
                    neighbors = operator.multi_axis_invert(src, axes_to_flip, top_n=10)
                if any(n.word == tgt for n in neighbors):
                    success = True
                    break
            if success:
                results_by_k[k] += 1

    for k, hits in results_by_k.items():
        print(f"  top-{k} axes: {hits}/{total} = {hits / total:.1%}")

    # ── Experiment 4: Asymmetric cases - does multi-axis help? ──
    print("\n" + "=" * 60)
    print("EXP 4: Asymmetric cases (one_high) - multi-axis rescue?")
    print("=" * 60)

    # Collect asymmetric pairs
    asym_pairs = []
    for w1, w2 in valid_pairs:
        s1 = space.score(w1)
        s2 = space.score(w2)
        diff = np.abs(s1 - s2)
        norm_diff = diff / axis_std
        best_axis = int(np.argmax(norm_diff))
        z1 = abs(float(s1[best_axis])) / float(axis_std[best_axis])
        z2 = abs(float(s2[best_axis])) / float(axis_std[best_axis])
        if (z1 >= 2.0) != (z2 >= 2.0):  # exactly one is high
            asym_pairs.append((w1, w2))

    print(f"  Asymmetric pairs: {len(asym_pairs)}")

    if asym_pairs:
        asym_sample = asym_pairs[: min(200, len(asym_pairs))]
        asym_results = {1: 0, 3: 0, 5: 0}
        for w1, w2 in asym_sample:
            s1 = space.score(w1)
            s2 = space.score(w2)
            diff = np.abs(s1 - s2)
            norm_diff = diff / axis_std
            top_axes = list(np.argsort(norm_diff)[::-1])

            for k in asym_results:
                axes_to_flip = top_axes[:k]
                success = False
                for src, tgt in [(w1, w2), (w2, w1)]:
                    if k == 1:
                        neighbors = operator.axis_invert(src, axes_to_flip[0], top_n=10)
                    else:
                        neighbors = operator.multi_axis_invert(src, axes_to_flip, top_n=10)
                    if any(n.word == tgt for n in neighbors):
                        success = True
                        break
                if success:
                    asym_results[k] += 1

        n = len(asym_sample)
        for k, hits in asym_results.items():
            print(f"  top-{k} axes: {hits}/{n} = {hits / n:.1%}")

    # ── Experiment 5: Canonical cases deep dive ─────────────────
    print("\n" + "=" * 60)
    print("EXP 5: Canonical cases - per-word confidence and axis overlap")
    print("=" * 60)

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
        ("huge", "tiny"),
        ("up", "down"),
        ("high", "low"),
        ("alive", "dead"),
        ("begin", "end"),
    ]

    for w1, w2 in canonical:
        s1 = space.score(w1)
        s2 = space.score(w2)
        if s1 is None or s2 is None:
            print(f"  {w1:10s} -> {w2:10s}: NOT IN VOCAB")
            continue

        diff = np.abs(s1 - s2)
        norm_diff = diff / axis_std
        best_axis = int(np.argmax(norm_diff))
        best_z = float(norm_diff[best_axis])

        z1 = abs(float(s1[best_axis])) / float(axis_std[best_axis])
        z2 = abs(float(s2[best_axis])) / float(axis_std[best_axis])

        # Check top-5 axes overlap
        top5_w1 = set(np.argsort(np.abs(s1) / axis_std)[-5:])
        top5_w2 = set(np.argsort(np.abs(s2) / axis_std)[-5:])
        overlap = len(top5_w1 & top5_w2)

        # Single axis inversion
        n1 = operator.axis_invert(w1, best_axis, top_n=5)
        single_result = n1[0].word if n1 else "?"

        # Multi-axis (top 3)
        top3 = list(np.argsort(norm_diff)[-3:][::-1])
        n3 = operator.multi_axis_invert(w1, top3, top_n=5)
        multi_result = n3[0].word if n3 else "?"

        label = scorer._label_map.get(best_axis, "")
        sym = "SYM" if z1 >= 2.0 and z2 >= 2.0 else ("ASYM" if z1 >= 2.0 or z2 >= 2.0 else "WEAK")

        hit1 = "o" if single_result == w2 else "x"
        hit3 = "o" if multi_result == w2 else "x"

        print(
            f"  {w1:10s}->{w2:10s}  axis={best_axis:2d}({label:12s})  "
            f"z_diff={best_z:.1f}  z1={z1:.1f} z2={z2:.1f}  {sym:4s}  "
            f"overlap={overlap}  1ax[{hit1}]={single_result:10s}  3ax[{hit3}]={multi_result:10s}"
        )


if __name__ == "__main__":
    main()
