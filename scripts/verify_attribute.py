"""
Verify: when a SOURCE word has high |z| on an axis,
does single-axis inversion on that axis succeed more often?

This tests the "attribute-based inversion" hypothesis:
  1. Identify word's strong axes (attributes)
  2. Invert on each individually
  3. Check if the antonym appears
"""
import numpy as np
import sys
sys.path.insert(0, ".")

from src.antonym_loader import extract_antonym_pairs
from src.axis_labeler import AxisLabeler
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

    axis_std = np.std(space.S, axis=0)
    axis_std = np.maximum(axis_std, 1e-8)

    antonym_pairs = extract_antonym_pairs()
    valid_pairs = [
        (w1, w2) for w1, w2 in antonym_pairs
        if space.score(w1) is not None and space.score(w2) is not None
    ]
    print(f"Valid pairs: {len(valid_pairs)}\n")

    TOP_N = 10

    # ── EXP 1: Source z-score vs single-axis inversion success ──
    # For ALL (word, axis) combinations where |z_source| > threshold,
    # check if inverting on that axis retrieves the antonym.
    print("=" * 60)
    print("EXP 1: Source word's |z| on axis vs inversion success")
    print("  (per-axis, not per-pair)")
    print("=" * 60)

    bins = {
        "z<1":   {"attempts": 0, "hits": 0},
        "1<=z<2": {"attempts": 0, "hits": 0},
        "2<=z<3": {"attempts": 0, "hits": 0},
        "3<=z<4": {"attempts": 0, "hits": 0},
        "z>=4":   {"attempts": 0, "hits": 0},
    }

    for w1, w2 in valid_pairs:
        s1 = space.score(w1)
        z1 = np.abs(s1) / axis_std

        for k in range(space.n_components):
            z = float(z1[k])
            if z < 1:
                b = "z<1"
            elif z < 2:
                b = "1<=z<2"
            elif z < 3:
                b = "2<=z<3"
            elif z < 4:
                b = "3<=z<4"
            else:
                b = "z>=4"

            # Only check axes above z=1 for speed (below is too many)
            if z < 1:
                # Sample 1% of z<1 cases
                if np.random.random() > 0.01:
                    continue

            bins[b]["attempts"] += 1
            nbrs = operator.axis_invert(w1, k, top_n=TOP_N)
            if any(n.word == w2 for n in nbrs):
                bins[b]["hits"] += 1

    for label, data in bins.items():
        if data["attempts"] > 0:
            rate = data["hits"] / data["attempts"]
            print(f"  {label:8s}: {data['hits']:5d}/{data['attempts']:5d} = {rate:.1%}")

    # ── EXP 2: Attribute-based inversion strategy ──────────────
    # For each pair: find source's strong axes (z>2), invert each
    # individually, check if ANY produces the antonym.
    print(f"\n{'='*60}")
    print("EXP 2: Attribute-based strategy")
    print("  For each word, try inverting on each axis where |z|>threshold")
    print("=" * 60)

    for threshold in [1.5, 2.0, 2.5, 3.0]:
        hits = 0
        total = 0
        n_axes_tried = []

        for w1, w2 in valid_pairs:
            total += 1
            s1 = space.score(w1)
            z1 = np.abs(s1) / axis_std
            strong_axes = np.where(z1 >= threshold)[0]
            n_axes_tried.append(len(strong_axes))

            found = False
            for k in strong_axes:
                nbrs = operator.axis_invert(w1, int(k), top_n=TOP_N)
                if any(n.word == w2 for n in nbrs):
                    found = True
                    break
            if found:
                hits += 1

        mean_axes = np.mean(n_axes_tried) if n_axes_tried else 0
        print(f"  z>={threshold:.1f}: {hits}/{total} = {hits/total:.1%}  "
              f"(avg {mean_axes:.1f} axes tried per word)")

    # ── EXP 3: Bidirectional attribute-based ───────────────────
    print(f"\n{'='*60}")
    print("EXP 3: Bidirectional attribute-based (try both directions)")
    print("=" * 60)

    for threshold in [1.5, 2.0, 2.5, 3.0]:
        hits_at = {1: 0, 5: 0, 10: 0}
        total = 0

        for w1, w2 in valid_pairs:
            total += 1
            best_rank = None

            for src, tgt in [(w1, w2), (w2, w1)]:
                s = space.score(src)
                z = np.abs(s) / axis_std
                strong_axes = np.where(z >= threshold)[0]

                for k in strong_axes:
                    nbrs = operator.axis_invert(src, int(k), top_n=TOP_N)
                    for j, n in enumerate(nbrs):
                        if n.word == tgt:
                            rank = j + 1
                            if best_rank is None or rank < best_rank:
                                best_rank = rank
                            break

            if best_rank is not None:
                for kk in [1, 5, 10]:
                    if best_rank <= kk:
                        hits_at[kk] += 1

        print(f"  z>={threshold:.1f}: Hits@1={hits_at[1]/total:.1%}  "
              f"Hits@5={hits_at[5]/total:.1%}  Hits@10={hits_at[10]/total:.1%}")

    # ── EXP 4: Canonical cases ─────────────────────────────────
    print(f"\n{'='*60}")
    print("EXP 4: Canonical cases - attribute-based inversion (z>=2)")
    print("=" * 60)

    canonical = [
        ("king", "queen"), ("boy", "girl"), ("father", "mother"),
        ("husband", "wife"), ("happy", "sad"), ("good", "bad"),
        ("love", "hate"), ("hot", "cold"), ("warm", "cool"),
        ("big", "small"), ("alive", "dead"), ("up", "down"),
    ]

    axis_to_label = {}
    for p in profiles:
        if p.label:
            axis_to_label[p.axis_idx] = p.label

    for w1, w2 in canonical:
        s1 = space.score(w1)
        if s1 is None:
            continue
        z1 = np.abs(s1) / axis_std
        strong_axes = np.where(z1 >= 2.0)[0]
        # Sort by z descending
        strong_axes = strong_axes[np.argsort(z1[strong_axes])[::-1]]

        print(f"\n  {w1} -> {w2}  ({len(strong_axes)} strong axes)")
        for k in strong_axes[:8]:
            label = axis_to_label.get(int(k), "")
            nbrs = operator.axis_invert(w1, int(k), top_n=5)
            top3 = [n.word for n in nbrs[:3]]
            hit = "***" if w2 in [n.word for n in nbrs[:10]] else "   "
            print(f"    axis {k:2d} ({label:15s}) z={z1[k]:.1f}  "
                  f"-> {', '.join(top3)} {hit}")


if __name__ == "__main__":
    main()
