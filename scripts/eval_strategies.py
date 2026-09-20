"""
Compare multi-axis inversion strategies on antonym retrieval.

Strategies:
  1. oracle_1   - best single axis (knows target) [baseline]
  2. oracle_k   - top-k axes by pair diff (knows target) [upper bound]
  3. source_topk - top-k axes by source word's |z-score| (practical)
  4. label_group - flip all axes with matching label (practical, needs label)
  5. scan_best  - try all label groups, pick best result (practical, automatic)
"""
import numpy as np
import sys
sys.path.insert(0, ".")

from src.antonym_loader import extract_antonym_pairs
from src.axis_labeler import AxisLabeler
from src.ica_transformer import load_ica_space, ICATransformer
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
    transformer = ICATransformer()

    antonym_pairs = extract_antonym_pairs()
    valid_pairs = [
        (w1, w2) for w1, w2 in antonym_pairs
        if space.score(w1) is not None and space.score(w2) is not None
    ]

    axis_std = np.std(space.S, axis=0)
    axis_std = np.maximum(axis_std, 1e-8)

    # Build label->axes map
    label_to_axes = {}
    for p in profiles:
        if p.label:
            label_to_axes.setdefault(p.label, []).append(p.axis_idx)

    # For each pair, find the label of the oracle best axis
    axis_to_label = {}
    for p in profiles:
        if p.label:
            axis_to_label[p.axis_idx] = p.label

    TOP_N = 10
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(valid_pairs), size=min(500, len(valid_pairs)), replace=False)
    sample = [valid_pairs[i] for i in sample_idx]
    print(f"Evaluating {len(sample)} pairs\n")

    strategies = {
        "oracle_1":     {"hits": {1: 0, 5: 0, 10: 0}},
        "oracle_3":     {"hits": {1: 0, 5: 0, 10: 0}},
        "oracle_5":     {"hits": {1: 0, 5: 0, 10: 0}},
        "source_top3":  {"hits": {1: 0, 5: 0, 10: 0}},
        "source_top5":  {"hits": {1: 0, 5: 0, 10: 0}},
        "source_top10": {"hits": {1: 0, 5: 0, 10: 0}},
        "label_oracle": {"hits": {1: 0, 5: 0, 10: 0}},
        "scan_best":    {"hits": {1: 0, 5: 0, 10: 0}},
        "scan_top3":    {"hits": {1: 0, 5: 0, 10: 0}},
    }
    total = 0

    for i, (w1, w2) in enumerate(sample):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sample)}...")

        s1 = space.score(w1)
        s2 = space.score(w2)

        # Oracle axis selection (knows both words)
        diff = np.abs(s1 - s2)
        norm_diff = diff / axis_std
        oracle_axes = list(np.argsort(norm_diff)[::-1])

        # Source-only axis selection (knows only source)
        z1 = np.abs(s1) / axis_std
        source_axes = list(np.argsort(z1)[::-1])

        total += 1

        def check_hit(neighbors, target):
            """Return rank of target in neighbors (1-indexed), or None."""
            for j, n in enumerate(neighbors):
                if n.word == target:
                    return j + 1
            return None

        def best_rank_bidirectional(axes, k_axes=None):
            """Try inversion in both directions, return best rank."""
            if k_axes is None:
                k_axes = axes
            best = None
            for src, tgt in [(w1, w2), (w2, w1)]:
                if len(k_axes) == 1:
                    nbrs = operator.axis_invert(src, k_axes[0], top_n=TOP_N)
                else:
                    nbrs = operator.multi_axis_invert(src, k_axes, top_n=TOP_N)
                rank = check_hit(nbrs, tgt)
                if rank is not None and (best is None or rank < best):
                    best = rank
            return best

        # Oracle strategies
        for name, k in [("oracle_1", 1), ("oracle_3", 3), ("oracle_5", 5)]:
            rank = best_rank_bidirectional(oracle_axes, oracle_axes[:k])
            if rank is not None:
                for kk in [1, 5, 10]:
                    if rank <= kk:
                        strategies[name]["hits"][kk] += 1

        # Source top-k strategies
        for name, k in [("source_top3", 3), ("source_top5", 5), ("source_top10", 10)]:
            rank = best_rank_bidirectional(source_axes, source_axes[:k])
            if rank is not None:
                for kk in [1, 5, 10]:
                    if rank <= kk:
                        strategies[name]["hits"][kk] += 1

        # Label-oracle: use the label of the oracle best axis, flip ALL axes with that label
        oracle_best_label = axis_to_label.get(oracle_axes[0], "")
        if oracle_best_label and oracle_best_label in label_to_axes:
            axes = label_to_axes[oracle_best_label]
            rank = best_rank_bidirectional(axes, axes)
            if rank is not None:
                for kk in [1, 5, 10]:
                    if rank <= kk:
                        strategies["label_oracle"]["hits"][kk] += 1

        # Scan: try all label groups, pick the one whose top-1 is most different from source
        best_scan_rank = None
        scan_results_all = []
        for src, tgt in [(w1, w2), (w2, w1)]:
            src_vec = model[src].astype(np.float32)
            src_norm = src_vec / max(np.linalg.norm(src_vec), 1e-10)
            for label, axes in label_to_axes.items():
                nbrs = operator.multi_axis_invert(src, axes, top_n=TOP_N)
                if not nbrs:
                    continue
                top_vec = model[nbrs[0].word].astype(np.float32)
                top_norm = top_vec / max(np.linalg.norm(top_vec), 1e-10)
                dissim = 1.0 - float(src_norm @ top_norm)
                rank = check_hit(nbrs, tgt)
                scan_results_all.append((dissim, rank, label, nbrs))

        # scan_best: pick label group with max dissimilarity
        if scan_results_all:
            scan_results_all.sort(key=lambda x: x[0], reverse=True)
            _, best_scan_rank, _, _ = scan_results_all[0]
            if best_scan_rank is not None:
                for kk in [1, 5, 10]:
                    if best_scan_rank <= kk:
                        strategies["scan_best"]["hits"][kk] += 1

            # scan_top3: check top-3 label groups by dissimilarity
            best_from_top3 = None
            for _, rank, _, _ in scan_results_all[:3]:
                if rank is not None and (best_from_top3 is None or rank < best_from_top3):
                    best_from_top3 = rank
            if best_from_top3 is not None:
                for kk in [1, 5, 10]:
                    if best_from_top3 <= kk:
                        strategies["scan_top3"]["hits"][kk] += 1

    # ── Print results ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Strategy':20s}  {'Hits@1':>8s}  {'Hits@5':>8s}  {'Hits@10':>8s}")
    print(f"{'='*70}")
    for name, data in strategies.items():
        h1 = data["hits"][1] / total
        h5 = data["hits"][5] / total
        h10 = data["hits"][10] / total
        marker = " *" if "oracle" not in name else "  "
        print(f"{name:20s}  {h1:8.1%}  {h5:8.1%}  {h10:8.1%}{marker}")
    print(f"{'='*70}")
    print(f"Total pairs: {total}")
    print(f"  * = practical (no oracle)")

    # ── Canonical deep dive ───────────────────────────────────
    print(f"\n{'='*70}")
    print("Canonical cases: strategy comparison")
    print(f"{'='*70}")

    canonical = [
        ("king", "queen", "gender"), ("boy", "girl", "gender"),
        ("father", "mother", "gender"), ("husband", "wife", "gender"),
        ("happy", "sad", "sentiment"), ("good", "bad", "sentiment"),
        ("hot", "cold", "temperature"), ("big", "small", "size"),
        ("alive", "dead", "activity"), ("up", "down", "direction"),
    ]

    print(f"{'pair':20s}  {'oracle1':>10s}  {'oracle5':>10s}  {'src_top5':>10s}  {'label':>10s}  {'scan':>10s}")
    print("-" * 70)

    for w1, w2, expected_label in canonical:
        s1 = space.score(w1)
        s2 = space.score(w2)
        if s1 is None or s2 is None:
            continue

        diff = np.abs(s1 - s2)
        norm_diff = diff / axis_std
        oracle_axes = list(np.argsort(norm_diff)[::-1])

        z1 = np.abs(s1) / axis_std
        source_axes = list(np.argsort(z1)[::-1])

        def get_top1(axes_list):
            for src, tgt in [(w1, w2), (w2, w1)]:
                if len(axes_list) == 1:
                    nbrs = operator.axis_invert(src, axes_list[0], top_n=5)
                else:
                    nbrs = operator.multi_axis_invert(src, axes_list, top_n=5)
                if nbrs:
                    return nbrs[0].word
            return "?"

        r_oracle1 = get_top1(oracle_axes[:1])
        r_oracle5 = get_top1(oracle_axes[:5])
        r_src5 = get_top1(source_axes[:5])

        # Label group
        label_axes = label_to_axes.get(expected_label, [])
        r_label = get_top1(label_axes) if label_axes else "no_axis"

        # Scan best
        src_vec = model[w1].astype(np.float32)
        src_norm_v = src_vec / max(np.linalg.norm(src_vec), 1e-10)
        best_dissim = -1
        r_scan = "?"
        for label, axes in label_to_axes.items():
            nbrs = operator.multi_axis_invert(w1, axes, top_n=5)
            if not nbrs:
                continue
            top_vec = model[nbrs[0].word].astype(np.float32)
            top_norm_v = top_vec / max(np.linalg.norm(top_vec), 1e-10)
            d = 1.0 - float(src_norm_v @ top_norm_v)
            if d > best_dissim:
                best_dissim = d
                r_scan = f"{nbrs[0].word}({label[:4]})"

        pair = f"{w1}->{w2}"
        def mark(result, target):
            return f"{'o':>1s} {result}" if result == target else f"{'x':>1s} {result}"

        print(f"{pair:20s}  {mark(r_oracle1, w2):>10s}  {mark(r_oracle5, w2):>10s}  "
              f"{mark(r_src5, w2):>10s}  {mark(r_label, w2):>10s}  {r_scan:>10s}")


if __name__ == "__main__":
    main()
