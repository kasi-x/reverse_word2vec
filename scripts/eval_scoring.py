"""
Compare scoring criteria for label-group selection in scan strategy.

The question: given a word, how to pick the RIGHT label group to invert?
"""

import sys

import numpy as np

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

    label_to_axes = {}
    for p in profiles:
        if p.label:
            label_to_axes.setdefault(p.label, []).append(p.axis_idx)

    antonym_pairs = extract_antonym_pairs()
    valid_pairs = [
        (w1, w2)
        for w1, w2 in antonym_pairs
        if space.score(w1) is not None and space.score(w2) is not None
    ]

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(valid_pairs), size=min(500, len(valid_pairs)), replace=False)
    sample = [valid_pairs[i] for i in sample_idx]
    print(f"Evaluating {len(sample)} pairs\n")

    TOP_N = 10

    # Scoring criteria to compare
    criteria = {
        "dissimilarity": {},  # 1 - cos(source, top_neighbor) [current]
        "source_z_mean": {},  # mean |z| of source on group's axes
        "source_z_max": {},  # max |z| of source on group's axes
        "score_change": {},  # L2 of ICA score change from inversion
        "combined": {},  # source_z_mean * score_change
    }
    for c in criteria:
        criteria[c] = {
            "hits@1": 0,
            "hits@5": 0,
            "hits@10": 0,
            "top3_hits@1": 0,
            "top3_hits@5": 0,
            "top3_hits@10": 0,
        }
    total = 0

    for i, (w1, w2) in enumerate(sample):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sample)}...")

        total += 1

        # For each scoring criterion, rank label groups and check results
        for src, tgt in [(w1, w2), (w2, w1)]:
            s_src = space.score(src)
            if s_src is None:
                continue
            src_vec = model[src].astype(np.float32)
            src_norm = src_vec / max(np.linalg.norm(src_vec), 1e-10)

            group_scores = {c: [] for c in criteria}  # criterion -> [(score, rank, label)]

            for label, axes in label_to_axes.items():
                nbrs = operator.multi_axis_invert(src, axes, top_n=TOP_N)
                if not nbrs:
                    continue

                rank = None
                for j, n in enumerate(nbrs):
                    if n.word == tgt:
                        rank = j + 1
                        break

                # Compute scores
                top_vec = model[nbrs[0].word].astype(np.float32)
                top_norm = top_vec / max(np.linalg.norm(top_vec), 1e-10)
                dissim = 1.0 - float(src_norm @ top_norm)

                z_on_axes = np.abs(s_src[axes]) / axis_std[axes]
                z_mean = float(np.mean(z_on_axes))
                z_max = float(np.max(z_on_axes))

                # Score change: how much the ICA representation changed
                modified = s_src.copy()
                for a in axes:
                    modified[a] = -modified[a]
                score_change = float(np.linalg.norm(s_src - modified))

                combined = z_mean * score_change

                group_scores["dissimilarity"].append((dissim, rank, label))
                group_scores["source_z_mean"].append((z_mean, rank, label))
                group_scores["source_z_max"].append((z_max, rank, label))
                group_scores["score_change"].append((score_change, rank, label))
                group_scores["combined"].append((combined, rank, label))

            # For each criterion, rank groups and check if best/top-3 hit
            for _criterion, groups in group_scores.items():
                if not groups:
                    continue
                groups.sort(key=lambda x: x[0], reverse=True)

                # Best group
                _, best_rank, _ = groups[0]
                if best_rank is not None:
                    for kk, _key in [(1, "hits@1"), (5, "hits@5"), (10, "hits@10")]:
                        if best_rank <= kk:
                            # Only count the better of the two directions
                            pass  # handled below

                # Top-3 groups
                top3_best = None
                for _, rank, _ in groups[:3]:
                    if rank is not None and (top3_best is None or rank < top3_best):
                        top3_best = rank

        # Now do bidirectional: for each criterion, try both directions and take best
        for c_name in criteria:
            best_rank_1 = None
            best_rank_3 = None

            for src, tgt in [(w1, w2), (w2, w1)]:
                s_src = space.score(src)
                if s_src is None:
                    continue
                src_vec = model[src].astype(np.float32)
                src_norm = src_vec / max(np.linalg.norm(src_vec), 1e-10)

                group_entries = []
                for _label, axes in label_to_axes.items():
                    nbrs = operator.multi_axis_invert(src, axes, top_n=TOP_N)
                    if not nbrs:
                        continue
                    rank = None
                    for j, n in enumerate(nbrs):
                        if n.word == tgt:
                            rank = j + 1
                            break

                    z_on_axes = np.abs(s_src[axes]) / axis_std[axes]
                    z_mean = float(np.mean(z_on_axes))
                    z_max = float(np.max(z_on_axes))

                    modified = s_src.copy()
                    for a in axes:
                        modified[a] = -modified[a]
                    score_change = float(np.linalg.norm(s_src - modified))

                    top_vec = model[nbrs[0].word].astype(np.float32)
                    top_norm = top_vec / max(np.linalg.norm(top_vec), 1e-10)
                    dissim = 1.0 - float(src_norm @ top_norm)

                    score_map = {
                        "dissimilarity": dissim,
                        "source_z_mean": z_mean,
                        "source_z_max": z_max,
                        "score_change": score_change,
                        "combined": z_mean * score_change,
                    }
                    group_entries.append((score_map[c_name], rank))

                if not group_entries:
                    continue
                group_entries.sort(key=lambda x: x[0], reverse=True)

                # Best pick
                _, r = group_entries[0]
                if r is not None and (best_rank_1 is None or r < best_rank_1):
                    best_rank_1 = r

                # Top-3 pick
                for _, r in group_entries[:3]:
                    if r is not None and (best_rank_3 is None or r < best_rank_3):
                        best_rank_3 = r

            if best_rank_1 is not None:
                for kk, key in [(1, "hits@1"), (5, "hits@5"), (10, "hits@10")]:
                    if best_rank_1 <= kk:
                        criteria[c_name][key] += 1

            if best_rank_3 is not None:
                for kk, key in [(1, "top3_hits@1"), (5, "top3_hits@5"), (10, "top3_hits@10")]:
                    if best_rank_3 <= kk:
                        criteria[c_name][key] += 1

    print(f"\n{'=' * 80}")
    print(
        f"{'Criterion':20s}  {'Best@1':>8s}  {'Best@5':>8s}  {'Best@10':>8s}  "
        f"{'Top3@1':>8s}  {'Top3@5':>8s}  {'Top3@10':>8s}"
    )
    print(f"{'=' * 80}")
    for name, data in criteria.items():
        print(
            f"{name:20s}  "
            f"{data['hits@1'] / total:8.1%}  {data['hits@5'] / total:8.1%}  {data['hits@10'] / total:8.1%}  "
            f"{data['top3_hits@1'] / total:8.1%}  {data['top3_hits@5'] / total:8.1%}  {data['top3_hits@10'] / total:8.1%}"
        )
    print(f"{'=' * 80}")
    print(f"Total: {total}")
    print("\n  Best = pick 1 label group; Top3 = check top-3 label groups")
    print("  For reference: oracle_1 ≈ 22.6% @1, label_oracle ≈ 20.2% @1")


if __name__ == "__main__":
    main()
