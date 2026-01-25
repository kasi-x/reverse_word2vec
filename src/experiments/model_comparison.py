"""
Experiment 5: Model Comparison

Compare antonym detection across different word embedding models:
- GloVe 50d, 100d, 200d, 300d
- Different training corpora

Goal: Understand how embedding quality affects antonym relationships.
"""
import sys
sys.path.insert(0, '..')

import numpy as np
from numpy.linalg import norm
from typing import List, Tuple, Dict
import time

from antonym_loader import extract_antonym_pairs
from word2vec_loader import Word2VecLoader


def evaluate_antonym_detection(model, antonym_pairs: List[Tuple[str, str]],
                                 vocab_limit: int = 30000) -> Dict:
    """
    Evaluate antonym detection quality for a model.
    """
    # Build direction matrix
    directions = []
    valid_pairs = []

    for w1, w2 in antonym_pairs:
        if w1 in model and w2 in model:
            d = model[w1] - model[w2]
            d_norm = norm(d)
            if d_norm > 1e-6:
                directions.append(d / d_norm)
                valid_pairs.append((w1, w2))

    D = np.array(directions, dtype=np.float32)

    # Get valid vocabulary
    valid_words = [
        w for w in list(model.key_to_index.keys())[:vocab_limit * 2]
        if w.isalpha() and w == w.lower() and 3 <= len(w) <= 12
    ][:vocab_limit]

    # Evaluate: for each pair, check if true antonym is found
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []

    test_pairs = valid_pairs[:500]  # Test on subset for speed

    for w1, w2 in test_pairs:
        v1 = model[w1]

        # Compute scores for all candidates
        scores = []
        for cand in valid_words:
            if cand == w1:
                continue
            if cand not in model:
                continue

            v_cand = model[cand]
            diff = v1 - v_cand
            diff_norm = norm(diff)
            if diff_norm < 1e-6:
                continue

            alignment = np.max(np.abs(D @ (diff / diff_norm)))
            scores.append((cand, alignment))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_words = [w for w, s in scores[:10]]

        if w2 in top_words:
            rank = top_words.index(w2) + 1
            reciprocal_ranks.append(1.0 / rank)
            if rank == 1:
                hits_at_1 += 1
            if rank <= 5:
                hits_at_5 += 1
            hits_at_10 += 1
        else:
            reciprocal_ranks.append(0.0)

    n = len(test_pairs)
    return {
        'n_valid_pairs': len(valid_pairs),
        'n_test_pairs': n,
        'vocab_coverage': len(valid_words) / vocab_limit,
        'hits@1': hits_at_1 / n if n > 0 else 0,
        'hits@5': hits_at_5 / n if n > 0 else 0,
        'hits@10': hits_at_10 / n if n > 0 else 0,
        'mrr': np.mean(reciprocal_ranks) if reciprocal_ranks else 0,
    }


def run_model_comparison():
    """Run the model comparison experiment."""
    print("=" * 70)
    print("EXPERIMENT 5: Model Comparison")
    print("=" * 70)

    # Load antonym pairs
    print("\n[1/2] Loading antonym pairs...")
    antonym_pairs = extract_antonym_pairs()
    print(f"  {len(antonym_pairs)} antonym pairs")

    # Test different GloVe dimensions
    print("\n[2/2] Evaluating models...")
    print("\n" + "=" * 70)
    print("GLOVE MODEL COMPARISON")
    print("=" * 70)

    loader = Word2VecLoader()
    results = []

    for dim in [50, 100, 200]:
        print(f"\n  Loading GloVe-{dim}...")
        try:
            model = loader.load_glove(dim)

            print(f"  Evaluating...")
            t0 = time.time()
            result = evaluate_antonym_detection(model, antonym_pairs)
            eval_time = time.time() - t0

            result['model'] = f'glove-{dim}'
            result['dimension'] = dim
            result['eval_time'] = eval_time
            results.append(result)

            print(f"    Valid pairs: {result['n_valid_pairs']}")
            print(f"    Hits@1:  {result['hits@1']*100:.1f}%")
            print(f"    Hits@10: {result['hits@10']*100:.1f}%")
            print(f"    MRR:     {result['mrr']:.3f}")
            print(f"    Time:    {eval_time:.1f}s")

        except Exception as e:
            print(f"    Error: {e}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n  Model       | Dim | Pairs | Hits@1 | Hits@10 |  MRR  ")
    print("  " + "-" * 55)

    for r in results:
        print(f"  {r['model']:12s} | {r['dimension']:3d} | {r['n_valid_pairs']:5d} | "
              f"{r['hits@1']*100:5.1f}% | {r['hits@10']*100:6.1f}% | {r['mrr']:.3f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if len(results) >= 2:
        best = max(results, key=lambda x: x['mrr'])
        print(f"\n  Best model: {best['model']} (MRR={best['mrr']:.3f})")

        # Dimension vs performance
        print("\n  Dimension vs Performance:")
        for r in sorted(results, key=lambda x: x['dimension']):
            bar = "█" * int(r['mrr'] * 50)
            print(f"    {r['dimension']:3d}d: {bar} {r['mrr']:.3f}")

    return results


if __name__ == "__main__":
    run_model_comparison()
