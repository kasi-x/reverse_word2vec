"""
Export English antonym analysis results in JSON format for cross-lingual comparison.

Runs the same metrics as japanese_analysis.py against the English GloVe model.
"""
import sys
import os
import json
import time
import random

import numpy as np
from numpy.linalg import norm

sys.path.insert(0, os.path.dirname(__file__))

from word2vec_loader import Word2VecLoader
from antonym_loader import extract_antonym_pairs
from fast_antonym_v2 import FastAntonymV2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def compute_svd_analysis(D: np.ndarray) -> dict:
    """Compute SVD on the antonym direction matrix."""
    print("  Computing SVD analysis...")
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    variance = s ** 2
    total_var = variance.sum()
    cumulative = np.cumsum(variance) / total_var

    dims_90 = int(np.searchsorted(cumulative, 0.9)) + 1
    dims_95 = int(np.searchsorted(cumulative, 0.95)) + 1
    dims_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    print(f"  SVD: {dims_90} dims for 90%, {dims_95} for 95%, {dims_99} for 99%")

    return {
        "singular_values": s.tolist(),
        "cumulative_variance": cumulative.tolist(),
        "dims_for_90_pct": dims_90,
        "dims_for_95_pct": dims_95,
        "dims_for_99_pct": dims_99,
        "total_directions": len(D),
    }


def evaluate_known_pairs(analyzer, test_pairs, top_n=10):
    """Evaluate performance on known antonym pairs."""
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    tested = 0
    pair_results = []

    for w1, w2 in test_pairs:
        if w1 not in analyzer.word_to_idx or w2 not in analyzer.word_to_idx:
            continue

        results = analyzer.find_antonyms(w1, top_n=top_n)
        found_words = [r.word for r in results]
        tested += 1

        rank = None
        score = 0.0
        if w2 in found_words:
            rank = found_words.index(w2) + 1
            score = results[rank - 1].score
            if rank == 1:
                hits_at_1 += 1
            if rank <= 5:
                hits_at_5 += 1
            if rank <= 10:
                hits_at_10 += 1

        pair_results.append({
            "word1": w1, "word2": w2,
            "rank": rank, "score": score,
            "top3": [r.word for r in results[:3]],
        })

    return {
        "tested": tested,
        "hits_at_1": hits_at_1,
        "hits_at_5": hits_at_5,
        "hits_at_10": hits_at_10,
        "accuracy_at_1": hits_at_1 / tested if tested > 0 else 0,
        "accuracy_at_5": hits_at_5 / tested if tested > 0 else 0,
        "accuracy_at_10": hits_at_10 / tested if tested > 0 else 0,
        "pair_results": pair_results,
    }


def main():
    print("=" * 70)
    print("ENGLISH ANTONYM ANALYSIS (Export for Comparison)")
    print("=" * 70)

    # Load
    print("\n[1/4] Loading data...")
    t0 = time.time()
    antonym_pairs = extract_antonym_pairs()
    loader = Word2VecLoader()
    model = loader.load_glove(100)
    print(f"  Loaded in {time.time()-t0:.2f}s")

    # Filter valid pairs
    valid_pairs = []
    for w1, w2 in antonym_pairs:
        if w1 in model and w2 in model:
            valid_pairs.append((w1, w2))
    print(f"  Valid pairs: {len(valid_pairs)} / {len(antonym_pairs)}")

    # Build analyzer
    print("\n[2/4] Building analyzer...")
    analyzer = FastAntonymV2(model, antonym_pairs, vocab_limit=50000)

    # SVD analysis
    print("\n[3/4] Running analysis...")
    svd_results = compute_svd_analysis(analyzer.D)

    # Benchmark
    test_words = ["good", "happy", "love", "hot", "big", "fast", "peace"]
    t0 = time.time()
    for word in test_words:
        _ = analyzer.find_antonyms(word, top_n=10)
    query_time_ms = (time.time() - t0) / len(test_words) * 1000
    print(f"  Query time: {query_time_ms:.1f}ms/word")

    # Known pairs evaluation
    print("\n[4/4] Evaluating...")
    test_pairs = [
        ("good", "bad"), ("happy", "sad"), ("love", "hate"),
        ("hot", "cold"), ("light", "dark"), ("big", "small"),
        ("fast", "slow"), ("young", "old"), ("peace", "war"),
        ("begin", "end"), ("import", "export"), ("increase", "decrease"),
        ("male", "female"), ("strong", "weak"), ("rich", "poor"),
        ("buy", "sell"), ("win", "lose"), ("open", "close"),
        ("long", "short"), ("deep", "shallow"),
    ]

    print("\n[Known Antonym Pairs]")
    for w1, w2 in test_pairs:
        if w1 not in analyzer.word_to_idx:
            continue
        results = analyzer.find_antonyms(w1, top_n=10)
        found = [r.word for r in results]
        if w2 in found:
            rank = found.index(w2) + 1
            score = results[rank - 1].score
            print(f"  {w1:10} -> {w2:10}: ✓ Rank {rank}, score={score:.3f}")
        else:
            top3 = ", ".join(found[:3])
            print(f"  {w1:10} -> {w2:10}: ✗ (top: {top3})")

    eval_results = evaluate_known_pairs(analyzer, test_pairs)
    print(f"\n  Hits@1:  {eval_results['hits_at_1']}/{eval_results['tested']} "
          f"({eval_results['accuracy_at_1']:.1%})")
    print(f"  Hits@5:  {eval_results['hits_at_5']}/{eval_results['tested']} "
          f"({eval_results['accuracy_at_5']:.1%})")
    print(f"  Hits@10: {eval_results['hits_at_10']}/{eval_results['tested']} "
          f"({eval_results['accuracy_at_10']:.1%})")

    # Bidirectional
    print("\n[Bidirectional Verification]")
    bidir_pairs = [
        ("good", "bad"), ("happy", "sad"), ("love", "hate"),
        ("hot", "cold"), ("big", "small"),
    ]
    bidir_results = []
    for w1, w2 in bidir_pairs:
        results = analyzer.find_antonyms_bidirectional(w1, top_n=10)
        found = [r.word for r in results]
        if w2 in found:
            r = next(r for r in results if r.word == w2)
            print(f"  {w1:10} <-> {w2:10}: ✓ score={r.score:.3f}")
            bidir_results.append({"word1": w1, "word2": w2, "score": r.score, "found": True})
        else:
            top3 = ", ".join(found[:3]) if found else "none"
            print(f"  {w1:10} <-> {w2:10}: ✗ (top: {top3})")
            bidir_results.append({"word1": w1, "word2": w2, "score": 0, "found": False})

    # Neutral words
    print("\n[Neutral Words]")
    neutral = analyzer.find_neutral_words(top_n=20)
    for word, score in neutral:
        print(f"  {word:15} (neutrality={score:.3f})")

    # Abstract concepts
    print("\n[Abstract Concepts]")
    abstract_words = ["democracy", "freedom", "truth", "beauty", "chaos", "power", "peace", "justice"]
    abstract_results = []
    for word in abstract_words:
        results = analyzer.find_antonyms(word, top_n=5)
        if results:
            ant_str = ", ".join([f"{r.word}({r.score:.2f})" for r in results[:3]])
            print(f"  {word:12}: {ant_str}")
            abstract_results.append({
                "word": word,
                "antonyms": [{"word": r.word, "score": r.score} for r in results[:5]],
            })

    # Comprehensive evaluation
    print("\n[Comprehensive Evaluation]")
    random.seed(42)
    sample_size = min(200, len(valid_pairs))
    sample_pairs = random.sample(valid_pairs, sample_size)
    comprehensive_eval = evaluate_known_pairs(analyzer, sample_pairs)
    print(f"  Sample size: {comprehensive_eval['tested']}")
    print(f"  Hits@1:  {comprehensive_eval['accuracy_at_1']:.1%}")
    print(f"  Hits@5:  {comprehensive_eval['accuracy_at_5']:.1%}")
    print(f"  Hits@10: {comprehensive_eval['accuracy_at_10']:.1%}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        "language": "en",
        "model": "glove-wiki-gigaword-100",
        "model_dimensions": model.vector_size,
        "model_vocab_size": len(model),
        "total_antonym_pairs": len(antonym_pairs),
        "valid_antonym_pairs": len(valid_pairs),
        "analyzer_vocab_size": len(analyzer.words),
        "n_antonym_directions": analyzer.n_directions,
        "query_time_ms": query_time_ms,
        "svd_analysis": svd_results,
        "known_pairs_evaluation": eval_results,
        "comprehensive_evaluation": {
            "sample_size": comprehensive_eval["tested"],
            "accuracy_at_1": comprehensive_eval["accuracy_at_1"],
            "accuracy_at_5": comprehensive_eval["accuracy_at_5"],
            "accuracy_at_10": comprehensive_eval["accuracy_at_10"],
        },
        "bidirectional_results": bidir_results,
        "neutral_words": [{"word": w, "score": s} for w, s in neutral],
        "abstract_concepts": abstract_results,
    }

    output_path = os.path.join(RESULTS_DIR, "english_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
