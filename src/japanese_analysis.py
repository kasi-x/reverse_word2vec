"""
Japanese antonym space analysis using fastText cc.ja.300.

Applies the same algorithm as fast_antonym_v2.py (full antonym directions,
precomputed projections) to Japanese word vectors and EPWING-derived antonym pairs.

Outputs metrics as JSON for cross-lingual comparison.
"""
import sys
import os
import json
import re
import time

import numpy as np
from numpy.linalg import norm
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(__file__))

from word2vec_loader import Word2VecLoader
from japanese_antonym_loader import load_pairs, extract_all_antonym_pairs, save_pairs, normalize_japanese_word

# Regex for valid Japanese words (hiragana, katakana, kanji, prolonged sound mark)
_JA_WORD = re.compile(r"^[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3005\u30FC]+$")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def is_valid_japanese_word(word: str) -> bool:
    """Check if word is a valid Japanese word for analysis."""
    if not _JA_WORD.match(word):
        return False
    if not 1 <= len(word) <= 10:
        return False
    # Reject repetitive symbols like ・・・・ or ーーーー
    if len(set(word)) <= 2 and len(word) > 3:
        return False
    return True


@dataclass
class AntonymResult:
    word: str
    score: float
    relatedness: float


class JapaneseAntonymAnalyzer:
    """
    Antonym analyzer for Japanese word vectors.

    Same algorithm as FastAntonymV2: uses all antonym direction vectors
    with precomputed projection matrix for fast queries.
    """

    def __init__(self, model, antonym_pairs: List[Tuple[str, str]],
                 vocab_limit: int = 50000):
        self.model = model
        self.dim = model.vector_size

        print("Building JapaneseAntonymAnalyzer (full directions)...")
        t0 = time.time()

        # Build antonym direction matrix
        directions = []
        self.pair_labels = []
        for w1, w2 in antonym_pairs:
            nw1 = normalize_japanese_word(w1)
            nw2 = normalize_japanese_word(w2)
            if nw1 in model and nw2 in model:
                d = model[nw1] - model[nw2]
                d_norm = norm(d)
                if d_norm > 1e-6:
                    directions.append(d / d_norm)
                    self.pair_labels.append((nw1, nw2))

        self.D = np.array(directions, dtype=np.float32)
        self.n_directions = len(directions)
        print(f"  {self.n_directions} antonym directions")

        # Build vocabulary (Japanese words only)
        self.words = []
        self.word_to_idx = {}
        for word in model.key_to_index:
            if len(self.words) >= vocab_limit:
                break
            if is_valid_japanese_word(word):
                self.word_to_idx[word] = len(self.words)
                self.words.append(word)
        print(f"  {len(self.words)} words in vocabulary")

        # Extract word vectors
        self.V = np.array([model[w] for w in self.words], dtype=np.float32)
        self.V_norms = norm(self.V, axis=1, keepdims=True) + 1e-10

        # Precompute projections: P = V @ D.T
        print(f"  Precomputing projections ({len(self.words)} x {self.n_directions})...")
        self.P = self.V @ self.D.T

        # Precompute normalized vectors for relatedness
        self.V_normalized = self.V / self.V_norms

        print(f"  Setup complete in {time.time()-t0:.2f}s")
        print(f"  Memory: P matrix = {self.P.nbytes / 1e6:.1f} MB")

    def find_antonyms(self, word: str, top_n: int = 10,
                      min_relatedness: float = 0.1,
                      chunk_size: int = 5000) -> List[AntonymResult]:
        """Find antonyms with chunked processing."""
        if word not in self.word_to_idx:
            return []

        idx = self.word_to_idx[word]
        v_query = self.V[idx]
        p_query = self.P[idx]
        v_query_norm_sq = np.dot(v_query, v_query)

        relatedness = self.V_normalized @ self.V_normalized[idx]
        valid_mask = (relatedness >= min_relatedness)
        valid_mask[idx] = False
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return []

        all_scores = np.empty(len(valid_indices), dtype=np.float32)

        for start in range(0, len(valid_indices), chunk_size):
            end = min(start + chunk_size, len(valid_indices))
            chunk_indices = valid_indices[start:end]

            V_chunk = self.V[chunk_indices]
            P_chunk = self.P[chunk_indices]

            dot_products = V_chunk @ v_query
            V_chunk_norms_sq = np.einsum('ij,ij->i', V_chunk, V_chunk)
            diff_norms = np.sqrt(np.maximum(
                v_query_norm_sq + V_chunk_norms_sq - 2 * dot_products, 1e-10
            ))

            max_proj_diff = np.max(np.abs(P_chunk - p_query), axis=1)
            all_scores[start:end] = max_proj_diff / diff_norms

        top_k = min(top_n, len(valid_indices))
        top_local = np.argpartition(all_scores, -top_k)[-top_k:]
        top_local = top_local[np.argsort(all_scores[top_local])[::-1]]

        return [
            AntonymResult(
                self.words[valid_indices[i]],
                float(all_scores[i]),
                float(relatedness[valid_indices[i]])
            )
            for i in top_local
        ]

    def find_antonyms_bidirectional(self, word: str, top_n: int = 10,
                                     min_relatedness: float = 0.1) -> List[AntonymResult]:
        """Find antonyms with bidirectional verification."""
        forward = self.find_antonyms(word, top_n=top_n*3, min_relatedness=min_relatedness)

        results = []
        for r in forward:
            reverse = self.find_antonyms(r.word, top_n=top_n*3, min_relatedness=min_relatedness)
            reverse_words = {rr.word for rr in reverse}

            if word in reverse_words:
                reverse_score = next((rr.score for rr in reverse if rr.word == word), 0)
                results.append(AntonymResult(
                    r.word,
                    min(r.score, reverse_score),
                    r.relatedness
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_n]

    def find_neutral_words(self, top_n: int = 50) -> List[Tuple[str, float]]:
        """Find words with minimal projection magnitude (neutral words)."""
        neutrality = norm(self.P, axis=1)
        top_indices = np.argsort(neutrality)[:top_n]
        return [(self.words[i], float(neutrality[i])) for i in top_indices]

    def compute_svd_analysis(self) -> dict:
        """Compute SVD on the antonym direction matrix for dimensionality analysis."""
        print("\n  Computing SVD analysis on antonym directions...")
        U, s, Vt = np.linalg.svd(self.D, full_matrices=False)
        variance = s ** 2
        total_var = variance.sum()
        cumulative = np.cumsum(variance) / total_var

        # Find number of components for various thresholds
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
            "total_directions": self.n_directions,
        }

    def evaluate_known_pairs(self, test_pairs: List[Tuple[str, str]],
                              top_n: int = 10) -> dict:
        """Evaluate performance on known antonym pairs."""
        hits_at_1 = 0
        hits_at_5 = 0
        hits_at_10 = 0
        tested = 0
        pair_results = []

        for w1, w2 in test_pairs:
            nw1 = normalize_japanese_word(w1)
            nw2 = normalize_japanese_word(w2)
            if nw1 not in self.word_to_idx or nw2 not in self.word_to_idx:
                continue

            results = self.find_antonyms(nw1, top_n=top_n)
            found_words = [r.word for r in results]
            tested += 1

            rank = None
            score = 0.0
            if nw2 in found_words:
                rank = found_words.index(nw2) + 1
                score = results[rank - 1].score
                if rank == 1:
                    hits_at_1 += 1
                if rank <= 5:
                    hits_at_5 += 1
                if rank <= 10:
                    hits_at_10 += 1

            pair_results.append({
                "word1": nw1, "word2": nw2,
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
    print("JAPANESE ANTONYM SPACE ANALYSIS")
    print("=" * 70)

    # Load antonym pairs
    print("\n[1/5] Loading antonym pairs...")
    pairs_path = os.path.join(DATA_DIR, "japanese_antonym_pairs.json")
    if os.path.exists(pairs_path):
        all_pairs = load_pairs(pairs_path)
        print(f"  Loaded {len(all_pairs)} pairs from cache")
    else:
        all_pairs = extract_all_antonym_pairs()
        save_pairs(all_pairs, pairs_path)

    # Load Japanese word vector model
    print("\n[2/5] Loading Japanese word vector model...")
    loader = Word2VecLoader()
    model = loader.load_ja_dict()

    # Filter pairs to those valid in model
    valid_pairs = []
    for w1, w2 in all_pairs:
        nw1 = normalize_japanese_word(w1)
        nw2 = normalize_japanese_word(w2)
        if nw1 in model and nw2 in model and nw1 != nw2:
            valid_pairs.append((nw1, nw2))
    # Deduplicate
    valid_pairs = list(set(tuple(sorted(p)) for p in valid_pairs))
    print(f"  Valid pairs in model: {len(valid_pairs)} / {len(all_pairs)}")

    # Build analyzer
    print("\n[3/5] Building analyzer...")
    analyzer = JapaneseAntonymAnalyzer(model, valid_pairs, vocab_limit=50000)

    # SVD analysis
    print("\n[4/5] Running analysis...")
    svd_results = analyzer.compute_svd_analysis()

    # Benchmark
    test_words = ["大きい", "明るい", "強い", "買う", "男"]
    t0 = time.time()
    for word in test_words:
        _ = analyzer.find_antonyms(word, top_n=10)
    query_time_ms = (time.time() - t0) / len(test_words) * 1000
    print(f"  Query time: {query_time_ms:.1f}ms/word")

    # Results
    print("\n[5/5] Evaluating results...")
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Known antonym pairs test
    print("\n[Known Antonym Pairs]")
    test_pairs = [
        ("大きい", "小さい"), ("明るい", "暗い"), ("上", "下"),
        ("左", "右"), ("男", "女"), ("善", "悪"),
        ("長い", "短い"), ("高い", "低い"), ("多い", "少ない"),
        ("強い", "弱い"), ("買う", "売る"), ("入る", "出る"),
        ("暑い", "寒い"), ("始める", "終える"), ("増える", "減る"),
        ("勝つ", "負ける"), ("生", "死"), ("天", "地"),
        ("朝", "夜"), ("表", "裏"),
    ]

    for w1, w2 in test_pairs:
        if w1 not in analyzer.word_to_idx:
            continue
        results = analyzer.find_antonyms(w1, top_n=10)
        found = [r.word for r in results]
        if w2 in found:
            rank = found.index(w2) + 1
            score = results[rank - 1].score
            print(f"  {w1:6} -> {w2:6}: ✓ Rank {rank}, score={score:.3f}")
        else:
            top3 = ", ".join(found[:3])
            print(f"  {w1:6} -> {w2:6}: ✗ (top: {top3})")

    eval_results = analyzer.evaluate_known_pairs(test_pairs)
    print(f"\n  Hits@1:  {eval_results['hits_at_1']}/{eval_results['tested']} "
          f"({eval_results['accuracy_at_1']:.1%})")
    print(f"  Hits@5:  {eval_results['hits_at_5']}/{eval_results['tested']} "
          f"({eval_results['accuracy_at_5']:.1%})")
    print(f"  Hits@10: {eval_results['hits_at_10']}/{eval_results['tested']} "
          f"({eval_results['accuracy_at_10']:.1%})")

    # Bidirectional verification
    print("\n[Bidirectional Verification]")
    bidir_pairs = [
        ("大きい", "小さい"), ("明るい", "暗い"), ("強い", "弱い"),
        ("買う", "売る"), ("男", "女"),
    ]
    bidir_results = []
    for w1, w2 in bidir_pairs:
        results = analyzer.find_antonyms_bidirectional(w1, top_n=10)
        found = [r.word for r in results]
        if w2 in found:
            r = next(r for r in results if r.word == w2)
            print(f"  {w1:6} <-> {w2:6}: ✓ score={r.score:.3f}")
            bidir_results.append({"word1": w1, "word2": w2, "score": r.score, "found": True})
        else:
            top3 = ", ".join(found[:3]) if found else "none"
            print(f"  {w1:6} <-> {w2:6}: ✗ (top: {top3})")
            bidir_results.append({"word1": w1, "word2": w2, "score": 0, "found": False})

    # Neutral words
    print("\n[Neutral Words (near zero in antonym space)]")
    neutral = analyzer.find_neutral_words(top_n=20)
    for word, score in neutral:
        print(f"  {word:10} (neutrality={score:.3f})")

    # Abstract concept antonyms
    print("\n[Abstract Concept Antonyms]")
    abstract_words = ["平和", "美", "知恵", "真実", "自然", "混乱", "自由", "正義"]
    abstract_results = []
    for word in abstract_words:
        results = analyzer.find_antonyms(word, top_n=5)
        if results:
            ant_str = ", ".join([f"{r.word}({r.score:.2f})" for r in results[:3]])
            print(f"  {word:6}: {ant_str}")
            abstract_results.append({
                "word": word,
                "antonyms": [{"word": r.word, "score": r.score} for r in results[:5]],
            })

    # Comprehensive evaluation on all valid pairs (sample)
    print("\n[Comprehensive Evaluation (sample of valid pairs)]")
    import random
    random.seed(42)
    sample_size = min(200, len(valid_pairs))
    sample_pairs = random.sample(valid_pairs, sample_size)
    comprehensive_eval = analyzer.evaluate_known_pairs(sample_pairs)
    print(f"  Sample size: {comprehensive_eval['tested']}")
    print(f"  Hits@1:  {comprehensive_eval['accuracy_at_1']:.1%}")
    print(f"  Hits@5:  {comprehensive_eval['accuracy_at_5']:.1%}")
    print(f"  Hits@10: {comprehensive_eval['accuracy_at_10']:.1%}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        "language": "ja",
        "model": "cc.ja.300",
        "model_dimensions": model.vector_size,
        "model_vocab_size": len(model),
        "total_antonym_pairs": len(all_pairs),
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

    output_path = os.path.join(RESULTS_DIR, "japanese_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
