"""
High-performance antonym analysis v2 - uses ALL antonym directions.

Key insight from experiments:
- SVD reduces variance but loses specific antonym relationships
- Using all 2352 antonym directions gives better accuracy
- Still use matrix operations for efficiency
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from numpy.linalg import norm
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time

from antonym_loader import extract_antonym_pairs
from word2vec_loader import Word2VecLoader


def is_valid_word(word: str) -> bool:
    return (word.isalpha() and word == word.lower() and
            3 <= len(word) <= 15 and not word.endswith('bb'))


@dataclass
class AntonymResult:
    word: str
    score: float
    relatedness: float


class FastAntonymV2:
    """
    Fast antonym analyzer using ALL antonym directions (no SVD reduction).

    Key optimizations:
    1. Precompute P = V @ D.T for all words (vocab x n_directions)
    2. For query, compute scores using vectorized operations
    3. Use chunked processing to manage memory
    """

    def __init__(self, model, antonym_pairs: List[Tuple[str, str]],
                 vocab_limit: int = 50000):
        self.model = model
        self.dim = model.vector_size

        print("Building FastAntonymV2 (full directions)...")
        t0 = time.time()

        # Build antonym direction matrix (all directions, not SVD-reduced)
        directions = []
        self.pair_labels = []
        for w1, w2 in antonym_pairs:
            if w1 in model and w2 in model:
                d = model[w1] - model[w2]
                d_norm = norm(d)
                if d_norm > 1e-6:
                    directions.append(d / d_norm)
                    self.pair_labels.append((w1, w2))

        self.D = np.array(directions, dtype=np.float32)  # (n_directions, dim)
        self.n_directions = len(directions)
        print(f"  {self.n_directions} antonym directions")

        # Build vocabulary
        self.words = []
        self.word_to_idx = {}
        for word in model.key_to_index:
            if len(self.words) >= vocab_limit:
                break
            if is_valid_word(word):
                self.word_to_idx[word] = len(self.words)
                self.words.append(word)
        print(f"  {len(self.words)} words in vocabulary")

        # Extract word vectors
        self.V = np.array([model[w] for w in self.words], dtype=np.float32)
        self.V_norms = norm(self.V, axis=1, keepdims=True) + 1e-10

        # Precompute projections: P = V @ D.T (vocab x n_directions)
        print(f"  Precomputing projections ({len(self.words)} x {self.n_directions})...")
        self.P = self.V @ self.D.T

        # Precompute normalized vectors for relatedness
        self.V_normalized = self.V / self.V_norms

        print(f"  Setup complete in {time.time()-t0:.2f}s")
        print(f"  Memory: P matrix = {self.P.nbytes / 1e6:.1f} MB")

    def find_antonyms(self, word: str, top_n: int = 10,
                      min_relatedness: float = 0.1,
                      chunk_size: int = 5000) -> List[AntonymResult]:
        """Find antonyms with chunked processing for better cache utilization."""
        if word not in self.word_to_idx:
            return []

        idx = self.word_to_idx[word]
        v_query = self.V[idx]
        p_query = self.P[idx]
        v_query_norm_sq = np.dot(v_query, v_query)

        # Relatedness filter (fast, uses precomputed normalized vectors)
        relatedness = self.V_normalized @ self.V_normalized[idx]

        valid_mask = (relatedness >= min_relatedness)
        valid_mask[idx] = False
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return []

        # Process in chunks to improve cache locality
        all_scores = np.empty(len(valid_indices), dtype=np.float32)

        for start in range(0, len(valid_indices), chunk_size):
            end = min(start + chunk_size, len(valid_indices))
            chunk_indices = valid_indices[start:end]

            V_chunk = self.V[chunk_indices]
            P_chunk = self.P[chunk_indices]

            # ||query - cand||
            dot_products = V_chunk @ v_query
            V_chunk_norms_sq = np.einsum('ij,ij->i', V_chunk, V_chunk)
            diff_norms = np.sqrt(np.maximum(
                v_query_norm_sq + V_chunk_norms_sq - 2 * dot_products, 1e-10
            ))

            # Score: max(|P[query] - P[cand]|) / ||diff||
            # Compute without creating full intermediate array
            max_proj_diff = np.max(np.abs(P_chunk - p_query), axis=1)
            all_scores[start:end] = max_proj_diff / diff_norms

        # Top results
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
        """Find words with minimal projection magnitude."""
        neutrality = norm(self.P, axis=1)
        top_indices = np.argsort(neutrality)[:top_n]
        return [(self.words[i], float(neutrality[i])) for i in top_indices]


def main():
    print("=" * 70)
    print("FAST ANTONYM V2 (Full Directions)")
    print("=" * 70)

    # Load
    print("\n[1/3] Loading data...")
    t0 = time.time()
    antonym_pairs = extract_antonym_pairs()
    loader = Word2VecLoader()
    model = loader.load_glove(100)
    print(f"  Loaded in {time.time()-t0:.2f}s")

    # Build
    print("\n[2/3] Building analyzer...")
    analyzer = FastAntonymV2(model, antonym_pairs, vocab_limit=50000)

    # Benchmark
    print("\n[3/3] Benchmarking...")

    test_words = ["good", "happy", "love", "hot", "big", "fast", "peace"]

    t0 = time.time()
    for word in test_words:
        _ = analyzer.find_antonyms(word, top_n=10)
    print(f"  Query time: {(time.time()-t0)/len(test_words)*1000:.1f}ms/word")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n[Known Antonym Pairs]")
    test_pairs = [
        ("good", "bad"), ("happy", "sad"), ("love", "hate"),
        ("hot", "cold"), ("light", "dark"), ("big", "small"),
        ("fast", "slow"), ("young", "old"), ("peace", "war")
    ]

    for w1, w2 in test_pairs:
        results = analyzer.find_antonyms(w1, top_n=10)
        found = [r.word for r in results]
        if w2 in found:
            rank = found.index(w2) + 1
            score = results[rank-1].score
            print(f"  {w1:10} -> {w2:10}: ✓ Rank {rank}, score={score:.3f}")
        else:
            top3 = ", ".join(found[:3])
            print(f"  {w1:10} -> {w2:10}: ✗ (top: {top3})")

    print("\n[Bidirectional Verification]")
    for w1, w2 in test_pairs[:5]:
        results = analyzer.find_antonyms_bidirectional(w1, top_n=10)
        found = [r.word for r in results]
        if w2 in found:
            r = next(r for r in results if r.word == w2)
            print(f"  {w1:10} <-> {w2:10}: ✓ score={r.score:.3f}")
        else:
            top3 = ", ".join(found[:3]) if found else "none"
            print(f"  {w1:10} <-> {w2:10}: ✗ (top: {top3})")

    print("\n[Abstract Concepts]")
    abstract = ["democracy", "freedom", "truth", "beauty", "chaos", "power"]
    for word in abstract:
        results = analyzer.find_antonyms(word, top_n=5)
        if results:
            ant_str = ", ".join([f"{r.word}({r.score:.2f})" for r in results[:3]])
            print(f"  {word:12}: {ant_str}")


if __name__ == "__main__":
    main()
