"""
High-performance antonym analysis using vectorized operations.

Key optimizations:
1. Precompute all projections using matrix multiplication
2. Use vectorized operations instead of Python loops
3. Efficient antonym search using precomputed projections
4. Memory-efficient chunked processing for large vocabularies
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from numpy.linalg import norm, svd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time

from antonym_loader import extract_antonym_pairs, group_antonyms_by_word
from word2vec_loader import Word2VecLoader


def is_valid_word(word: str) -> bool:
    """Filter noise tokens."""
    return (word.isalpha() and word == word.lower() and
            3 <= len(word) <= 15 and not word.endswith('bb'))


@dataclass
class AntonymResult:
    word: str
    score: float
    relatedness: float


class FastAntonymAnalyzer:
    """
    Ultra-fast antonym analyzer using precomputed matrix operations.

    Algorithm:
    1. Build antonym direction matrix D (n_pairs x dim)
    2. Extract word vectors as matrix V (vocab x dim)
    3. Compute all projections at once: P = V @ D.T (vocab x n_pairs)
    4. For antonym search: find words with opposite projection patterns

    Complexity:
    - Setup: O(vocab * dim * n_pairs) for projection matrix
    - Query: O(vocab) per word (vectorized)
    """

    def __init__(self, model, antonym_pairs: List[Tuple[str, str]],
                 n_components: int = 100, vocab_limit: int = 50000):
        """
        Args:
            model: Gensim KeyedVectors
            antonym_pairs: List of (word1, word2) pairs
            n_components: Number of principal antonym directions to use
            vocab_limit: Max vocabulary size for efficiency
        """
        self.model = model
        self.dim = model.vector_size

        print("Building fast antonym analyzer...")
        t0 = time.time()

        # 1. Build antonym direction matrix
        directions = []
        self.pair_labels = []
        for w1, w2 in antonym_pairs:
            if w1 in model and w2 in model:
                d = model[w1] - model[w2]
                d_norm = norm(d)
                if d_norm > 1e-6:
                    directions.append(d / d_norm)
                    self.pair_labels.append((w1, w2))

        D_full = np.array(directions, dtype=np.float32)  # (n_pairs, dim)
        print(f"  {len(directions)} antonym directions")

        # 2. SVD to get principal directions
        print(f"  Computing {n_components} principal directions...")
        U, S, Vh = svd(D_full, full_matrices=False)
        self.D = Vh[:n_components].astype(np.float32)  # (n_components, dim)
        self.singular_values = S[:n_components]
        print(f"  Top singular values: {S[:5]}")

        # 3. Build vocabulary index (valid words only)
        print(f"  Building vocabulary index...")
        self.words = []
        self.word_to_idx = {}
        for word in model.key_to_index:
            if len(self.words) >= vocab_limit:
                break
            if is_valid_word(word):
                self.word_to_idx[word] = len(self.words)
                self.words.append(word)

        print(f"  {len(self.words)} words in index")

        # 4. Extract word vectors as matrix
        print(f"  Extracting word vectors...")
        self.V = np.array([model[w] for w in self.words], dtype=np.float32)  # (vocab, dim)

        # 5. Precompute all projections: P = V @ D.T
        print(f"  Precomputing projections...")
        self.P = self.V @ self.D.T  # (vocab, n_components)

        # 6. Precompute norms for cosine similarity
        self.V_norms = norm(self.V, axis=1, keepdims=True)  # (vocab, 1)
        self.P_norms = norm(self.P, axis=1, keepdims=True) + 1e-10  # (vocab, 1)

        # 7. Normalize projections for fast cosine computation
        self.P_normalized = self.P / self.P_norms  # (vocab, n_components)

        t1 = time.time()
        print(f"  Setup complete in {t1-t0:.2f}s")

    def find_antonyms(self, word: str, top_n: int = 10,
                      min_relatedness: float = 0.1) -> List[AntonymResult]:
        """
        Find antonyms using precomputed projections.

        The key insight: if (w, c) is an antonym pair, then normalize(w - c) aligns
        with some known antonym direction. Using projections:
            D @ normalize(w - c) = (D @ w - D @ c) / ||w - c||
                                 = (P[w] - P[c]) / ||w - c||

        Args:
            word: Query word
            top_n: Number of results
            min_relatedness: Minimum cosine similarity in original space
        """
        if word not in self.word_to_idx:
            return []

        idx = self.word_to_idx[word]
        v_query = self.V[idx]  # (dim,)
        p_query = self.P[idx]  # (n_components,)

        # Compute difference vectors: query - all
        diff = v_query - self.V  # (vocab, dim)

        # Compute norms of differences
        diff_norms = norm(diff, axis=1, keepdims=True) + 1e-10  # (vocab, 1)

        # Projection difference: P[query] - P[all]
        proj_diff = p_query - self.P  # (vocab, n_components)

        # Normalized projection difference = D @ normalize(diff)
        proj_diff_normalized = proj_diff / diff_norms  # (vocab, n_components)

        # Score: max alignment with any antonym direction
        scores = np.max(np.abs(proj_diff_normalized), axis=1)  # (vocab,)

        # Compute relatedness in original space (for filtering)
        v_query_normalized = v_query / (self.V_norms[idx] + 1e-10)
        relatedness = (self.V / (self.V_norms + 1e-10)) @ v_query_normalized  # (vocab,)

        # Apply relatedness filter (antonyms should be related concepts)
        valid_mask = (relatedness >= min_relatedness) & (np.arange(len(self.words)) != idx)
        scores[~valid_mask] = -np.inf

        # Get top results
        top_indices = np.argsort(scores)[::-1][:top_n]

        results = []
        for i in top_indices:
            if scores[i] == -np.inf:
                break
            results.append(AntonymResult(
                word=self.words[i],
                score=float(scores[i]),
                relatedness=float(relatedness[i])
            ))

        return results

    def find_antonyms_bidirectional(self, word: str, top_n: int = 10,
                                     min_relatedness: float = 0.1) -> List[AntonymResult]:
        """
        Find antonyms with bidirectional verification.
        A is antonym of B only if B is also antonym of A.
        """
        forward = self.find_antonyms(word, top_n=top_n*3, min_relatedness=min_relatedness)

        results = []
        for r in forward:
            # Check reverse
            reverse = self.find_antonyms(r.word, top_n=top_n*3, min_relatedness=min_relatedness)
            reverse_words = {rr.word for rr in reverse}

            if word in reverse_words:
                # Bidirectional confirmed
                reverse_score = next((rr.score for rr in reverse if rr.word == word), 0)
                results.append(AntonymResult(
                    word=r.word,
                    score=min(r.score, reverse_score),  # Conservative score
                    relatedness=r.relatedness
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_n]

    def find_neutral_words(self, top_n: int = 50) -> List[Tuple[str, float]]:
        """
        Find words with low projection magnitude (neutral on all antonym axes).
        Uses precomputed projection norms.
        """
        neutrality = self.P_norms.flatten()  # Lower = more neutral
        top_indices = np.argsort(neutrality)[:top_n]

        return [(self.words[i], float(neutrality[i])) for i in top_indices]

    def batch_find_antonyms(self, words: List[str], top_n: int = 5) -> Dict[str, List[AntonymResult]]:
        """
        Find antonyms for multiple words efficiently.
        Uses batched matrix operations.
        """
        # Get indices for valid words
        valid_words = [(w, self.word_to_idx[w]) for w in words if w in self.word_to_idx]
        if not valid_words:
            return {}

        word_list, indices = zip(*valid_words)
        indices = list(indices)

        # Get relatedness
        V_query = self.V[indices] / (self.V_norms[indices] + 1e-10)
        all_relatedness = V_query @ (self.V / (self.V_norms + 1e-10)).T  # (batch, vocab)

        results = {}
        for i, (word, idx) in enumerate(zip(word_list, indices)):
            v_query = self.V[idx]
            p_query = self.P[idx]

            # Difference vectors
            diff = v_query - self.V  # (vocab, dim)
            diff_norms = norm(diff, axis=1, keepdims=True) + 1e-10

            # Normalized projection difference
            proj_diff = (p_query - self.P) / diff_norms  # (vocab, n_components)

            # Score: max alignment
            scores = np.max(np.abs(proj_diff), axis=1)
            relatedness = all_relatedness[i]

            # Mask invalid
            scores[idx] = -np.inf
            scores[relatedness < 0.1] = -np.inf

            top_indices = np.argsort(scores)[::-1][:top_n]
            results[word] = [
                AntonymResult(self.words[j], float(scores[j]), float(relatedness[j]))
                for j in top_indices if scores[j] > -np.inf
            ]

        return results


def benchmark():
    """Benchmark the fast analyzer."""
    print("=" * 70)
    print("FAST ANTONYM ANALYZER BENCHMARK")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    t0 = time.time()
    antonym_pairs = extract_antonym_pairs()
    known_antonyms = group_antonyms_by_word()
    loader = Word2VecLoader()
    model = loader.load_glove(100)
    print(f"  Data loaded in {time.time()-t0:.2f}s")

    # Build analyzer
    print("\n[2/3] Building analyzer...")
    t0 = time.time()
    analyzer = FastAntonymAnalyzer(model, antonym_pairs, n_components=100, vocab_limit=100000)
    build_time = time.time() - t0
    print(f"  Analyzer built in {build_time:.2f}s")

    # Benchmark queries
    print("\n[3/3] Benchmarking queries...")

    test_words = ["good", "happy", "love", "hot", "big", "fast", "young", "rich",
                  "strong", "open", "high", "long", "peace", "truth", "beauty"]

    # Single queries
    t0 = time.time()
    for word in test_words:
        _ = analyzer.find_antonyms(word, top_n=10)
    single_time = time.time() - t0
    print(f"  Single queries ({len(test_words)} words): {single_time*1000:.2f}ms "
          f"({single_time/len(test_words)*1000:.2f}ms/word)")

    # Batch queries
    t0 = time.time()
    _ = analyzer.batch_find_antonyms(test_words, top_n=10)
    batch_time = time.time() - t0
    print(f"  Batch query ({len(test_words)} words): {batch_time*1000:.2f}ms "
          f"({batch_time/len(test_words)*1000:.2f}ms/word)")

    # Bidirectional queries
    t0 = time.time()
    for word in test_words[:5]:
        _ = analyzer.find_antonyms_bidirectional(word, top_n=10)
    bidir_time = time.time() - t0
    print(f"  Bidirectional queries (5 words): {bidir_time*1000:.2f}ms "
          f"({bidir_time/5*1000:.2f}ms/word)")

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
        found_words = [r.word for r in results]
        if w2 in found_words:
            rank = found_words.index(w2) + 1
            print(f"  {w1:10} -> {w2:10}: ✓ Rank {rank}")
        else:
            top3 = ", ".join(found_words[:3])
            print(f"  {w1:10} -> {w2:10}: ✗ (top: {top3})")

    print("\n[Bidirectional Verification]")
    for w1, w2 in test_pairs[:5]:
        results = analyzer.find_antonyms_bidirectional(w1, top_n=10)
        found_words = [r.word for r in results]
        if w2 in found_words:
            r = next(r for r in results if r.word == w2)
            print(f"  {w1:10} <-> {w2:10}: ✓ score={r.score:.3f}")
        else:
            top3 = ", ".join(found_words[:3]) if found_words else "none"
            print(f"  {w1:10} <-> {w2:10}: ✗ (top: {top3})")

    print("\n[Neutral Words]")
    neutrals = analyzer.find_neutral_words(top_n=20)
    for word, score in neutrals[:10]:
        print(f"  {word:20}: {score:.4f}")

    print("\n[Abstract Concepts]")
    abstract_words = ["technology", "democracy", "philosophy", "nature",
                      "culture", "power", "freedom", "wisdom", "chaos"]
    batch_results = analyzer.batch_find_antonyms(abstract_words, top_n=3)

    for word in abstract_words:
        if word in batch_results and batch_results[word]:
            antonyms = ", ".join([f"{r.word}({r.score:.2f})" for r in batch_results[word]])
            print(f"  {word:15}: {antonyms}")


if __name__ == "__main__":
    benchmark()
