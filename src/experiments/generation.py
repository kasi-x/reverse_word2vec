"""
Experiment 3: Antonym Generation

Goal: Given any word, generate its "ideal antonym vector" and find
the nearest real words to that vector.

This allows finding antonyms for words that don't have known antonyms
in the dictionary.

Approach:
1. Learn an "antonym transformation" from known antonym pairs
2. Apply it to any word to generate its antonym vector
3. Find nearest neighbors to the generated vector
"""
import sys
sys.path.insert(0, '..')

import numpy as np
from numpy.linalg import norm, lstsq, svd
from typing import List, Tuple, Dict
from dataclasses import dataclass
import time

from antonym_loader import extract_antonym_pairs, group_antonyms_by_word
from word2vec_loader import Word2VecLoader


class AntonymGenerator:
    """
    Learns to generate antonym vectors from known pairs.

    Methods:
    1. Average offset: antonym(w) ≈ w + avg_offset
    2. Linear transformation: antonym(w) ≈ A @ w + b
    3. Mirroring: antonym(w) ≈ 2*centroid - w
    """

    def __init__(self, model, antonym_pairs: List[Tuple[str, str]]):
        self.model = model
        self.dim = model.vector_size

        print("Learning antonym transformations...")

        # Collect valid pairs
        self.pairs = []
        for w1, w2 in antonym_pairs:
            if w1 in model and w2 in model:
                self.pairs.append((w1, w2))

        print(f"  {len(self.pairs)} valid antonym pairs")

        # Learn transformations
        self._learn_average_offset()
        self._learn_linear_transform()
        self._learn_mirror_transform()

    def _learn_average_offset(self):
        """Method 1: Average offset between antonym pairs."""
        offsets = []
        for w1, w2 in self.pairs:
            offset = self.model[w2] - self.model[w1]
            offsets.append(offset)

        self.avg_offset = np.mean(offsets, axis=0)
        print(f"  Average offset learned (norm={norm(self.avg_offset):.3f})")

    def _learn_linear_transform(self):
        """Method 2: Linear transformation A @ w + b ≈ antonym(w)."""
        # Stack source and target vectors
        X = np.array([self.model[w1] for w1, w2 in self.pairs])  # (n, d)
        Y = np.array([self.model[w2] for w1, w2 in self.pairs])  # (n, d)

        # Add bias term
        X_with_bias = np.hstack([X, np.ones((len(X), 1))])  # (n, d+1)

        # Solve least squares: X_with_bias @ W ≈ Y
        # W is (d+1, d)
        W, residuals, rank, s = lstsq(X_with_bias, Y, rcond=None)

        self.linear_A = W[:-1, :].T  # (d, d)
        self.linear_b = W[-1, :]     # (d,)

        # Compute reconstruction error
        Y_pred = X @ self.linear_A.T + self.linear_b
        error = np.mean(norm(Y - Y_pred, axis=1))
        print(f"  Linear transform learned (avg error={error:.3f})")

    def _learn_mirror_transform(self):
        """Method 3: Mirror around centroid of pairs."""
        # Compute centroid of all antonym words
        all_words = set()
        for w1, w2 in self.pairs:
            all_words.add(w1)
            all_words.add(w2)

        vectors = [self.model[w] for w in all_words]
        self.centroid = np.mean(vectors, axis=0)
        print(f"  Mirror centroid computed")

    def generate_offset(self, word: str) -> np.ndarray:
        """Generate antonym using average offset."""
        if word not in self.model:
            return None
        return self.model[word] + self.avg_offset

    def generate_linear(self, word: str) -> np.ndarray:
        """Generate antonym using linear transformation."""
        if word not in self.model:
            return None
        return self.linear_A @ self.model[word] + self.linear_b

    def generate_mirror(self, word: str) -> np.ndarray:
        """Generate antonym by mirroring around centroid."""
        if word not in self.model:
            return None
        return 2 * self.centroid - self.model[word]

    def generate_ensemble(self, word: str) -> np.ndarray:
        """Generate antonym using ensemble of methods."""
        if word not in self.model:
            return None

        v_offset = self.generate_offset(word)
        v_linear = self.generate_linear(word)
        v_mirror = self.generate_mirror(word)

        # Simple average
        return (v_offset + v_linear + v_mirror) / 3

    def find_nearest_words(self, vector: np.ndarray, candidates: List[str],
                           top_n: int = 10, exclude: List[str] = None) -> List[Tuple[str, float]]:
        """Find nearest words to a vector."""
        if vector is None:
            return []

        exclude = set(exclude or [])
        results = []

        vector_norm = norm(vector)
        if vector_norm < 1e-6:
            return []

        for word in candidates:
            if word in exclude or word not in self.model:
                continue

            v = self.model[word]
            similarity = np.dot(vector, v) / (vector_norm * norm(v) + 1e-10)
            results.append((word, float(similarity)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def evaluate_method(self, method: str, test_pairs: List[Tuple[str, str]],
                        candidates: List[str], k: int = 10) -> Dict:
        """
        Evaluate a generation method.

        Returns accuracy metrics for finding true antonyms.
        """
        generate_fn = {
            'offset': self.generate_offset,
            'linear': self.generate_linear,
            'mirror': self.generate_mirror,
            'ensemble': self.generate_ensemble,
        }[method]

        hits_at_1 = 0
        hits_at_k = 0
        reciprocal_ranks = []

        for w1, w2 in test_pairs:
            if w1 not in self.model or w2 not in self.model:
                continue

            generated = generate_fn(w1)
            nearest = self.find_nearest_words(generated, candidates, top_n=k, exclude=[w1])
            nearest_words = [w for w, s in nearest]

            if w2 in nearest_words:
                rank = nearest_words.index(w2) + 1
                reciprocal_ranks.append(1.0 / rank)
                if rank == 1:
                    hits_at_1 += 1
                hits_at_k += 1
            else:
                reciprocal_ranks.append(0.0)

        n = len(test_pairs)
        return {
            'method': method,
            'hits@1': hits_at_1 / n if n > 0 else 0,
            'hits@k': hits_at_k / n if n > 0 else 0,
            'mrr': np.mean(reciprocal_ranks) if reciprocal_ranks else 0,
            'k': k,
            'n_pairs': n
        }


def run_generation_experiment():
    """Run the antonym generation experiment."""
    print("=" * 70)
    print("EXPERIMENT 3: Antonym Generation")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    antonym_pairs = extract_antonym_pairs()
    known_antonyms = group_antonyms_by_word()
    loader = Word2VecLoader()
    model = loader.load_glove(100)

    # Split into train/test
    np.random.seed(42)
    pairs = [(w1, w2) for w1, w2 in antonym_pairs if w1 in model and w2 in model]
    np.random.shuffle(pairs)
    split = int(len(pairs) * 0.8)
    train_pairs = pairs[:split]
    test_pairs = pairs[split:]
    print(f"  Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    # Build generator
    print("\n[2/3] Learning antonym transformations...")
    generator = AntonymGenerator(model, train_pairs)

    # Get candidates
    valid_words = [
        w for w in model.key_to_index
        if w.isalpha() and w == w.lower() and 3 <= len(w) <= 12
    ][:30000]

    # Evaluate methods
    print("\n[3/3] Evaluating methods...")
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    methods = ['offset', 'linear', 'mirror', 'ensemble']
    results = []

    for method in methods:
        result = generator.evaluate_method(method, test_pairs, valid_words, k=10)
        results.append(result)
        print(f"\n  {method.upper()}:")
        print(f"    Hits@1:  {result['hits@1']*100:.1f}%")
        print(f"    Hits@10: {result['hits@k']*100:.1f}%")
        print(f"    MRR:     {result['mrr']:.3f}")

    # Generate antonyms for words without known antonyms
    print("\n" + "=" * 70)
    print("GENERATING ANTONYMS FOR NEW WORDS")
    print("=" * 70)

    test_words = [
        "technology", "computer", "democracy", "philosophy", "science",
        "music", "art", "nature", "culture", "history",
        "future", "present", "reality", "dream", "hope",
        "fear", "courage", "wisdom", "knowledge", "ignorance"
    ]

    print("\nGenerated antonyms (ensemble method):")
    for word in test_words:
        if word not in model:
            continue

        known = known_antonyms.get(word, set())

        # Generate using all methods
        generated_ensemble = generator.generate_ensemble(word)
        nearest = generator.find_nearest_words(generated_ensemble, valid_words, top_n=5, exclude=[word])

        print(f"\n  {word}:")
        if known:
            print(f"    Known antonyms: {', '.join(known)}")
        else:
            print(f"    Known antonyms: (none)")
        print(f"    Generated:      {', '.join([f'{w}({s:.2f})' for w, s in nearest])}")

    # Compare methods on specific examples
    print("\n" + "=" * 70)
    print("METHOD COMPARISON ON EXAMPLES")
    print("=" * 70)

    example_words = ["good", "happy", "love", "light", "fast"]

    for word in example_words:
        if word not in model:
            continue

        print(f"\n  {word}:")
        for method in methods:
            generate_fn = {
                'offset': generator.generate_offset,
                'linear': generator.generate_linear,
                'mirror': generator.generate_mirror,
                'ensemble': generator.generate_ensemble,
            }[method]

            generated = generate_fn(word)
            nearest = generator.find_nearest_words(generated, valid_words, top_n=3, exclude=[word])
            nearest_str = ", ".join([w for w, s in nearest])
            print(f"    {method:10s}: {nearest_str}")

    return generator, results


if __name__ == "__main__":
    run_generation_experiment()
