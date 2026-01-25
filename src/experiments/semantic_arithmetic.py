"""
Semantic Arithmetic using Antonym Axes

Key insight: If zero has meaning (neutral point on antonym axes),
then addition/subtraction also have meaning (moving along semantic axes).

This module demonstrates that antonym axes form a "coordinate system"
for meaning, enabling semantic operations through vector arithmetic.
"""
import sys
sys.path.insert(0, '..')

import numpy as np
from numpy.linalg import norm
from typing import List, Dict, Tuple, Optional

from antonym_loader import extract_antonym_pairs
from word2vec_loader import Word2VecLoader


class SemanticArithmetic:
    """
    Perform semantic arithmetic using antonym axes as coordinate system.
    """

    def __init__(self, model):
        self.model = model
        self.dim = model.vector_size

        # Vocabulary
        self.vocab = [
            w for w in model.key_to_index
            if w.isalpha() and w == w.lower() and 3 <= len(w) <= 12
        ][:30000]

        # Predefined axes
        self.axes = {}

    def create_axis(self, positive: str, negative: str, name: str = None) -> np.ndarray:
        """Create a semantic axis from two poles."""
        if positive not in self.model or negative not in self.model:
            return None

        direction = self.model[positive] - self.model[negative]
        normalized = direction / norm(direction)

        if name:
            self.axes[name] = {
                'direction': normalized,
                'magnitude': norm(direction),
                'positive': positive,
                'negative': negative
            }

        return direction

    def find_nearest(self, vector: np.ndarray, exclude: List[str] = None,
                      top_n: int = 10) -> List[Tuple[str, float]]:
        """Find nearest words to a vector."""
        exclude = set(exclude or [])
        results = []
        vec_norm = norm(vector)

        for w in self.vocab:
            if w in exclude:
                continue
            v = self.model[w]
            sim = np.dot(vector, v) / (vec_norm * norm(v) + 1e-10)
            results.append((w, float(sim)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    # =================================================================
    # Semantic Operations
    # =================================================================

    def add_quality(self, word: str, axis_name: str, magnitude: float = 1.0) -> List[Tuple[str, float]]:
        """Add a semantic quality to a word."""
        if word not in self.model or axis_name not in self.axes:
            return []

        axis = self.axes[axis_name]
        v_word = self.model[word]
        v_new = v_word + axis['direction'] * axis['magnitude'] * magnitude

        return self.find_nearest(v_new, exclude=[word])

    def intensify(self, word: str, reference_pair: Tuple[str, str],
                   magnitude: float = 1.0) -> List[Tuple[str, float]]:
        """Intensify a word using a reference pair (weak, strong)."""
        weak, strong = reference_pair

        if word not in self.model or weak not in self.model or strong not in self.model:
            return []

        intensity_vector = self.model[strong] - self.model[weak]
        v_word = self.model[word]
        v_intensified = v_word + intensity_vector * magnitude

        return self.find_nearest(v_intensified, exclude=[word, weak, strong])

    def transform_opposite(self, word: str, source: str, target: str) -> List[Tuple[str, float]]:
        """Transform a word in the direction from source to target."""
        if word not in self.model or source not in self.model or target not in self.model:
            return []

        transform = self.model[target] - self.model[source]
        v_word = self.model[word]
        v_transformed = v_word + transform

        return self.find_nearest(v_transformed, exclude=[word, source, target])

    def compose(self, word1: str, word2: str) -> List[Tuple[str, float]]:
        """Compose two words (simple addition)."""
        if word1 not in self.model or word2 not in self.model:
            return []

        v_sum = self.model[word1] + self.model[word2]
        return self.find_nearest(v_sum, exclude=[word1, word2])

    def decompose(self, word: str, axes: List[str] = None) -> Dict[str, float]:
        """Decompose a word into its semantic components."""
        if word not in self.model:
            return {}

        if axes is None:
            axes = list(self.axes.keys())

        v = self.model[word]
        components = {}

        for axis_name in axes:
            if axis_name in self.axes:
                axis = self.axes[axis_name]
                proj = float(np.dot(v, axis['direction']))
                components[axis_name] = proj

        return components

    def neutralize(self, word: str, axis_name: str) -> List[Tuple[str, float]]:
        """Remove a semantic component from a word."""
        if word not in self.model or axis_name not in self.axes:
            return []

        axis = self.axes[axis_name]
        v = self.model[word]
        proj = np.dot(v, axis['direction'])
        v_neutralized = v - proj * axis['direction']

        return self.find_nearest(v_neutralized, exclude=[word])


def run_semantic_arithmetic_demo():
    """Demonstrate semantic arithmetic operations."""
    print("=" * 70)
    print("SEMANTIC ARITHMETIC DEMO")
    print("=" * 70)

    # Load model
    loader = Word2VecLoader()
    model = loader.load_glove(100)

    arith = SemanticArithmetic(model)

    # Create axes
    arith.create_axis('big', 'small', 'size')
    arith.create_axis('good', 'bad', 'quality')
    arith.create_axis('hot', 'cold', 'temperature')
    arith.create_axis('fast', 'slow', 'speed')
    arith.create_axis('happy', 'sad', 'emotion')

    # Demo 1: Intensity
    print("\n" + "=" * 60)
    print("INTENSITY SCALING")
    print("=" * 60)

    intensity_examples = [
        ('good', ('good', 'great')),
        ('bad', ('bad', 'terrible')),
        ('happy', ('happy', 'joyful')),
        ('cold', ('cold', 'freezing')),
    ]

    for word, pair in intensity_examples:
        result = arith.intensify(word, pair)
        print(f"  {word} + intensity → {', '.join([w for w, _ in result[:3]])}")

    # Demo 2: Attribute Addition
    print("\n" + "=" * 60)
    print("ATTRIBUTE ADDITION")
    print("=" * 60)

    for noun in ['car', 'house', 'idea']:
        print(f"\n  {noun}:")
        for axis in ['size', 'quality']:
            result = arith.add_quality(noun, axis, magnitude=0.5)
            print(f"    +{axis}: {', '.join([w for w, _ in result[:3]])}")

    # Demo 3: Decomposition
    print("\n" + "=" * 60)
    print("SEMANTIC DECOMPOSITION")
    print("=" * 60)

    for word in ['excellent', 'terrible', 'cottage', 'mansion']:
        components = arith.decompose(word)
        sorted_comp = sorted(components.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  {word}:")
        for axis, value in sorted_comp[:3]:
            print(f"    {axis}: {value:+.2f}")

    return arith


if __name__ == "__main__":
    run_semantic_arithmetic_demo()
