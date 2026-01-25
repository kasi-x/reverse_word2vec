"""
Experiment 2: Antonym Intensity/Degree Analysis

Goal: Learn not just IF two words are antonyms, but HOW STRONG the opposition is.

Examples:
- big/small vs huge/tiny (same axis, different intensity)
- warm/cool vs hot/cold (same axis, different intensity)
- dislike/like vs hate/love (same axis, different intensity)

Approach:
1. Use the distance between antonym pairs as intensity measure
2. Project words onto antonym axes and measure distance from center
3. Build intensity scales for each antonym dimension
"""
import sys
sys.path.insert(0, '..')

import numpy as np
from numpy.linalg import norm
from collections import defaultdict
from typing import List, Tuple, Dict
from dataclasses import dataclass
import time

from antonym_loader import extract_antonym_pairs
from word2vec_loader import Word2VecLoader


@dataclass
class IntensityScale:
    """Represents an intensity scale on an antonym axis."""
    positive_pole: str  # e.g., "hot"
    negative_pole: str  # e.g., "cold"
    midpoint: np.ndarray
    direction: np.ndarray
    words_on_scale: List[Tuple[str, float]]  # (word, position) sorted by position


class IntensityAnalyzer:
    """
    Analyzer for antonym intensity/degree.

    Maps words onto antonym scales and measures their intensity.
    """

    def __init__(self, model, antonym_pairs: List[Tuple[str, str]]):
        self.model = model
        self.dim = model.vector_size

        print("Building intensity analyzer...")

        # Build scales from antonym pairs
        self.scales: Dict[Tuple[str, str], IntensityScale] = {}

        for w1, w2 in antonym_pairs:
            if w1 not in model or w2 not in model:
                continue

            v1, v2 = model[w1], model[w2]
            direction = v1 - v2
            direction_norm = norm(direction)

            if direction_norm < 1e-6:
                continue

            direction = direction / direction_norm
            midpoint = (v1 + v2) / 2

            self.scales[(w1, w2)] = IntensityScale(
                positive_pole=w1,
                negative_pole=w2,
                midpoint=midpoint,
                direction=direction,
                words_on_scale=[]
            )

        print(f"  Created {len(self.scales)} intensity scales")

    def get_position_on_scale(self, word: str, scale_key: Tuple[str, str]) -> float:
        """
        Get a word's position on an antonym scale.

        Returns:
            Float in range roughly [-1, 1] where:
            - Positive = closer to positive pole
            - Negative = closer to negative pole
            - 0 = neutral (at midpoint)
        """
        if word not in self.model or scale_key not in self.scales:
            return None

        scale = self.scales[scale_key]
        v = self.model[word]

        # Project onto the scale direction, relative to midpoint
        position = np.dot(v - scale.midpoint, scale.direction)

        # Normalize by the scale's half-length (distance from midpoint to pole)
        pole_distance = norm(self.model[scale.positive_pole] - scale.midpoint)
        normalized_position = position / (pole_distance + 1e-10)

        return float(normalized_position)

    def build_word_scale(self, scale_key: Tuple[str, str],
                         candidates: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Build an ordered intensity scale with many words.

        Returns words ordered from negative pole to positive pole.
        """
        if scale_key not in self.scales:
            return []

        scale = self.scales[scale_key]
        positions = []

        for word in candidates:
            if word not in self.model:
                continue
            pos = self.get_position_on_scale(word, scale_key)
            if pos is not None:
                positions.append((word, pos))

        # Sort by position
        positions.sort(key=lambda x: x[1])

        # Return words spread across the scale
        if len(positions) <= top_n:
            return positions

        # Sample evenly across the scale
        indices = np.linspace(0, len(positions) - 1, top_n, dtype=int)
        return [positions[i] for i in indices]

    def find_intensity_neighbors(self, word: str, scale_key: Tuple[str, str],
                                  candidates: List[str], top_n: int = 5) -> Dict:
        """
        Find words at similar and different intensities on a scale.
        """
        if word not in self.model or scale_key not in self.scales:
            return {}

        word_pos = self.get_position_on_scale(word, scale_key)
        if word_pos is None:
            return {}

        all_positions = []
        for cand in candidates:
            if cand == word or cand not in self.model:
                continue
            pos = self.get_position_on_scale(cand, scale_key)
            if pos is not None:
                all_positions.append((cand, pos))

        # Find similar intensity (close absolute position)
        similar = sorted(all_positions, key=lambda x: abs(x[1] - word_pos))[:top_n]

        # Find opposite intensity (opposite sign, similar magnitude)
        opposite = sorted(all_positions, key=lambda x: abs(x[1] + word_pos))[:top_n]

        # Find stronger versions (same sign, higher magnitude)
        same_sign = [p for p in all_positions if p[1] * word_pos > 0]
        stronger = sorted(same_sign, key=lambda x: abs(x[1]), reverse=True)[:top_n]

        # Find weaker versions (same sign, lower magnitude)
        weaker = sorted(same_sign, key=lambda x: abs(x[1]))[:top_n]

        return {
            'word': word,
            'position': word_pos,
            'similar_intensity': similar,
            'opposite_intensity': opposite,
            'stronger': stronger,
            'weaker': weaker
        }

    def analyze_intensity_clusters(self, scale_key: Tuple[str, str],
                                    candidates: List[str]) -> Dict:
        """
        Cluster words on a scale by intensity level.
        """
        if scale_key not in self.scales:
            return {}

        positions = []
        for word in candidates:
            if word not in self.model:
                continue
            pos = self.get_position_on_scale(word, scale_key)
            if pos is not None:
                positions.append((word, pos))

        if not positions:
            return {}

        # Divide into intensity bins
        bins = {
            'strong_negative': [],   # < -0.7
            'moderate_negative': [], # -0.7 to -0.3
            'weak_negative': [],     # -0.3 to 0
            'weak_positive': [],     # 0 to 0.3
            'moderate_positive': [], # 0.3 to 0.7
            'strong_positive': [],   # > 0.7
        }

        for word, pos in positions:
            if pos < -0.7:
                bins['strong_negative'].append((word, pos))
            elif pos < -0.3:
                bins['moderate_negative'].append((word, pos))
            elif pos < 0:
                bins['weak_negative'].append((word, pos))
            elif pos < 0.3:
                bins['weak_positive'].append((word, pos))
            elif pos < 0.7:
                bins['moderate_positive'].append((word, pos))
            else:
                bins['strong_positive'].append((word, pos))

        # Sort each bin
        for key in bins:
            bins[key].sort(key=lambda x: x[1], reverse=('positive' in key))

        return bins


def run_intensity_experiment():
    """Run the intensity analysis experiment."""
    print("=" * 70)
    print("EXPERIMENT 2: Antonym Intensity Analysis")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    antonym_pairs = extract_antonym_pairs()
    loader = Word2VecLoader()
    model = loader.load_glove(100)

    # Build analyzer
    print("\n[2/3] Building analyzer...")
    analyzer = IntensityAnalyzer(model, antonym_pairs)

    # Get candidates
    valid_words = [
        w for w in model.key_to_index
        if w.isalpha() and w == w.lower() and 3 <= len(w) <= 12
    ][:30000]

    # Test scales
    print("\n[3/3] Analyzing intensity scales...")

    test_scales = [
        ("hot", "cold"),
        ("big", "small"),
        ("good", "bad"),
        ("happy", "sad"),
        ("love", "hate"),
        ("fast", "slow"),
        ("strong", "weak"),
        ("rich", "poor"),
        ("young", "old"),
        ("light", "dark"),
    ]

    print("\n" + "=" * 70)
    print("INTENSITY SCALES")
    print("=" * 70)

    for pos_pole, neg_pole in test_scales:
        scale_key = (pos_pole, neg_pole)
        if scale_key not in analyzer.scales:
            continue

        print(f"\n{neg_pole.upper()} <-----> {pos_pole.upper()}")

        scale_words = analyzer.build_word_scale(scale_key, valid_words, top_n=15)
        if scale_words:
            for word, pos in scale_words:
                bar_pos = int((pos + 1) * 15)  # Map [-1,1] to [0,30]
                bar_pos = max(0, min(30, bar_pos))
                bar = "-" * bar_pos + "|" + "-" * (30 - bar_pos)
                print(f"  {word:15s} [{bar}] {pos:+.2f}")

    # Intensity neighbors
    print("\n" + "=" * 70)
    print("INTENSITY NEIGHBORS")
    print("=" * 70)

    test_cases = [
        ("warm", ("hot", "cold")),
        ("large", ("big", "small")),
        ("nice", ("good", "bad")),
        ("quick", ("fast", "slow")),
    ]

    for word, scale_key in test_cases:
        if scale_key not in analyzer.scales:
            continue

        neighbors = analyzer.find_intensity_neighbors(word, scale_key, valid_words, top_n=5)
        if not neighbors:
            continue

        print(f"\n{word.upper()} on {scale_key[1]}/{scale_key[0]} scale (pos={neighbors['position']:.2f}):")

        print(f"  Stronger versions: ", end="")
        print(", ".join([f"{w}({p:.2f})" for w, p in neighbors['stronger'][:3]]))

        print(f"  Weaker versions:   ", end="")
        print(", ".join([f"{w}({p:.2f})" for w, p in neighbors['weaker'][:3]]))

        print(f"  Similar intensity: ", end="")
        print(", ".join([f"{w}({p:.2f})" for w, p in neighbors['similar_intensity'][:3]]))

        print(f"  Opposite:          ", end="")
        print(", ".join([f"{w}({p:.2f})" for w, p in neighbors['opposite_intensity'][:3]]))

    # Intensity clusters
    print("\n" + "=" * 70)
    print("INTENSITY CLUSTERS")
    print("=" * 70)

    for pos_pole, neg_pole in test_scales[:3]:
        scale_key = (pos_pole, neg_pole)
        if scale_key not in analyzer.scales:
            continue

        clusters = analyzer.analyze_intensity_clusters(scale_key, valid_words)

        print(f"\n{neg_pole.upper()} <-----> {pos_pole.upper()}:")

        for level, words in clusters.items():
            if words:
                sample = [w for w, p in words[:5]]
                print(f"  {level:20s}: {', '.join(sample)}")

    return analyzer


if __name__ == "__main__":
    run_intensity_experiment()
