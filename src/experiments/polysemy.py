"""
Experiment 1: Polysemy Handling - Multiple Antonym Dimensions

Key insight: A word like "light" has multiple antonyms:
- light ↔ heavy (weight)
- light ↔ dark (brightness)

Instead of treating these as a single relationship, we create
separate dimensions for each antonym pair the word participates in.

This allows:
1. A word to have different positions on different antonym axes
2. Discovery of which "sense" of a word is being used
3. More accurate antonym prediction
"""
import sys
sys.path.insert(0, '..')

import numpy as np
from numpy.linalg import norm
from collections import defaultdict
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import time

from antonym_loader import extract_antonym_pairs, group_antonyms_by_word
from word2vec_loader import Word2VecLoader


@dataclass
class PolysemousAntonymRelation:
    """Represents one antonym relation for a polysemous word."""
    word: str
    antonym: str
    direction: np.ndarray
    sense_label: str  # e.g., "weight", "brightness"


class PolysemyAwareAnalyzer:
    """
    Analyzer that handles polysemous words with multiple antonym relations.

    For each word, we track ALL its antonym relationships as separate dimensions.
    """

    def __init__(self, model, antonym_pairs: List[Tuple[str, str]]):
        self.model = model
        self.dim = model.vector_size

        print("Building polysemy-aware analyzer...")

        # Build word -> list of antonyms mapping
        self.word_antonyms: Dict[str, List[str]] = defaultdict(list)
        for w1, w2 in antonym_pairs:
            if w1 in model and w2 in model:
                self.word_antonyms[w1].append(w2)
                self.word_antonyms[w2].append(w1)

        # Find polysemous words (multiple antonyms)
        self.polysemous_words = {
            w: ants for w, ants in self.word_antonyms.items()
            if len(ants) > 1
        }
        print(f"  Found {len(self.polysemous_words)} polysemous words")

        # Build separate direction vectors for each antonym relation
        self.relations: Dict[str, List[PolysemousAntonymRelation]] = defaultdict(list)
        self.all_directions = []
        self.direction_labels = []

        for word, antonyms in self.word_antonyms.items():
            for ant in antonyms:
                d = model[word] - model[ant]
                d_norm = norm(d)
                if d_norm > 1e-6:
                    direction = d / d_norm
                    # Create a sense label based on the antonym
                    sense_label = f"{word}/{ant}"

                    rel = PolysemousAntonymRelation(
                        word=word,
                        antonym=ant,
                        direction=direction,
                        sense_label=sense_label
                    )
                    self.relations[word].append(rel)
                    self.all_directions.append(direction)
                    self.direction_labels.append(sense_label)

        self.D = np.array(self.all_directions, dtype=np.float32)
        print(f"  Created {len(self.all_directions)} antonym direction vectors")

    def get_word_senses(self, word: str) -> List[PolysemousAntonymRelation]:
        """Get all antonym-based senses of a word."""
        return self.relations.get(word, [])

    def analyze_polysemy(self, word: str) -> Dict:
        """
        Analyze a word's position on each of its antonym dimensions.

        Returns projections onto each antonym axis the word participates in.
        """
        if word not in self.model:
            return {}

        v = self.model[word]
        senses = self.get_word_senses(word)

        result = {
            'word': word,
            'num_senses': len(senses),
            'senses': []
        }

        for rel in senses:
            proj = np.dot(v, rel.direction)
            result['senses'].append({
                'antonym': rel.antonym,
                'projection': float(proj),
                'direction_strength': float(norm(self.model[word] - self.model[rel.antonym]))
            })

        return result

    def find_antonyms_by_sense(self, word: str, sense_antonym: str,
                                candidates: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find antonyms for a specific sense of a word.

        Args:
            word: The query word
            sense_antonym: The antonym that defines which sense to use
            candidates: Candidate words to search
            top_n: Number of results

        Returns:
            List of (word, score) tuples
        """
        if word not in self.model or sense_antonym not in self.model:
            return []

        # Get the direction for this specific sense
        direction = self.model[word] - self.model[sense_antonym]
        direction = direction / (norm(direction) + 1e-10)

        results = []
        v_word = self.model[word]

        for cand in candidates:
            if cand == word or cand not in self.model:
                continue

            v_cand = self.model[cand]
            diff = v_word - v_cand
            diff_norm = norm(diff)
            if diff_norm < 1e-6:
                continue

            # How well does this candidate align with this specific sense?
            alignment = np.dot(diff / diff_norm, direction)
            results.append((cand, float(alignment)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def cluster_senses(self, word: str, threshold: float = 0.7) -> List[List[str]]:
        """
        Cluster a word's antonyms by sense similarity.

        If two antonyms define similar directions, they likely relate to the same sense.
        """
        senses = self.get_word_senses(word)
        if len(senses) <= 1:
            return [[s.antonym for s in senses]]

        # Compute pairwise similarities between directions
        directions = [s.direction for s in senses]
        n = len(directions)

        # Simple clustering: merge if similarity > threshold
        clusters = [[i] for i in range(n)]
        merged = [False] * n

        for i in range(n):
            if merged[i]:
                continue
            for j in range(i + 1, n):
                if merged[j]:
                    continue
                sim = abs(np.dot(directions[i], directions[j]))
                if sim > threshold:
                    clusters[i].extend(clusters[j])
                    merged[j] = True

        result = []
        for i, cluster in enumerate(clusters):
            if not merged[i]:
                result.append([senses[idx].antonym for idx in cluster])

        return result


def run_polysemy_experiment():
    """Run the polysemy handling experiment."""
    print("=" * 70)
    print("EXPERIMENT 1: Polysemy Handling")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    antonym_pairs = extract_antonym_pairs()
    loader = Word2VecLoader()
    model = loader.load_glove(100)

    # Build analyzer
    print("\n[2/4] Building analyzer...")
    analyzer = PolysemyAwareAnalyzer(model, antonym_pairs)

    # Analyze polysemous words
    print("\n[3/4] Analyzing polysemous words...")
    print("\n" + "=" * 70)
    print("TOP POLYSEMOUS WORDS (most antonym relationships)")
    print("=" * 70)

    sorted_poly = sorted(
        analyzer.polysemous_words.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:20]

    for word, antonyms in sorted_poly:
        print(f"\n  {word} ({len(antonyms)} antonyms):")
        for ant in antonyms[:5]:
            print(f"    ↔ {ant}")
        if len(antonyms) > 5:
            print(f"    ... and {len(antonyms) - 5} more")

    # Test specific polysemous words
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS OF SELECTED POLYSEMOUS WORDS")
    print("=" * 70)

    test_words = ["light", "hard", "fast", "right", "fair", "clear", "free", "open"]

    for word in test_words:
        if word not in analyzer.polysemous_words:
            continue

        analysis = analyzer.analyze_polysemy(word)
        clusters = analyzer.cluster_senses(word, threshold=0.5)

        print(f"\n{word.upper()} - {analysis['num_senses']} senses:")

        for i, cluster in enumerate(clusters):
            print(f"  Sense cluster {i + 1}: {', '.join(cluster)}")

        print(f"  Projections on each antonym axis:")
        for sense in analysis['senses'][:5]:
            print(f"    {word} ↔ {sense['antonym']:15s}: proj={sense['projection']:+.3f}")

    # Find sense-specific antonyms
    print("\n" + "=" * 70)
    print("SENSE-SPECIFIC ANTONYM DISCOVERY")
    print("=" * 70)

    # Get valid words
    valid_words = [w for w in model.key_to_index if w.isalpha() and w == w.lower() and 3 <= len(w) <= 12][:30000]

    test_cases = [
        ("light", "heavy"),   # weight sense
        ("light", "dark"),    # brightness sense
        ("hard", "soft"),     # texture sense
        ("hard", "easy"),     # difficulty sense
        ("fast", "slow"),     # speed sense
        ("right", "wrong"),   # correctness sense
        ("right", "left"),    # direction sense
    ]

    print("\nFinding antonyms for specific word senses:")
    for word, sense_antonym in test_cases:
        if word not in model or sense_antonym not in model:
            continue

        results = analyzer.find_antonyms_by_sense(word, sense_antonym, valid_words, top_n=10)

        print(f"\n  {word} (sense: {sense_antonym}):")
        top5 = ", ".join([f"{w}({s:.2f})" for w, s in results[:5]])
        print(f"    Top antonyms: {top5}")

    # Statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    sense_counts = [len(ants) for ants in analyzer.polysemous_words.values()]
    print(f"\n  Words with multiple antonyms: {len(analyzer.polysemous_words)}")
    print(f"  Max antonyms per word: {max(sense_counts)}")
    print(f"  Average antonyms per polysemous word: {np.mean(sense_counts):.2f}")

    # Distribution
    print(f"\n  Distribution of antonym counts:")
    for n in range(2, min(8, max(sense_counts) + 1)):
        count = sum(1 for c in sense_counts if c == n)
        print(f"    {n} antonyms: {count} words")
    count_8plus = sum(1 for c in sense_counts if c >= 8)
    if count_8plus > 0:
        print(f"    8+ antonyms: {count_8plus} words")

    return analyzer


if __name__ == "__main__":
    run_polysemy_experiment()
