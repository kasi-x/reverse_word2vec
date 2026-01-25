"""
WSD-based Antonym Detection & New Axis Discovery

Goals:
1. Use context to automatically select the appropriate antonym axis
2. Discover NEW antonym axes not in WordNet
"""
import sys
sys.path.insert(0, '..')

import numpy as np
from numpy.linalg import norm
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from collections import defaultdict

from antonym_loader import extract_antonym_pairs, group_antonyms_by_word
from word2vec_loader import Word2VecLoader


@dataclass
class SenseAxis:
    """Represents one sense/axis for a polysemous word."""
    word: str
    antonym: str
    direction: np.ndarray  # normalized direction vector
    example_words: List[str]  # words that cluster with this sense


class WSDBasedAntonymFinder:
    """
    Word Sense Disambiguation based antonym finder.

    Automatically selects the appropriate antonym axis based on context,
    and can discover new potential antonym axes.
    """

    def __init__(self, model, antonym_pairs: List[Tuple[str, str]]):
        self.model = model
        self.dim = model.vector_size

        print("Building WSD-based antonym finder...")

        # Build known antonyms mapping
        self.known_antonyms = group_antonyms_by_word()

        # Build direction matrix for all antonym pairs
        self.directions = []
        self.direction_pairs = []
        for w1, w2 in antonym_pairs:
            if w1 in model and w2 in model:
                d = model[w1] - model[w2]
                d_norm = norm(d)
                if d_norm > 1e-6:
                    self.directions.append(d / d_norm)
                    self.direction_pairs.append((w1, w2))

        self.D = np.array(self.directions, dtype=np.float32)
        print(f"  {len(self.directions)} antonym directions loaded")

        # Build vocabulary
        self.vocab = [
            w for w in model.key_to_index
            if w.isalpha() and w == w.lower() and 3 <= len(w) <= 12
        ][:30000]
        print(f"  {len(self.vocab)} candidate words")

    def get_sense_axes(self, word: str) -> List[SenseAxis]:
        """Get all known sense axes for a word."""
        axes = []
        antonyms = self.known_antonyms.get(word, set())

        for ant in antonyms:
            if ant not in self.model or word not in self.model:
                continue

            direction = self.model[word] - self.model[ant]
            direction = direction / (norm(direction) + 1e-10)

            axes.append(SenseAxis(
                word=word,
                antonym=ant,
                direction=direction,
                example_words=[]
            ))

        return axes

    def context_to_vector(self, context_words: List[str]) -> Optional[np.ndarray]:
        """Convert context words to an average vector."""
        vectors = []
        for w in context_words:
            w_lower = w.lower()
            if w_lower in self.model:
                vectors.append(self.model[w_lower])

        if not vectors:
            return None

        return np.mean(vectors, axis=0)

    def select_sense_by_context(self, word: str, context_words: List[str]) -> Optional[SenseAxis]:
        """
        WSD: Select the most appropriate sense axis based on context.

        Method: Find which axis's antonym is most similar to the context.
        If context talks about "weight", "heavy" axis is more relevant.
        """
        axes = self.get_sense_axes(word)
        if not axes:
            return None

        if len(axes) == 1:
            return axes[0]

        context_vec = self.context_to_vector(context_words)
        if context_vec is None:
            return axes[0]  # Default to first

        # Score each axis by how well context aligns with the semantic field
        best_axis = None
        best_score = -float('inf')

        for axis in axes:
            # Method 1: Similarity of context to the antonym
            ant_vec = self.model[axis.antonym]
            score1 = np.dot(context_vec, ant_vec) / (norm(context_vec) * norm(ant_vec) + 1e-10)

            # Method 2: How much does context project onto the axis direction?
            score2 = abs(np.dot(context_vec, axis.direction)) / (norm(context_vec) + 1e-10)

            # Combined score
            score = score1 + 0.5 * score2

            if score > best_score:
                best_score = score
                best_axis = axis

        return best_axis

    def find_antonyms_for_sense(self, word: str, sense_axis: SenseAxis,
                                  top_n: int = 10) -> List[Tuple[str, float]]:
        """Find antonyms aligned with a specific sense axis."""
        if word not in self.model:
            return []

        v_word = self.model[word]
        results = []

        for cand in self.vocab:
            if cand == word or cand not in self.model:
                continue

            v_cand = self.model[cand]
            diff = v_word - v_cand
            diff_norm = norm(diff)
            if diff_norm < 1e-6:
                continue

            # Alignment with the sense axis
            alignment = abs(np.dot(diff / diff_norm, sense_axis.direction))
            results.append((cand, alignment))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def find_antonyms_with_context(self, word: str, context: str,
                                    top_n: int = 10) -> Dict:
        """
        Main WSD interface: Find antonyms based on context sentence.
        """
        # Tokenize context (simple split)
        context_words = [w.strip('.,!?()[]"\'') for w in context.lower().split()]
        context_words = [w for w in context_words if w != word.lower() and len(w) > 2]

        # Select sense
        selected_sense = self.select_sense_by_context(word, context_words)

        if selected_sense is None:
            return {
                'word': word,
                'context': context,
                'selected_sense': None,
                'antonyms': []
            }

        # Find antonyms for this sense
        antonyms = self.find_antonyms_for_sense(word, selected_sense, top_n)

        return {
            'word': word,
            'context': context,
            'selected_sense': selected_sense.antonym,
            'all_senses': [a.antonym for a in self.get_sense_axes(word)],
            'antonyms': antonyms
        }

    def discover_new_axes(self, word: str, top_n: int = 20,
                           min_score: float = 0.5) -> List[Dict]:
        """
        Discover potential NEW antonym axes not in WordNet.

        Method:
        1. Find words with high antonym scores
        2. Filter out those aligned with known axes
        3. Remaining high-scorers might be new axes
        """
        if word not in self.model:
            return []

        known_axes = self.get_sense_axes(word)
        known_antonyms = {ax.antonym for ax in known_axes}

        v_word = self.model[word]

        # Find all candidates with antonym-like relationships
        candidates = []
        for cand in self.vocab:
            if cand == word or cand not in self.model:
                continue

            v_cand = self.model[cand]
            diff = v_word - v_cand
            diff_norm = norm(diff)
            if diff_norm < 1e-6:
                continue

            diff_normalized = diff / diff_norm

            # Score: alignment with ANY known antonym direction
            global_score = np.max(np.abs(self.D @ diff_normalized))

            # Score on each known axis for this word
            axis_scores = {}
            for ax in known_axes:
                axis_scores[ax.antonym] = abs(np.dot(diff_normalized, ax.direction))

            candidates.append({
                'word': cand,
                'global_score': float(global_score),
                'axis_scores': axis_scores,
                'diff_vector': diff_normalized
            })

        # Sort by global antonym score
        candidates.sort(key=lambda x: x['global_score'], reverse=True)

        # Find candidates that:
        # 1. Have high global antonym score
        # 2. Don't align strongly with any known axis
        new_axes = []

        for cand in candidates[:200]:  # Check top 200
            if cand['global_score'] < min_score:
                continue

            # Check if it aligns with known axes
            max_known_alignment = max(cand['axis_scores'].values()) if cand['axis_scores'] else 0

            # If not strongly aligned with known axes, it might be a new axis
            if max_known_alignment < 0.7:  # Threshold for "not aligned"
                new_axes.append({
                    'candidate_antonym': cand['word'],
                    'global_antonym_score': cand['global_score'],
                    'known_axis_alignments': cand['axis_scores'],
                    'novelty': cand['global_score'] - max_known_alignment
                })

        # Sort by novelty (high global score but low known alignment)
        new_axes.sort(key=lambda x: x['novelty'], reverse=True)

        return new_axes[:top_n]

    def analyze_potential_axis(self, word: str, candidate_antonym: str,
                                 top_n: int = 15) -> Dict:
        """
        Analyze a potential new axis in detail.

        Returns words that would be antonyms on this new axis.
        """
        if word not in self.model or candidate_antonym not in self.model:
            return {}

        # Create axis
        direction = self.model[word] - self.model[candidate_antonym]
        direction = direction / (norm(direction) + 1e-10)

        new_axis = SenseAxis(
            word=word,
            antonym=candidate_antonym,
            direction=direction,
            example_words=[]
        )

        # Find words on this axis
        antonyms = self.find_antonyms_for_sense(word, new_axis, top_n)

        # Find words on the positive pole (similar to word)
        positive_pole = []
        for cand in self.vocab:
            if cand == word or cand not in self.model:
                continue
            v_cand = self.model[cand]
            # Projection onto axis (positive = toward word)
            proj = np.dot(v_cand - self.model[candidate_antonym], direction)
            positive_pole.append((cand, proj))

        positive_pole.sort(key=lambda x: x[1], reverse=True)

        # Find words on the negative pole (similar to candidate_antonym)
        negative_pole = []
        for cand in self.vocab:
            if cand == candidate_antonym or cand not in self.model:
                continue
            v_cand = self.model[cand]
            proj = np.dot(v_cand - self.model[word], -direction)
            negative_pole.append((cand, proj))

        negative_pole.sort(key=lambda x: x[1], reverse=True)

        return {
            'word': word,
            'candidate_antonym': candidate_antonym,
            'antonyms_on_axis': antonyms,
            'positive_pole': positive_pole[:10],
            'negative_pole': negative_pole[:10],
        }


def run_wsd_experiment():
    """Run the WSD-based antonym experiment."""
    print("=" * 70)
    print("WSD-BASED ANTONYM DETECTION & NEW AXIS DISCOVERY")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    antonym_pairs = extract_antonym_pairs()
    loader = Word2VecLoader()
    model = loader.load_glove(100)

    # Build finder
    print("\n[2/4] Building WSD-based finder...")
    finder = WSDBasedAntonymFinder(model, antonym_pairs)

    # Test WSD with context
    print("\n[3/4] Testing WSD with context...")
    print("\n" + "=" * 70)
    print("CONTEXT-BASED SENSE SELECTION")
    print("=" * 70)

    test_cases = [
        ("light", "The bag is very light and easy to carry."),
        ("light", "The room was filled with natural light from the window."),
        ("light", "She has light blonde hair."),
        ("right", "Turn right at the next intersection."),
        ("right", "That's not the right answer to the question."),
        ("right", "Everyone has the right to free speech."),
        ("fast", "He drives too fast on the highway."),
        ("fast", "Muslims fast during Ramadan."),
        ("hard", "The exam was very hard."),
        ("hard", "The surface is hard like stone."),
    ]

    for word, context in test_cases:
        result = finder.find_antonyms_with_context(word, context, top_n=5)

        print(f"\n「{word}」in: \"{context}\"")
        print(f"  Available senses: {result.get('all_senses', [])}")
        print(f"  Selected sense: {word} ↔ {result['selected_sense']}")
        print(f"  Top antonyms: ", end="")
        if result['antonyms']:
            print(", ".join([f"{w}({s:.2f})" for w, s in result['antonyms'][:5]]))
        else:
            print("(none)")

    # Discover new axes
    print("\n\n[4/4] Discovering new antonym axes...")
    print("\n" + "=" * 70)
    print("NEW AXIS DISCOVERY FOR 'light'")
    print("=" * 70)

    word = "light"
    known = finder.known_antonyms.get(word, set())
    print(f"\nKnown antonyms for '{word}': {known}")

    new_axes = finder.discover_new_axes(word, top_n=15, min_score=0.4)

    print(f"\nPotential NEW antonym axes (not in WordNet):")
    print("-" * 60)

    for i, ax in enumerate(new_axes, 1):
        print(f"\n{i}. {word} ↔ {ax['candidate_antonym']}")
        print(f"   Global antonym score: {ax['global_antonym_score']:.3f}")
        print(f"   Novelty score: {ax['novelty']:.3f}")
        print(f"   Alignments with known axes: ", end="")
        for known_ax, score in ax['known_axis_alignments'].items():
            print(f"{known_ax}({score:.2f}) ", end="")
        print()

    # Analyze top candidates in detail
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS OF POTENTIAL NEW AXES")
    print("=" * 70)

    for ax in new_axes[:5]:
        cand = ax['candidate_antonym']
        analysis = finder.analyze_potential_axis(word, cand, top_n=10)

        print(f"\n{'─' * 60}")
        print(f"AXIS: {word} ↔ {cand}")
        print(f"{'─' * 60}")

        print(f"\n  Words on '{word}' pole (positive):")
        for w, score in analysis['positive_pole'][:7]:
            print(f"    {w:15s} ({score:.2f})")

        print(f"\n  Words on '{cand}' pole (negative):")
        for w, score in analysis['negative_pole'][:7]:
            print(f"    {w:15s} ({score:.2f})")

    # Try other polysemous words
    print("\n\n" + "=" * 70)
    print("NEW AXIS DISCOVERY FOR OTHER WORDS")
    print("=" * 70)

    for test_word in ["good", "hot", "fast", "cold"]:
        known = finder.known_antonyms.get(test_word, set())
        print(f"\n\n{'=' * 40}")
        print(f"'{test_word}' - Known antonyms: {known}")
        print(f"{'=' * 40}")

        new_axes = finder.discover_new_axes(test_word, top_n=10, min_score=0.4)

        if new_axes:
            print("\nPotential new axes:")
            for ax in new_axes[:5]:
                print(f"  {test_word} ↔ {ax['candidate_antonym']:15s} "
                      f"(score: {ax['global_antonym_score']:.2f}, "
                      f"novelty: {ax['novelty']:.2f})")
        else:
            print("  No new axes found.")

    return finder


if __name__ == "__main__":
    run_wsd_experiment()
