"""
New Antonym Axis Explorer

Interactive tool to discover and validate new antonym relationships
not found in WordNet.
"""
import sys
sys.path.insert(0, '..')

import numpy as np
from numpy.linalg import norm
from typing import List, Tuple, Dict, Set
from collections import defaultdict

from antonym_loader import extract_antonym_pairs, group_antonyms_by_word
from word2vec_loader import Word2VecLoader


class NewAxisExplorer:
    """
    Explore and validate potential new antonym axes.
    """

    def __init__(self, model, antonym_pairs: List[Tuple[str, str]]):
        self.model = model
        self.dim = model.vector_size
        self.known_antonyms = group_antonyms_by_word()

        # Build direction matrix
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

        # Vocabulary
        self.vocab = [
            w for w in model.key_to_index
            if w.isalpha() and w == w.lower() and 3 <= len(w) <= 15
        ][:50000]

        # Pre-compute normalized vectors
        self.vocab_vectors = {}
        for w in self.vocab:
            v = model[w]
            self.vocab_vectors[w] = v / (norm(v) + 1e-10)

    def find_axis_words(self, word1: str, word2: str, top_n: int = 20) -> Dict:
        """
        Find words that lie on the axis between word1 and word2.
        """
        if word1 not in self.model or word2 not in self.model:
            return {}

        v1 = self.model[word1]
        v2 = self.model[word2]
        direction = v1 - v2
        direction = direction / (norm(direction) + 1e-10)
        midpoint = (v1 + v2) / 2

        # Project all words onto axis
        projections = []
        for w in self.vocab:
            if w == word1 or w == word2:
                continue
            v = self.model[w]
            # Distance from axis (for filtering)
            diff_from_mid = v - midpoint
            proj_onto_axis = np.dot(diff_from_mid, direction)
            orthogonal = diff_from_mid - proj_onto_axis * direction
            dist_from_axis = norm(orthogonal)

            projections.append({
                'word': w,
                'position': proj_onto_axis,
                'distance_from_axis': dist_from_axis
            })

        # Sort by position
        projections.sort(key=lambda x: x['position'])

        # Get distance of poles
        pole_dist = norm(v1 - midpoint)

        # Categorize
        word1_pole = []
        word2_pole = []
        neutral = []

        for p in projections:
            normalized_pos = p['position'] / (pole_dist + 1e-10)
            p['normalized_position'] = normalized_pos

            # Filter: only keep words reasonably close to axis
            if p['distance_from_axis'] < 3.0:  # threshold
                if normalized_pos > 0.3:
                    word1_pole.append(p)
                elif normalized_pos < -0.3:
                    word2_pole.append(p)
                else:
                    neutral.append(p)

        # Sort poles by position (extreme first)
        word1_pole.sort(key=lambda x: x['normalized_position'], reverse=True)
        word2_pole.sort(key=lambda x: x['normalized_position'])
        neutral.sort(key=lambda x: abs(x['normalized_position']))

        return {
            'word1': word1,
            'word2': word2,
            f'{word1}_pole': word1_pole[:top_n],
            f'{word2}_pole': word2_pole[:top_n],
            'neutral': neutral[:top_n]
        }

    def validate_axis(self, word1: str, word2: str) -> Dict:
        """
        Validate if word1-word2 forms a meaningful antonym axis.

        Checks:
        1. Semantic distance (not too close, not too far)
        2. Alignment with known antonym directions
        3. Symmetry of pole words
        """
        if word1 not in self.model or word2 not in self.model:
            return {'valid': False, 'reason': 'Words not in vocabulary'}

        v1 = self.model[word1]
        v2 = self.model[word2]

        # 1. Distance check
        distance = norm(v1 - v2)
        cosine = np.dot(v1, v2) / (norm(v1) * norm(v2))

        # 2. Alignment with known antonym directions
        direction = (v1 - v2) / distance
        alignments = np.abs(self.D @ direction)
        max_alignment = np.max(alignments)
        avg_alignment = np.mean(alignments)

        # 3. Get pole words for symmetry check
        axis_words = self.find_axis_words(word1, word2, top_n=10)

        # Compute scores
        scores = {
            'distance': float(distance),
            'cosine_similarity': float(cosine),
            'max_antonym_alignment': float(max_alignment),
            'avg_antonym_alignment': float(avg_alignment),
        }

        # Validity criteria
        is_valid = (
            3.0 < distance < 10.0 and  # Not too close or far
            cosine < 0.8 and  # Not synonyms
            max_alignment > 0.3  # Has some antonym-like property
        )

        return {
            'valid': is_valid,
            'scores': scores,
            'axis_words': axis_words
        }

    def discover_new_axes_for_word(self, word: str, top_n: int = 20) -> List[Dict]:
        """
        Discover potential new antonym axes for a given word.

        Strategy:
        1. Find words with high "antonymness" score
        2. Filter out those that align with known axes
        3. Validate remaining candidates
        """
        if word not in self.model:
            return []

        known_axes = self.known_antonyms.get(word, set())
        v_word = self.model[word]

        candidates = []
        for cand in self.vocab:
            if cand == word or cand in known_axes:
                continue
            if cand not in self.model:
                continue

            v_cand = self.model[cand]
            diff = v_word - v_cand
            diff_norm = norm(diff)
            if diff_norm < 1e-6:
                continue

            diff_normalized = diff / diff_norm

            # Global antonym score
            global_score = np.max(np.abs(self.D @ diff_normalized))

            # Alignment with known axes for this word
            known_alignments = {}
            for known_ant in known_axes:
                if known_ant in self.model:
                    known_dir = v_word - self.model[known_ant]
                    known_dir = known_dir / (norm(known_dir) + 1e-10)
                    known_alignments[known_ant] = abs(np.dot(diff_normalized, known_dir))

            max_known = max(known_alignments.values()) if known_alignments else 0

            # Novelty = high global score but low alignment with known
            novelty = global_score - max_known

            if novelty > 0.2 and global_score > 0.4:
                candidates.append({
                    'candidate': cand,
                    'global_score': float(global_score),
                    'max_known_alignment': float(max_known),
                    'novelty': float(novelty),
                    'known_alignments': known_alignments
                })

        candidates.sort(key=lambda x: x['novelty'], reverse=True)
        return candidates[:top_n]

    def suggest_missing_antonyms(self, word: str) -> List[Dict]:
        """
        Suggest antonyms that might be missing from WordNet.

        Uses semantic patterns to find likely antonyms.
        """
        if word not in self.model:
            return []

        v_word = self.model[word]
        known = self.known_antonyms.get(word, set())

        # Strategy: Find words where word-candidate direction
        # aligns well with known antonym directions globally,
        # but the candidate is semantically related to word.
        suggestions = []

        for cand in self.vocab:
            if cand == word or cand in known:
                continue

            v_cand = self.model[cand]
            diff = v_word - v_cand
            diff_norm = norm(diff)
            if diff_norm < 0.5:  # Too similar
                continue

            diff_normalized = diff / diff_norm

            # Antonym-like direction?
            antonym_score = np.max(np.abs(self.D @ diff_normalized))

            # Semantic relatedness (cosine similarity)
            cosine = np.dot(v_word, v_cand) / (norm(v_word) * norm(v_cand))

            # Good candidates: high antonym score + moderate relatedness
            if antonym_score > 0.5 and 0.2 < cosine < 0.7:
                suggestions.append({
                    'word': cand,
                    'antonym_score': float(antonym_score),
                    'relatedness': float(cosine),
                    'combined_score': float(antonym_score * (1 - abs(cosine - 0.4)))
                })

        suggestions.sort(key=lambda x: x['combined_score'], reverse=True)
        return suggestions[:20]


def explore_light_axes():
    """Detailed exploration of 'light' antonym axes."""
    print("=" * 70)
    print("EXPLORING NEW ANTONYM AXES FOR 'light'")
    print("=" * 70)

    # Load
    antonym_pairs = extract_antonym_pairs()
    loader = Word2VecLoader()
    model = loader.load_glove(100)

    explorer = NewAxisExplorer(model, antonym_pairs)

    word = "light"
    known = explorer.known_antonyms.get(word, set())
    print(f"\nWord: {word}")
    print(f"Known antonyms: {known}")

    # Explore known axes
    print("\n" + "=" * 70)
    print("KNOWN AXES - DETAILED")
    print("=" * 70)

    for ant in sorted(known):
        if ant not in model:
            continue

        result = explorer.validate_axis(word, ant)
        print(f"\n{'─' * 60}")
        print(f"AXIS: {word} ↔ {ant}")
        print(f"{'─' * 60}")
        print(f"  Distance: {result['scores']['distance']:.2f}")
        print(f"  Cosine: {result['scores']['cosine_similarity']:.3f}")
        print(f"  Max antonym alignment: {result['scores']['max_antonym_alignment']:.3f}")

        axis_words = result['axis_words']
        print(f"\n  {ant}側:")
        for p in axis_words.get(f'{ant}_pole', [])[:8]:
            print(f"    {p['word']:15s} (pos: {p['normalized_position']:+.2f})")

        print(f"\n  {word}側:")
        for p in axis_words.get(f'{word}_pole', [])[:8]:
            print(f"    {p['word']:15s} (pos: {p['normalized_position']:+.2f})")

    # Discover new axes
    print("\n\n" + "=" * 70)
    print("DISCOVERING NEW AXES")
    print("=" * 70)

    new_axes = explorer.discover_new_axes_for_word(word, top_n=30)

    # Filter to interesting candidates
    interesting = [
        ax for ax in new_axes
        if ax['candidate'] in ['serious', 'severe', 'intense', 'dense',
                                'complex', 'rich', 'strong', 'thick',
                                'mild', 'gentle', 'soft', 'faint', 'dim',
                                'trivial', 'minor', 'slight']
    ]

    # Add top novel ones
    for ax in new_axes[:10]:
        if ax not in interesting:
            interesting.append(ax)

    print(f"\n候補数: {len(new_axes)}")
    print(f"詳細分析: {len(interesting)}")

    for ax in interesting[:15]:
        cand = ax['candidate']
        print(f"\n{'─' * 60}")
        print(f"POTENTIAL: {word} ↔ {cand}")
        print(f"  Novelty: {ax['novelty']:.3f}")
        print(f"  Global antonym score: {ax['global_score']:.3f}")

        if ax['known_alignments']:
            print(f"  Alignments with known: ", end="")
            for k, v in ax['known_alignments'].items():
                print(f"{k}({v:.2f}) ", end="")
            print()

        # Validate and show axis words
        validation = explorer.validate_axis(word, cand)
        axis_words = validation['axis_words']

        print(f"\n  {cand}側の単語:")
        for p in axis_words.get(f'{cand}_pole', [])[:5]:
            print(f"    {p['word']:15s}")

        print(f"\n  {word}側の単語:")
        for p in axis_words.get(f'{word}_pole', [])[:5]:
            print(f"    {p['word']:15s}")

    # Suggest missing antonyms
    print("\n\n" + "=" * 70)
    print("SUGGESTED MISSING ANTONYMS")
    print("=" * 70)

    suggestions = explorer.suggest_missing_antonyms(word)

    print(f"\nWordNetにない可能性のある対義語:")
    for s in suggestions[:15]:
        known_marker = "(KNOWN)" if s['word'] in known else ""
        print(f"  {word} ↔ {s['word']:15s} "
              f"(antonym: {s['antonym_score']:.2f}, "
              f"related: {s['relatedness']:.2f}) {known_marker}")

    return explorer


def interactive_explore():
    """Interactive exploration mode."""
    antonym_pairs = extract_antonym_pairs()
    loader = Word2VecLoader()
    model = loader.load_glove(100)
    explorer = NewAxisExplorer(model, antonym_pairs)

    print("\n" + "=" * 70)
    print("INTERACTIVE AXIS EXPLORER")
    print("=" * 70)
    print("\nCommands:")
    print("  axis <word1> <word2>  - Analyze axis between two words")
    print("  new <word>            - Find new axes for word")
    print("  suggest <word>        - Suggest missing antonyms")
    print("  quit                  - Exit")

    while True:
        try:
            cmd = input("\n> ").strip().split()
            if not cmd:
                continue

            if cmd[0] == 'quit':
                break

            elif cmd[0] == 'axis' and len(cmd) >= 3:
                word1, word2 = cmd[1], cmd[2]
                result = explorer.validate_axis(word1, word2)
                print(f"\nAxis: {word1} ↔ {word2}")
                print(f"Valid: {result['valid']}")
                print(f"Scores: {result['scores']}")

                axis_words = result['axis_words']
                print(f"\n{word2}側: {[p['word'] for p in axis_words.get(f'{word2}_pole', [])[:5]]}")
                print(f"{word1}側: {[p['word'] for p in axis_words.get(f'{word1}_pole', [])[:5]]}")

            elif cmd[0] == 'new' and len(cmd) >= 2:
                word = cmd[1]
                axes = explorer.discover_new_axes_for_word(word, top_n=10)
                print(f"\nNew axes for '{word}':")
                for ax in axes:
                    print(f"  {word} ↔ {ax['candidate']:15s} (novelty: {ax['novelty']:.2f})")

            elif cmd[0] == 'suggest' and len(cmd) >= 2:
                word = cmd[1]
                suggestions = explorer.suggest_missing_antonyms(word)
                print(f"\nSuggested antonyms for '{word}':")
                for s in suggestions[:10]:
                    print(f"  {s['word']:15s} (score: {s['combined_score']:.2f})")

            else:
                print("Unknown command. Try: axis, new, suggest, quit")

        except EOFError:
            break
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '-i':
        interactive_explore()
    else:
        explore_light_axes()
