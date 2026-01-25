"""
Improved antonym space analysis using direct projection onto antonym directions.

Instead of orthogonalizing, we measure how much each word aligns with
the difference vectors of known antonym pairs.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from numpy.linalg import norm, svd
from collections import defaultdict
from tqdm import tqdm

from antonym_loader import extract_antonym_pairs, group_antonyms_by_word
from word2vec_loader import Word2VecLoader


def is_valid_word(word: str) -> bool:
    """Check if word is a valid English word (not noise)."""
    if not word.isalpha():
        return False
    if word != word.lower():
        return False
    if len(word) < 3 or len(word) > 15:
        return False
    return True


class DirectAntonymAnalysis:
    """
    Analyze words by direct projection onto antonym difference vectors.

    For each antonym pair (w+, w-), the difference vector d = v(w+) - v(w-)
    represents the semantic direction of that opposition.

    A word is "neutral" if it has low projection onto all these directions.
    A potential antonym for word w is one with opposite projections.
    """

    def __init__(self, model, antonym_pairs):
        self.model = model
        self.dimension = model.vector_size

        # Compute all valid antonym difference vectors
        self.directions = []
        self.pair_labels = []

        print("Computing antonym direction vectors...")
        for w1, w2 in tqdm(antonym_pairs):
            if w1 in model and w2 in model:
                v1, v2 = model[w1], model[w2]
                d = v1 - v2
                d_norm = norm(d)
                if d_norm > 1e-6:
                    self.directions.append(d / d_norm)
                    self.pair_labels.append((w1, w2))

        self.D = np.array(self.directions)  # n_pairs x dim
        print(f"  {len(self.directions)} valid antonym directions")

        # SVD to find principal antonym directions
        print("Computing principal antonym directions via SVD...")
        U, S, Vh = svd(self.D, full_matrices=False)
        self.principal_directions = Vh  # k x dim
        self.singular_values = S

        # Keep top directions that explain 90% variance
        total_var = np.sum(S**2)
        cumvar = np.cumsum(S**2) / total_var
        self.n_principal = np.searchsorted(cumvar, 0.90) + 1
        print(f"  {self.n_principal} principal directions explain 90% variance")

    def project_word(self, word):
        """Project word onto all antonym directions."""
        if word not in self.model:
            return None
        v = self.model[word]
        return self.D @ v  # projection onto each antonym direction

    def project_word_principal(self, word):
        """Project word onto principal antonym directions."""
        if word not in self.model:
            return None
        v = self.model[word]
        return self.principal_directions[:self.n_principal] @ v

    def find_neutral_words(self, words, top_n=50):
        """Find words with lowest total projection magnitude."""
        results = []
        for word in tqdm(words, desc="Finding neutrals"):
            proj = self.project_word_principal(word)
            if proj is not None:
                # Use L2 norm of projections
                neutrality = norm(proj)
                results.append((word, neutrality))

        results.sort(key=lambda x: x[1])
        return results[:top_n]

    def find_antonyms_by_opposition(self, word, candidates, top_n=10):
        """
        Find words with opposite projection patterns.

        Uses cosine similarity of projection vectors - negative similarity
        indicates opposite positioning on antonym axes.
        """
        proj = self.project_word_principal(word)
        if proj is None:
            return []

        proj_norm = norm(proj)
        if proj_norm < 1e-6:
            return []

        results = []
        for cand in candidates:
            if cand == word:
                continue
            cand_proj = self.project_word_principal(cand)
            if cand_proj is None:
                continue

            cand_norm = norm(cand_proj)
            if cand_norm < 1e-6:
                continue

            # Negative cosine similarity = opposition
            similarity = np.dot(proj, cand_proj) / (proj_norm * cand_norm)
            results.append((cand, -similarity))  # Negate so higher = more opposite

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def find_antonyms_by_vector_difference(self, word, candidates, top_n=10):
        """
        Find potential antonyms by looking for words where w - candidate
        is similar to known antonym directions.
        """
        if word not in self.model:
            return []

        v = self.model[word]
        results = []

        for cand in candidates:
            if cand == word:
                continue
            if cand not in self.model:
                continue

            # Compute difference vector
            diff = v - self.model[cand]
            diff_norm = norm(diff)
            if diff_norm < 1e-6:
                continue
            diff_normalized = diff / diff_norm

            # How similar is this difference to known antonym directions?
            similarities = self.D @ diff_normalized
            max_sim = np.max(np.abs(similarities))

            results.append((cand, max_sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]


def main():
    print("=" * 60)
    print("IMPROVED ANTONYM SPACE ANALYSIS")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...")
    antonym_pairs = extract_antonym_pairs()
    known_antonyms = group_antonyms_by_word()
    print(f"  Loaded {len(antonym_pairs)} antonym pairs")

    # Load model
    print("\n[2/3] Loading model...")
    loader = Word2VecLoader()
    model = loader.load_glove(100)

    # Build analysis
    print("\n[3/3] Building antonym analysis...")
    analysis = DirectAntonymAnalysis(model, antonym_pairs)

    # Get valid words
    valid_words = [w for w in model.key_to_index.keys() if is_valid_word(w)]
    print(f"  {len(valid_words)} valid words in vocabulary")

    # Find neutral words
    print("\n" + "=" * 60)
    print("SEMANTICALLY NEUTRAL WORDS")
    print("(Low projection on all antonym directions)")
    print("=" * 60)

    neutrals = analysis.find_neutral_words(valid_words[:50000], top_n=50)
    print("\nTop 30 neutral words:")
    for word, score in neutrals[:30]:
        print(f"  {word:20s}: {score:.4f}")

    # Test antonym discovery
    print("\n" + "=" * 60)
    print("ANTONYM DISCOVERY (Vector Difference Method)")
    print("=" * 60)

    test_words = [
        "happy", "good", "love", "light", "hot", "big", "fast",
        "young", "rich", "strong", "hard", "open", "high", "long"
    ]

    print("\nFinding potential antonyms for words with known antonyms:")
    for word in test_words:
        if word not in model:
            continue

        known = known_antonyms.get(word, set())
        potentials = analysis.find_antonyms_by_vector_difference(
            word, valid_words[:50000], top_n=10
        )

        # Show results
        pot_str = ", ".join([f"{w}({s:.2f})" for w, s in potentials[:5]])
        found_known = [w for w, s in potentials if w in known]

        print(f"\n  {word}:")
        print(f"    Known antonyms: {', '.join(known) if known else 'none'}")
        print(f"    Top discovered: {pot_str}")
        if found_known:
            print(f"    ✓ Found known: {', '.join(found_known)}")

    # Test on words without known antonyms
    print("\n" + "=" * 60)
    print("DISCOVERY FOR WORDS WITHOUT KNOWN ANTONYMS")
    print("=" * 60)

    test_words_no_antonym = [
        "technology", "computer", "science", "music", "democracy",
        "philosophy", "history", "nature", "culture", "time", "space",
        "money", "power", "freedom", "truth", "beauty", "death", "life"
    ]

    print("\nPotential antonyms for abstract concepts:")
    for word in test_words_no_antonym:
        if word not in model:
            continue

        potentials = analysis.find_antonyms_by_vector_difference(
            word, valid_words[:50000], top_n=5
        )
        pot_str = ", ".join([f"{w}({s:.2f})" for w, s in potentials])
        print(f"  {word:15s}: {pot_str}")

    # Verify with opposition method
    print("\n" + "=" * 60)
    print("VERIFICATION: Opposition Method for Known Pairs")
    print("=" * 60)

    known_pairs = [
        ("good", "bad"), ("hot", "cold"), ("love", "hate"),
        ("happy", "sad"), ("light", "dark"), ("big", "small"),
        ("fast", "slow"), ("young", "old"), ("rich", "poor"),
        ("up", "down"), ("in", "out"), ("yes", "no")
    ]

    print("\nChecking if known antonyms appear in top results:")
    for w1, w2 in known_pairs:
        if w1 not in model or w2 not in model:
            continue

        potentials = analysis.find_antonyms_by_opposition(
            w1, valid_words[:50000], top_n=20
        )
        potential_words = [p[0] for p in potentials]

        if w2 in potential_words:
            rank = potential_words.index(w2) + 1
            print(f"  {w1:8s} -> {w2:8s}: ✓ Rank {rank}")
        else:
            top3 = ", ".join(potential_words[:3])
            print(f"  {w1:8s} -> {w2:8s}: ✗ Not in top 20 (top: {top3})")


if __name__ == "__main__":
    main()
