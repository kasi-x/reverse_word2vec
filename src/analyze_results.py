"""
Interactive analysis of antonym space with better filtering.
"""
import sys
sys.path.insert(0, '.')

import re
import numpy as np
from antonym_loader import extract_antonym_pairs, group_antonyms_by_word
from word2vec_loader import Word2VecLoader
from antonym_space import create_antonym_space_from_model


def is_valid_word(word: str) -> bool:
    """Check if word is a valid English word (not noise)."""
    # Must be alphabetic
    if not word.isalpha():
        return False
    # Must be lowercase
    if word != word.lower():
        return False
    # Must be 3+ characters
    if len(word) < 3:
        return False
    # Skip very long words (likely noise)
    if len(word) > 20:
        return False
    return True


def main():
    print("=" * 60)
    print("ANTONYM SPACE ANALYSIS (Filtered)")
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

    # Filter pairs and build space
    valid_pairs = [
        (w1, w2) for w1, w2 in antonym_pairs
        if loader.has_word(w1) and loader.has_word(w2)
    ]
    print(f"  Valid pairs: {len(valid_pairs)}")

    print("\n[3/3] Building antonym space...")
    space = create_antonym_space_from_model(
        model, valid_pairs, method='greedy', max_axes=100, min_orthogonality=0.1
    )
    print(f"  Created {len(space.axes)} axes")

    # Show axes
    print("\n" + "=" * 60)
    print("TOP ANTONYM AXES")
    print("=" * 60)
    for i, axis in enumerate(space.axes[:20]):
        print(f"  {i:2d}. {axis.negative_word:18s} <-> {axis.positive_word:18s} (d={axis.separation:.2f})")

    # Find near-zero words (filtered)
    print("\n" + "=" * 60)
    print("WORDS NEAR ZERO (Semantically Neutral)")
    print("=" * 60)

    # Filter vocabulary to valid words
    valid_words = [w for w in model.key_to_index.keys() if is_valid_word(w)]
    print(f"  Analyzing {len(valid_words)} valid words...")

    near_zero = space.find_near_zero_words(words=valid_words, top_n=50)

    print("\n  Top 30 semantically neutral words:")
    for word, dist, _ in near_zero[:30]:
        print(f"    {word:20s}: {dist:.4f}")

    # Discover antonyms for words without known antonyms
    print("\n" + "=" * 60)
    print("DISCOVERED POTENTIAL ANTONYMS")
    print("=" * 60)

    # Find common words without known antonyms
    common_words = [w for w in valid_words[:20000] if w not in known_antonyms]

    print(f"\n  Testing words without known antonyms...")

    # Test specific interesting words
    test_words = [
        "technology", "computer", "science", "music", "art", "democracy",
        "philosophy", "history", "future", "nature", "culture", "economy",
        "society", "education", "health", "food", "water", "fire", "earth",
        "time", "space", "mind", "body", "heart", "soul", "spirit",
        "money", "power", "freedom", "justice", "truth", "beauty"
    ]

    for word in test_words:
        if not loader.has_word(word):
            continue
        potentials = space.find_potential_antonyms(
            word,
            candidates=valid_words[:50000],
            top_n=5
        )
        if potentials:
            # Filter to valid words
            filtered = [(w, s) for w, s in potentials if is_valid_word(w)][:3]
            if filtered:
                ant_str = ", ".join([f"{a}({s:.2f})" for a, s in filtered])
                print(f"    {word:15s}: {ant_str}")

    # Analyze some known antonyms to verify
    print("\n" + "=" * 60)
    print("VERIFICATION: Known Antonym Pairs")
    print("=" * 60)

    test_pairs = [
        ("good", "bad"), ("hot", "cold"), ("love", "hate"),
        ("happy", "sad"), ("light", "dark"), ("big", "small"),
        ("fast", "slow"), ("young", "old"), ("rich", "poor"),
        ("strong", "weak"), ("hard", "soft"), ("open", "close")
    ]

    print("\n  Checking if known antonyms appear in discovered lists:")
    for w1, w2 in test_pairs:
        if not (loader.has_word(w1) and loader.has_word(w2)):
            continue

        # Find potential antonyms for w1
        potentials = space.find_potential_antonyms(
            w1, candidates=valid_words[:50000], top_n=20
        )
        potential_words = [p[0] for p in potentials]

        if w2 in potential_words:
            rank = potential_words.index(w2) + 1
            print(f"    {w1} -> {w2}: Found at rank {rank}")
        else:
            # Show what was found instead
            top3 = [p[0] for p in potentials[:3]]
            print(f"    {w1} -> {w2}: NOT in top 20 (top 3: {', '.join(top3)})")


if __name__ == "__main__":
    main()
