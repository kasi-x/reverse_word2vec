"""
Antonym data loader using NLTK's WordNet interface.
"""
import os
from typing import List, Tuple, Set
from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn


def ensure_wordnet_downloaded():
    """Download WordNet if not already present."""
    try:
        wn.synsets('test')
    except LookupError:
        print("Downloading WordNet...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')  # Open Multilingual Wordnet


def extract_antonym_pairs() -> List[Tuple[str, str]]:
    """
    Extract all antonym pairs from WordNet.

    Returns:
        List of (word1, word2) tuples representing antonym pairs.
    """
    ensure_wordnet_downloaded()

    antonym_pairs: Set[Tuple[str, str]] = set()

    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            antonyms = lemma.antonyms()
            if antonyms:
                word1 = lemma.name().lower().replace('_', ' ')
                for ant in antonyms:
                    word2 = ant.name().lower().replace('_', ' ')
                    # Normalize pair order to avoid duplicates
                    pair = tuple(sorted([word1, word2]))
                    antonym_pairs.add(pair)

    return list(antonym_pairs)


def extract_antonym_pairs_by_pos(pos: str = None) -> List[Tuple[str, str]]:
    """
    Extract antonym pairs filtered by part of speech.

    Args:
        pos: Part of speech filter ('n'=noun, 'v'=verb, 'a'=adjective, 'r'=adverb)
             None means all POS.

    Returns:
        List of (word1, word2) tuples.
    """
    ensure_wordnet_downloaded()

    antonym_pairs: Set[Tuple[str, str]] = set()

    synsets = wn.all_synsets(pos=pos) if pos else wn.all_synsets()

    for synset in synsets:
        for lemma in synset.lemmas():
            antonyms = lemma.antonyms()
            if antonyms:
                word1 = lemma.name().lower().replace('_', ' ')
                for ant in antonyms:
                    word2 = ant.name().lower().replace('_', ' ')
                    pair = tuple(sorted([word1, word2]))
                    antonym_pairs.add(pair)

    return list(antonym_pairs)


def group_antonyms_by_word() -> dict:
    """
    Create a dictionary mapping each word to its antonyms.

    Returns:
        Dict mapping word -> set of antonyms.
    """
    ensure_wordnet_downloaded()

    antonym_map = defaultdict(set)

    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            antonyms = lemma.antonyms()
            if antonyms:
                word = lemma.name().lower().replace('_', ' ')
                for ant in antonyms:
                    ant_word = ant.name().lower().replace('_', ' ')
                    antonym_map[word].add(ant_word)
                    antonym_map[ant_word].add(word)

    return dict(antonym_map)


def save_antonym_pairs(pairs: List[Tuple[str, str]], filepath: str):
    """Save antonym pairs to a file."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for w1, w2 in pairs:
            f.write(f"{w1}\t{w2}\n")


def load_antonym_pairs(filepath: str) -> List[Tuple[str, str]]:
    """Load antonym pairs from a file."""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


if __name__ == "__main__":
    # Test extraction
    pairs = extract_antonym_pairs()
    print(f"Found {len(pairs)} antonym pairs")
    print("Sample pairs:")
    for p in pairs[:10]:
        print(f"  {p[0]} <-> {p[1]}")
