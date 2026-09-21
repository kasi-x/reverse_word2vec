"""
ConceptNet antonym loader.

Downloads and caches English antonym pairs from ConceptNet 5.
Two strategies:
  1. Stream the full assertions CSV (1GB) - most complete
  2. Query the REST API (rate-limited, but no large download)

Also includes WordNet synset-based expansion:
  If A antonym B, and C is a near-synonym of A (same synset),
  then (C, B) is also an antonym pair.
"""

from __future__ import annotations

import csv
import gzip
import io
import re
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn

# ConceptNet 5.7 full assertions (gzipped CSV, ~1GB)
CN_ASSERTIONS_URL = (
    "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
)
CACHE_PATH = Path("data/conceptnet_antonyms.tsv")

# Simple word pattern: lowercase letters only, 2+ chars
_WORD_RE = re.compile(r"^[a-z][a-z'-]*[a-z]$|^[a-z]{2,}$")


def _clean_concept(uri: str) -> str | None:
    """
    Convert /c/en/hot_dog to 'hot dog', return None if not English.
    """
    parts = uri.strip("/").split("/")
    if len(parts) < 3 or parts[1] != "en":
        return None
    word = parts[2].replace("_", " ").lower()
    # Only accept single-token words for our embedding vocab
    if " " in word:
        return None
    if not _WORD_RE.match(word):
        return None
    return word


def stream_conceptnet_antonyms() -> Iterator[tuple[str, str]]:
    """
    Stream English antonym pairs directly from ConceptNet S3.
    Filters for /r/Antonym with both ends in /c/en/.
    Does NOT save to disk — use download_conceptnet_antonyms() for caching.
    """
    print("Streaming ConceptNet assertions from S3...")
    print("(Filtering for /r/Antonym — this may take a few minutes)")

    req = urllib.request.Request(
        CN_ASSERTIONS_URL,
        headers={"User-Agent": "reverse_word2vec research project"},
    )
    seen: set[tuple[str, str]] = set()
    n_yielded = 0

    with urllib.request.urlopen(req) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            reader = csv.reader(io.TextIOWrapper(gz, encoding="utf-8"), delimiter="\t")
            for row in reader:
                if len(row) < 4:
                    continue
                # Row: assertion_uri, relation, start, end, json
                relation = row[1]
                if relation != "/r/Antonym":
                    continue
                w1 = _clean_concept(row[2])
                w2 = _clean_concept(row[3])
                if w1 is None or w2 is None or w1 == w2:
                    continue
                pair = tuple(sorted([w1, w2]))
                if pair in seen:
                    continue
                seen.add(pair)
                n_yielded += 1
                if n_yielded % 1000 == 0:
                    print(f"  {n_yielded} antonym pairs found so far...")
                yield pair

    print(f"Done. Total ConceptNet antonym pairs: {n_yielded}")


def download_conceptnet_antonyms(
    cache_path: str | Path = CACHE_PATH,
    force: bool = False,
) -> list[tuple[str, str]]:
    """
    Download and cache ConceptNet English antonym pairs.

    Args:
        cache_path: Where to save the TSV file.
        force: Re-download even if cache exists.

    Returns:
        List of (word1, word2) antonym pairs.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force:
        print(f"Loading cached ConceptNet antonyms from {cache_path}")
        pairs = []
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        print(f"  Loaded {len(pairs)} pairs")
        return pairs

    # Stream and save
    pairs = list(stream_conceptnet_antonyms())
    with open(cache_path, "w", encoding="utf-8") as f:
        for w1, w2 in pairs:
            f.write(f"{w1}\t{w2}\n")
    print(f"Saved {len(pairs)} pairs to {cache_path}")
    return pairs


def expand_via_wordnet_synsets(
    pairs: list[tuple[str, str]],
    max_expansion: int = 3,
) -> list[tuple[str, str]]:
    """
    Expand antonym pairs via WordNet synset membership.

    If (A, B) is an antonym pair and C is in the same synset as A,
    then (C, B) is also added. Likewise for B's synset-mates.

    Args:
        pairs: Base antonym pairs to expand.
        max_expansion: Max synset-mates to add per word.

    Returns:
        Expanded list of pairs (deduplicated).
    """
    try:
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    # Build word -> synset-mates map
    print("Building WordNet synset expansion...")
    synset_mates: dict[str, set[str]] = defaultdict(set)
    for synset in wn.all_synsets():
        lemma_names = [lemma.name().lower().replace("_", " ") for lemma in synset.lemmas()]
        # Only single-token names
        lemma_names = [w for w in lemma_names if " " not in w and _WORD_RE.match(w)]
        for w in lemma_names:
            for mate in lemma_names:
                if mate != w:
                    synset_mates[w].add(mate)

    expanded: set[tuple[str, str]] = set(pairs)
    base_set = set(pairs)

    for w1, w2 in pairs:
        # Expand via w1's synset-mates
        for mate in list(synset_mates.get(w1, set()))[:max_expansion]:
            candidate = tuple(sorted([mate, w2]))
            if candidate not in base_set:
                expanded.add(candidate)
        # Expand via w2's synset-mates
        for mate in list(synset_mates.get(w2, set()))[:max_expansion]:
            candidate = tuple(sorted([w1, mate]))
            if candidate not in base_set:
                expanded.add(candidate)

    result = list(expanded)
    print(f"Synset expansion: {len(pairs)} → {len(result)} pairs (+{len(result) - len(pairs)})")
    return result


def load_all_antonym_pairs(
    use_conceptnet: bool = True,
    use_wordnet_expansion: bool = True,
    conceptnet_cache: str | Path = CACHE_PATH,
    expansion_size: int = 3,
) -> list[tuple[str, str]]:
    """
    Load the combined antonym dataset:
      1. WordNet direct antonyms (via NLTK)
      2. ConceptNet antonyms (from cache or download)
      3. WordNet synset expansion (optional)

    Returns deduplicated list of (word1, word2) pairs.
    """
    from src.antonym_loader import extract_antonym_pairs

    all_pairs: set[tuple[str, str]] = set()

    # 1. WordNet
    print("Loading WordNet antonyms...")
    wn_pairs = extract_antonym_pairs()
    all_pairs.update(wn_pairs)
    print(f"  WordNet: {len(wn_pairs)} pairs")

    # 2. ConceptNet
    if use_conceptnet:
        cn_pairs = download_conceptnet_antonyms(conceptnet_cache)
        all_pairs.update(cn_pairs)
        print(f"  After ConceptNet: {len(all_pairs)} pairs total")

    # 3. WordNet synset expansion
    if use_wordnet_expansion:
        expanded = expand_via_wordnet_synsets(list(all_pairs), expansion_size)
        all_pairs.update(expanded)
        print(f"  After synset expansion: {len(all_pairs)} pairs total")

    return list(all_pairs)
