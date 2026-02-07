"""
Japanese antonym pair extraction from EPWING dictionaries (大辞林, 明鏡).

Extracts antonym relationships marked with ⇔ in dictionary entry text.
"""
import json
import os
import re
from typing import List, Tuple, Set, Optional
from collections import defaultdict


# Default EPWING dictionary paths
DAIJIRIN_PATH = "/home/user/dev/shinar-backup-20260119/epwing_parser/data/daijirin.json"
MEIKYO_PATH = "/home/user/dev/shinar-backup-20260119/epwing_parser/data/meikyo.json"

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "japanese_antonym_pairs.json")

# Regex for Japanese characters (hiragana, katakana, kanji, prolonged sound mark)
_JA_CHAR = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3005\u30FC]")
# Regex for non-Japanese noise (pure ASCII, symbols, etc.)
_NOISE = re.compile(r"^[A-Za-z0-9\s\.\-_,;:!?]+$")


def _is_valid_japanese_word(word: str) -> bool:
    """Check if a word is a valid Japanese word for antonym analysis."""
    if not word or len(word) < 1:
        return False
    # Must contain at least one Japanese character
    if not _JA_CHAR.search(word):
        return False
    # Reject if starts with special chars
    if word[0] in "―〔〈《「（(・‐─":
        return False
    # Reject if too long (likely a phrase, not a word)
    if len(word) > 15:
        return False
    return True


def _clean_tags(s: str) -> str:
    """Remove all XML-like tags and {{...}} markers."""
    s = re.sub(r"<begin_narrow>.*?<end_narrow>", "", s)
    s = re.sub(r"<begin_superscript>.*?<end_superscript>", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\{\{[^}]+\}\}", "", s)
    return s.strip()


def _extract_headword_daijirin(heading: str) -> Optional[str]:
    """Extract the main word from a daijirin heading.

    Headings look like:
    - 'あい‐じつ【愛日】─ヂャウ'  -> '愛日'
    - 'アーティフィシャル{{w_44666}}<begin_narrow>artificial<end_narrow>{{w_44667}}'  -> 'アーティフィシャル'
    - 'アール{{w_44666}}<begin_narrow>R </narrow>...'  -> 'アール'
    """
    clean = _clean_tags(heading)

    # If contains 【kanji】, use kanji form
    kanji_match = re.search(r"【(.+?)】", clean)
    if kanji_match:
        word = kanji_match.group(1)
        word = re.sub(r"[▼▽]", "", word)
        # Remove variant separators like ・
        if "・" in word:
            word = word.split("・")[0]
        word = word.strip()
        # Validate: only Japanese chars
        if re.fullmatch(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3005\u30FC]+", word):
            return word
        return None

    # Otherwise use the kana/katakana heading
    # Remove romanization and symbols
    clean = re.sub(r"[a-zA-Z\-]+", "", clean)
    clean = re.sub(r"[‐・─―\s]", "", clean)
    # Extract leading Japanese chars
    m = re.match(r"([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3005\u30FC]+)", clean)
    if m:
        return m.group(1)
    return None


def _extract_headword_meikyo(heading: str) -> Optional[str]:
    """Extract the main word from a meikyo heading.

    Headings look like:
    - 'アーバン<begin_narrow>[urban]<end_narrow>'  -> 'アーバン'
    - 'あい‐じょう【愛嬢】─ヂャウ'  -> '愛嬢'
    - 'あ・う【会う・遭う・▼逢う・▽遇う】アフ'  -> '会う'
    """
    clean = _clean_tags(heading)

    # If contains 【kanji】, use kanji form
    kanji_match = re.search(r"【(.+?)】", clean)
    if kanji_match:
        word = kanji_match.group(1)
        word = re.sub(r"[▼▽]", "", word)
        if "・" in word:
            word = word.split("・")[0]
        word = word.strip()
        if re.fullmatch(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3005\u30FC]+", word):
            return word
        return None

    # Use the kana/katakana heading
    clean = re.sub(r"\[.*?\]", "", clean)
    clean = re.sub(r"[a-zA-Z\-]+", "", clean)
    clean = re.sub(r"[‐・─―\s]", "", clean)
    m = re.match(r"([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3005\u30FC]+)", clean)
    if m:
        return m.group(1)
    return None


def _extract_antonym_word_daijirin(text: str, pos: int) -> Optional[str]:
    """Extract the antonym word from text following ⇔ in daijirin.

    In daijirin, antonyms appear as:
    <begin_reference>⇔WORD<end_reference page=... offset=...>
    The word is contained between ⇔ and <end_reference>.
    """
    after = text[pos:pos + 300]

    # In daijirin, antonym refs are inside <begin_reference>⇔WORD<end_reference>
    # Find the end_reference boundary
    end_ref = re.search(r"<end_reference", after)
    if end_ref:
        segment = after[:end_ref.start()]
    else:
        # Fallback: take until next tag or newline
        segment_match = re.match(r"([^<\n]+)", after)
        segment = segment_match.group(1) if segment_match else after[:50]

    # Remove inline narrow/superscript annotations
    clean = re.sub(r"<begin_narrow>(.*?)<end_narrow>", "", segment)
    clean = re.sub(r"<begin_superscript>(.*?)<end_superscript>", "", clean)
    clean = re.sub(r"<[^>]+>", "", clean)
    clean = re.sub(r"\{\{[^}]+\}\}", "", clean)
    clean = clean.strip()

    # Get first word-like token (Japanese chars only)
    m = re.match(r"([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3005\u30FC]+)", clean)
    if m:
        word = m.group(1)
        return word
    return None


def _extract_antonym_word_meikyo(text: str, pos: int) -> Optional[str]:
    """Extract the antonym word from text following ⇔ in meikyo.

    In meikyo, antonyms appear inline as:
    ⇔WORD (possibly followed by definition text without separator)

    Since meikyo doesn't use reference tags, we use heuristics:
    - Check for common boundaries: ◇ ◆ decoration tags, newlines, 「
    - If the match is a raw Japanese string, limit to reasonable word length
    """
    after = text[pos:pos + 300]

    # Check if there's a tag/marker boundary close by
    # Meikyo often has: ⇔WORD<begin_decoration>◇<end_decoration>
    # or ⇔WORD<newline> or ⇔WORD「
    # First, try to find the word before the first boundary marker
    boundary = re.search(
        r"<begin_decoration>|<newline>|<set_indent>|\n|◇|◆|「|（|\(|〔|→|⇔|〘",
        after,
    )
    if boundary:
        segment = after[:boundary.start()]
    else:
        segment = after[:50]

    # Remove superscript readings (furigana)
    clean = re.sub(r"<begin_superscript>.*?<end_superscript>", "", segment)
    clean = re.sub(r"<begin_narrow>.*?<end_narrow>", "", clean)
    clean = re.sub(r"<[^>]+>", "", clean)
    clean = re.sub(r"\{\{[^}]+\}\}", "", clean)
    clean = clean.strip()

    # Extract only Japanese characters (strict)
    m = re.match(r"([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3005\u30FC]+)", clean)
    if m:
        word = m.group(1)
        # Sanity check: dictionary antonyms are typically 1-8 chars
        if len(word) <= 10:
            return word
    return None


def extract_antonyms_from_daijirin(json_path: str = DAIJIRIN_PATH) -> List[Tuple[str, str]]:
    """Extract antonym pairs from daijirin.json using ⇔ markers.

    Returns:
        List of (word1, word2) tuples, deduplicated and sorted.
    """
    print(f"Loading daijirin from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data["subbooks"][0]["entries"]
    pairs: Set[Tuple[str, str]] = set()

    for entry in entries:
        heading = entry["heading"]
        text = entry.get("text", "")

        headword = _extract_headword_daijirin(heading)
        if not _is_valid_japanese_word(headword):
            continue

        for m in re.finditer(r"⇔", text):
            antonym = _extract_antonym_word_daijirin(text, m.end())
            if not _is_valid_japanese_word(antonym) or antonym == headword:
                continue

            pair = tuple(sorted([headword, antonym]))
            pairs.add(pair)

    print(f"  Extracted {len(pairs)} unique pairs from daijirin")
    return sorted(pairs)


def extract_antonyms_from_meikyo(json_path: str = MEIKYO_PATH) -> List[Tuple[str, str]]:
    """Extract antonym pairs from meikyo.json using ⇔ markers.

    Returns:
        List of (word1, word2) tuples, deduplicated and sorted.
    """
    print(f"Loading meikyo from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data["subbooks"][0]["entries"]
    pairs: Set[Tuple[str, str]] = set()

    for entry in entries:
        heading = entry["heading"]
        text = entry.get("text", "")

        headword = _extract_headword_meikyo(heading)
        if not _is_valid_japanese_word(headword):
            continue

        for m in re.finditer(r"⇔", text):
            antonym = _extract_antonym_word_meikyo(text, m.end())
            if not _is_valid_japanese_word(antonym) or antonym == headword:
                continue

            pair = tuple(sorted([headword, antonym]))
            pairs.add(pair)

    print(f"  Extracted {len(pairs)} unique pairs from meikyo")
    return sorted(pairs)


def extract_all_antonym_pairs(
    daijirin_path: str = DAIJIRIN_PATH,
    meikyo_path: str = MEIKYO_PATH,
) -> List[Tuple[str, str]]:
    """Extract and merge antonym pairs from both dictionaries.

    Returns:
        Deduplicated, sorted list of (word1, word2) tuples.
    """
    daijirin_pairs = set(extract_antonyms_from_daijirin(daijirin_path))
    meikyo_pairs = set(extract_antonyms_from_meikyo(meikyo_path))

    all_pairs = daijirin_pairs | meikyo_pairs
    overlap = daijirin_pairs & meikyo_pairs

    print(f"\nMerge statistics:")
    print(f"  Daijirin only: {len(daijirin_pairs - meikyo_pairs)}")
    print(f"  Meikyo only:   {len(meikyo_pairs - daijirin_pairs)}")
    print(f"  Overlap:       {len(overlap)}")
    print(f"  Total unique:  {len(all_pairs)}")

    return sorted(all_pairs)


def normalize_japanese_word(word: str) -> str:
    """Normalize a Japanese word for matching against word vectors.

    - Remove okurigana variation marks
    - Normalize long vowel marks
    - Strip whitespace
    """
    word = word.strip()
    word = re.sub(r"[▼▽]", "", word)
    # Normalize katakana middle dot
    word = word.replace("・", "")
    # Normalize dashes
    word = word.replace("‐", "").replace("─", "").replace("―", "")
    return word


def filter_valid_pairs(
    pairs: List[Tuple[str, str]],
    model,
) -> List[Tuple[str, str]]:
    """Filter pairs to only those where both words exist in the word vector model.

    Args:
        pairs: List of (word1, word2) antonym pairs.
        model: gensim KeyedVectors model.

    Returns:
        Filtered list of pairs.
    """
    valid = []
    for w1, w2 in pairs:
        nw1 = normalize_japanese_word(w1)
        nw2 = normalize_japanese_word(w2)
        if nw1 in model and nw2 in model:
            valid.append((nw1, nw2))
    return valid


def save_pairs(pairs: List[Tuple[str, str]], output_path: str = OUTPUT_PATH):
    """Save antonym pairs to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "description": "Japanese antonym pairs extracted from EPWING dictionaries (daijirin + meikyo)",
        "count": len(pairs),
        "pairs": [[w1, w2] for w1, w2 in pairs],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(pairs)} pairs to {output_path}")


def load_pairs(input_path: str = OUTPUT_PATH) -> List[Tuple[str, str]]:
    """Load antonym pairs from JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(p[0], p[1]) for p in data["pairs"]]


if __name__ == "__main__":
    pairs = extract_all_antonym_pairs()

    # Show samples
    print(f"\nSample pairs (first 30):")
    for w1, w2 in pairs[:30]:
        print(f"  {w1} ⇔ {w2}")

    # Save
    save_pairs(pairs)

    # Category analysis
    print(f"\nPair length distribution:")
    lengths = defaultdict(int)
    for w1, w2 in pairs:
        l = max(len(w1), len(w2))
        bucket = f"{l}" if l <= 5 else "6+"
        lengths[bucket] += 1
    for k in sorted(lengths.keys()):
        print(f"  max len {k}: {lengths[k]} pairs")
