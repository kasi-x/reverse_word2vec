"""
Train a Japanese Word2Vec model from EPWING dictionary text.

Uses dictionary definitions from daijirin (302K entries) and meikyo (61K entries)
as training corpus, tokenized with vibrato/ipadic.
"""
import json
import re
import os
import time

import vibrato
import zstandard
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

DAIJIRIN_PATH = "/home/user/dev/shinar-backup-20260119/epwing_parser/data/daijirin.json"
MEIKYO_PATH = "/home/user/dev/shinar-backup-20260119/epwing_parser/data/meikyo.json"
DICT_PATH = "/home/user/dev/shinar/nlp/furigana/dictionaries/ipadic-mecab-2_7_0/system.dic.zst"
OUTPUT_DIR = os.path.expanduser("~/gensim-data")
MODEL_PATH = os.path.join(OUTPUT_DIR, "ja-dict-w2v-300.kv")
CORPUS_PATH = os.path.join(OUTPUT_DIR, "ja-dict-corpus.txt")


def clean_text(text: str) -> str:
    """Clean EPWING text by removing all tags and markers."""
    text = re.sub(r"<begin_narrow>.*?<end_narrow>", "", text)
    text = re.sub(r"<begin_superscript>.*?<end_superscript>", "", text)
    text = re.sub(r"<begin_decoration>.*?<end_decoration>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    # Remove reference markers, special symbols
    text = re.sub(r"[⇔→＝（）()「」【】〔〕《》〈〉『』─―‐・]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_corpus_sentences(json_path: str, tokenizer) -> list[list[str]]:
    """Extract tokenized sentences from EPWING dictionary."""
    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data["subbooks"][0]["entries"]
    sentences = []
    ja_pattern = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")

    for entry in entries:
        text = entry.get("text", "")
        clean = clean_text(text)
        if not clean or len(clean) < 10:
            continue

        # Split on period-like boundaries
        for segment in re.split(r"[。\n]+", clean):
            segment = segment.strip()
            if len(segment) < 5 or not ja_pattern.search(segment):
                continue

            # Tokenize with vibrato
            tokens = tokenizer.tokenize(segment)
            words = []
            for token in tokens:
                surface = token.surface()
                # Only keep content words (not just punctuation/symbols)
                if ja_pattern.search(surface) and len(surface) >= 1:
                    words.append(surface)

            if len(words) >= 3:
                sentences.append(words)

    return sentences


def main():
    print("=" * 70)
    print("TRAINING JAPANESE WORD2VEC FROM EPWING DICTIONARIES")
    print("=" * 70)

    # Initialize tokenizer
    print("\n[1/4] Loading tokenizer...")
    with open(DICT_PATH, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        dict_data = reader.read()
        reader.close()
    tokenizer = vibrato.Vibrato(dict_data)
    print(f"  Vibrato tokenizer loaded (ipadic)")

    # Extract corpus
    print("\n[2/4] Extracting corpus from dictionaries...")
    t0 = time.time()
    sentences_daijirin = extract_corpus_sentences(DAIJIRIN_PATH, tokenizer)
    print(f"  Daijirin: {len(sentences_daijirin)} sentences")
    sentences_meikyo = extract_corpus_sentences(MEIKYO_PATH, tokenizer)
    print(f"  Meikyo: {len(sentences_meikyo)} sentences")

    all_sentences = sentences_daijirin + sentences_meikyo
    print(f"  Total: {len(all_sentences)} sentences ({time.time()-t0:.1f}s)")

    # Count unique tokens
    vocab = set()
    total_tokens = 0
    for sent in all_sentences:
        vocab.update(sent)
        total_tokens += len(sent)
    print(f"  Unique tokens: {len(vocab)}")
    print(f"  Total tokens: {total_tokens}")

    # Save corpus for inspection
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for sent in all_sentences:
            f.write(" ".join(sent) + "\n")
    print(f"  Corpus saved to {CORPUS_PATH}")

    # Train Word2Vec
    print("\n[3/4] Training Word2Vec (dim=300, window=5, min_count=3)...")
    t0 = time.time()
    model = Word2Vec(
        sentences=all_sentences,
        vector_size=300,
        window=5,
        min_count=3,
        workers=4,
        epochs=15,
        sg=1,  # Skip-gram
    )
    train_time = time.time() - t0
    print(f"  Training complete in {train_time:.1f}s")
    print(f"  Vocabulary: {len(model.wv)} words")

    # Save as KeyedVectors
    print("\n[4/4] Saving model...")
    model.wv.save(MODEL_PATH)
    file_size = os.path.getsize(MODEL_PATH)
    print(f"  Saved to {MODEL_PATH} ({file_size/1e6:.1f}MB)")

    # Quick test
    print("\n[Test] Nearest words:")
    test_words = ["大きい", "男", "買う", "平和", "明るい"]
    for word in test_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            sim_str = ", ".join([f"{w}({s:.2f})" for w, s in similar])
            print(f"  {word}: {sim_str}")
        else:
            print(f"  {word}: not in vocabulary")


if __name__ == "__main__":
    main()
