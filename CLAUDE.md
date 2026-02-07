# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project explores Word2Vec semantic spaces through antonym relationships. The core idea is to use known antonym pairs to define semantic directions, then analyze:
1. Which words lie near the zero vector (semantically "neutral" words)
2. Whether new antonym relationships can be discovered for words without known antonyms
3. Cross-lingual comparison of antonym vector spaces (English vs Japanese)

## Commands

```bash
# Setup (pixi)
pixi install

# Task runner (requires go-task)
task --list

# Download models
task download-models:en   # GloVe-100 (~130MB)
task download-models:ja   # fastText cc.ja.300 (~1.3GB)

# Extract Japanese antonym pairs from EPWING dictionaries
task extract:ja

# Train Japanese Word2Vec from dictionary text (alternative to fastText)
pixi run python src/train_japanese_w2v.py

# Run English analysis (exports JSON for comparison)
task run:en

# Run Japanese analysis
task run:ja

# Run cross-lingual comparison (requires EN + JA results)
task run:compare

# Run full pipeline (EN + JA + comparison)
task run:all

# Generate academic paper from results
task paper:generate
```

## Architecture

### Three Analysis Approaches (in order of effectiveness)

**1. Advanced Method** (`src/advanced_analysis.py`) - **Best Results**
- Clusters antonym directions into 12 semantic categories (valence, size, etc.)
- Bidirectional verification: requires A→B AND B→A
- Semantic relatedness filter: antonyms must be conceptually related
- Shows which semantic dimensions differ between words
- Achieves perfect scores (1.000) on known pairs like begin/end, import/export

**2. Vector Difference Method** (`src/improved_analysis.py`)
- For each antonym pair (w+, w-), computes direction d = v(w+) - v(w-)
- To find antonyms for word w, searches for candidates c where (w - c) aligns with known antonym directions
- Successfully finds known antonyms at rank 1 (happy→unhappy, good→bad, love→hate, etc.)

**3. Orthogonal Basis Method** (`src/antonym_space.py`)
- Builds orthonormal basis from antonym directions via Gram-Schmidt or SVD
- Projects words onto this basis
- Less effective for discovering new antonyms (prioritizes orthogonality over semantics)

### Key Algorithm (Vector Difference Method)

```
1. Build matrix D of all normalized antonym difference vectors
2. Use SVD to find principal antonym directions (76 directions explain 90% variance for EN)
3. For antonym discovery: find word c that maximizes max(|D · normalize(w - c)|)
4. For neutral words: find words with minimal ||projection onto principal directions||
```

### Data Flow

```
English:
  WordNet (via NLTK) → antonym_loader.py → 3,556 pairs → 2,352 valid in GloVe
  gensim API → word2vec_loader.py → 400K word vectors (GloVe-100)
  → english_analysis_export.py → results/english_analysis.json

Japanese:
  EPWING辞書 (大辞林+明鏡) → japanese_antonym_loader.py → 5,100 pairs → 1,911 valid in fastText
  fastText cc.ja.300 → word2vec_loader.py → 200K word vectors (300d)
  → japanese_analysis.py → results/japanese_analysis.json

Cross-lingual:
  english_analysis.json + japanese_analysis.json
  → cross_lingual_comparison.py → results/cross_lingual_comparison.json + paper/figures/

Paper:
  results/*.json → generate_paper.py → paper/paper.md
```

### Japanese Antonym Extraction

`src/japanese_antonym_loader.py` extracts antonym pairs from EPWING dictionary JSON files:
- **大辞林** (daijirin.json): ~4,000 pairs from `<begin_reference>⇔WORD<end_reference>` tags
- **明鏡国語辞典** (meikyo.json): ~2,700 pairs from inline `⇔WORD` markers
- Total: ~5,100 unique pairs after deduplication
- EPWING data at `/home/user/dev/shinar-backup-20260119/epwing_parser/data/`

### Key Findings

**English** (GloVe-100, 2,352 pairs):
- Hits@1: 89.0%, Hits@10: 98.8%
- 76 SVD components for 90% variance
- Perfect scores on: begin/end, import/export, male/female, buy/sell

**Japanese** (fastText-300, 1,911 pairs):
- Hits@1: 86.7%, Hits@10: 100%
- 220 SVD components for 90% variance
- Perfect scores on: 上/下, 善/悪, 強い/弱い, 自然/人工, 真実/虚偽

**Cross-lingual**:
- Both languages achieve ~87-89% Hits@1 accuracy
- KS test on SVD variance curves: p=0.996 (no significant structural difference)
- Translation pairs show consistent detection across languages

## Model Options

| Model | Dimensions | Vocab Size | Download Size |
|-------|------------|------------|---------------|
| glove-50 | 50 | 400K | ~70MB |
| glove-100 | 100 | 400K | ~130MB |
| glove-200 | 200 | 400K | ~250MB |
| glove-300 | 300 | 400K | ~380MB |
| google-news | 300 | 3M | ~1.5GB |
| cc.ja.300 | 300 | 2M | ~1.3GB |

GloVe models download automatically via gensim. fastText cc.ja.300 downloads from Facebook AI.

## Dependencies

Managed via `pixi.toml`. Key packages:
- gensim, numpy, scipy, scikit-learn (analysis)
- nltk (WordNet antonyms)
- matplotlib, pandas (visualization)
- vibrato, zstandard (Japanese tokenization for W2V training)
