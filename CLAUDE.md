# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project explores Word2Vec semantic spaces through antonym relationships. The core idea is to use known antonym pairs to define semantic directions, then analyze:
1. Which words lie near the zero vector (semantically "neutral" words)
2. Whether new antonym relationships can be discovered for words without known antonyms

## Commands

```bash
# Setup (requires Python 3.10+)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run fast analysis (recommended - 4x faster, good accuracy)
python src/fast_antonym_v2.py

# Run advanced analysis (semantic clustering, slightly slower)
python src/advanced_analysis.py

# Run basic experiment (orthogonal approach - for comparison)
python run_experiment.py --model glove-100 --method greedy --max-axes 100
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
- For each antonym pair (w+, w-), computes direction d = v(w+) - v(w-)
- To find antonyms for word w, searches for candidates c where (w - c) aligns with known antonym directions
- Successfully finds known antonyms at rank 1 (happy→unhappy, good→bad, love→hate, etc.)

### Key Algorithm (Vector Difference Method)

```
1. Build matrix D of all normalized antonym difference vectors
2. Use SVD to find principal antonym directions (76 directions explain 90% variance)
3. For antonym discovery: find word c that maximizes max(|D · normalize(w - c)|)
4. For neutral words: find words with minimal ||projection onto principal directions||
```

### Data Flow

```
WordNet (via NLTK) → antonym_loader.py → 3556 antonym pairs → 2352 valid in GloVe
                                              ↓
gensim API → word2vec_loader.py → 400K word vectors (GloVe-100)
                                              ↓
                              improved_analysis.py (recommended)
                                    or
                              antonym_space.py → experiment.py
```

### Key Findings

**Perfectly identified antonym pairs** (bidirectional score = 1.000):
- begin/end, start/finish, enter/exit
- import/export, increase/decrease, expand/contract
- majority/minority, maximum/minimum, positive/negative
- rural/urban, domestic/foreign

**Neutral words** (near zero in antonym space):
- Discourse connectors: "likewise", "ironically", "conversely", "interestingly"
- These lack positive/negative semantic valence

**Discovered antonyms for abstract concepts**:
- peace ↔ war (0.808, bidirectional)
- beauty → ugliness (0.741)
- wisdom → folly (0.670)
- truth → false (0.601)
- nature → supernatural (0.598)
- chaos → security/stability

## Model Options

| Model | Dimensions | Vocab Size | Download Size |
|-------|------------|------------|---------------|
| glove-50 | 50 | 400K | ~70MB |
| glove-100 | 100 | 400K | ~130MB |
| glove-200 | 200 | 400K | ~250MB |
| glove-300 | 300 | 400K | ~380MB |
| google-news | 300 | 3M | ~1.5GB |

GloVe models download automatically via gensim. Google News requires more RAM (~4GB+).
