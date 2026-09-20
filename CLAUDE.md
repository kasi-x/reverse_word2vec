# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project decomposes word embedding spaces (GloVe) using ICA (Independent Component Analysis) to discover interpretable semantic axes. Each ICA component corresponds to an independent semantic dimension (e.g., gender, sentiment, size). By inverting a word's score on a specific axis, we perform targeted semantic inversion (e.g., king -> queen on the gender axis).

## Commands

```bash
# Setup (pixi)
pixi install

# Task runner (requires go-task)
task --list

# Download models
task download-models:en   # GloVe-100 (~130MB)

# Download Google Analogy Dataset
task download-analogy

# Run phases individually
task run:ica           # Phase 1: ICA decomposition + axis labeling
task run:qualitative   # Phase 2: Qualitative eval (canonical test cases)
task run:quantitative  # Phase 3: Quantitative eval (5-fold CV + analogy)
task run:analysis      # Phase 4: Analysis & visualization
task paper:generate    # Phase 5: Generate paper from results

# Run full pipeline (all phases)
task run:all

# CLI options
pixi run python run_pipeline.py --model glove-300 --components 100 --phase 1
pixi run python run_pipeline.py --load-space  # use cached ICA space
```

## Architecture

### ICA-based Semantic Axis Approach

**Core idea**: ICA decomposes embedding space into statistically independent components.
Unlike PCA (which maximizes variance), ICA maximizes statistical independence, yielding
more interpretable semantic axes.

### Modules

| Module | Description |
|--------|-------------|
| `src/ica_transformer.py` | ICA decomposition core: fit, transform, reconstruct, save/load |
| `src/axis_labeler.py` | Automatic axis labeling using WordNet antonym pairs |
| `src/semantic_operations.py` | Axis inversion, sliding, nearest-neighbor search |
| `src/qualitative_eval.py` | Canonical test cases (king->queen, hot->cold, etc.) |
| `src/quantitative_eval.py` | 5-fold CV antonym retrieval + Google Analogy eval |
| `src/analysis_report.py` | Visualizations, bias analysis, reconstruction quality |
| `src/generate_paper.py` | Generate paper.md from results JSON |
| `run_pipeline.py` | CLI pipeline orchestrating all phases |

### Infrastructure (kept from previous work)

| Module | Description |
|--------|-------------|
| `src/word2vec_loader.py` | GloVe/fastText/Google News model loading via gensim |
| `src/antonym_loader.py` | WordNet antonym pair extraction |
| `src/japanese_antonym_loader.py` | EPWING dictionary antonym extraction (future use) |

### Key Algorithm

```
Phase 1 - ICA Decomposition:
  1. Load GloVe-100 (400K words, 100d)
  2. Filter to valid English words (top 50K)
  3. Center and apply FastICA -> 100 independent components
  4. Auto-label axes using WordNet antonym pair alignment

Phase 2 - Semantic Inversion:
  For word w, axis k:
    1. s = ICA_transform(w)
    2. s[k] = -s[k]
    3. v' = ICA_reconstruct(s)
    4. Find nearest neighbor to v'

Phase 3 - Evaluation:
  - Qualitative: 15 canonical cases (king->queen, hot->cold, etc.)
  - Quantitative: 5-fold CV on ~2000 WordNet antonym pairs
  - Comparison: ICA inversion vs traditional vector analogy (b-a+c)
```

### Data Flow

```
GloVe-100 (gensim) -> ica_transformer.py -> ICA space (results/ica_space.{npz,json})
WordNet (NLTK)      -> antonym_loader.py  -> antonym pairs
                    -> axis_labeler.py    -> axis profiles (results/axis_profiles.json)
                    -> semantic_operations.py -> inversion results
                    -> qualitative_eval.py   -> results/qualitative_eval.json
                    -> quantitative_eval.py  -> results/antonym_retrieval.json
                                             -> results/analogy_eval.json
                    -> analysis_report.py    -> results/analysis_report.json
                                             -> paper/figures/*.png
                    -> generate_paper.py     -> paper/paper.md
```

## Model Options

| Model | Dimensions | Vocab Size | Download Size |
|-------|------------|------------|---------------|
| glove-50 | 50 | 400K | ~70MB |
| glove-100 | 100 | 400K | ~130MB |
| glove-200 | 200 | 400K | ~250MB |
| glove-300 | 300 | 400K | ~380MB |
| google-news | 300 | 3M | ~1.5GB |

GloVe models download automatically via gensim.

## Dependencies

Managed via `pixi.toml`. Key packages:
- gensim, numpy, scipy, scikit-learn (analysis + ICA)
- nltk (WordNet antonyms)
- matplotlib, pandas (visualization)
