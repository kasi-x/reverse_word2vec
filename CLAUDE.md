# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project retrieves lexical antonyms from ICA-decomposed word embedding spaces.
GloVe-100 is decomposed with FastICA into ~100 statistically independent semantic
axes (gender, sentiment, temperature, ...). Two retrieval approaches are studied:

1. **Axis inversion** — negate a word's score on a semantic axis, reconstruct, and
   take the nearest neighbour (weak on its own; see paper).
2. **MLP + reranker (headline)** — an MLP classifier scores antonymy from axis-wise
   pair features `(|s1−s2| ∥ s1⊙s2)`; a logistic reranker re-orders the top-10
   using ICA cosine, word frequency, and MLP rank signals.

Counter-fitting (Mrkšić et al., 2016) was investigated and **rejected**: a global
fit leaks antonym labels into the space, and a leak-free per-fold fit collapses
retrieval to ~0 Hits@1. All headline results use raw GloVe + ICA (see
`results/leakfree_comparison.json` and the paper's Method section).

## Commands

```bash
# Setup (pixi)
pixi install

# Task runner (requires go-task)
task --list

# Download models / data
task download-models:en   # GloVe-100 (~130MB, cached in ~/gensim-data)
task download-analogy     # Google Analogy Dataset -> data/questions-words.txt

# Pipeline phases
task run:ica           # Phase 1: ICA decomposition + axis labeling
task run:qualitative   # Phase 2: Qualitative eval (blind + oracle protocols)
task run:quantitative  # Phase 3: Quantitative eval (oracle CV + analogy)
task run:analysis      # Phase 4: Analysis & visualization
task run:relabel       # Re-label axes on the cached space (no refit)
task run:all           # Full pipeline (phases 1-5)
task run:cf            # Experimental: full pipeline with counter-fitting

# Evaluations (scripts/)
task eval:reranker     # MLP + reranker retrieval (CV + 70/15/15 holdout)
task eval:leakfree     # Leak-free counter-fitting comparison
task eval:translation  # Translation-baseline refinements
task eval:discovery    # WordNet -> ConceptNet-only generalization

# Paper
task paper:generate    # Phase 5: regenerate paper/paper.md from results/*.json

# Quality
task test              # pytest (fast, synthetic data, no downloads)
task lint              # ruff check
task format            # ruff format

# CLI options
pixi run python run_pipeline.py --model glove-300 --components 100 --phase 1
pixi run python run_pipeline.py --load-space  # use cached ICA space
pixi run python run_pipeline.py --relabel     # relabel axes on cached space
```

## Architecture

### Modules

| Module | Description |
|--------|-------------|
| `src/ica_transformer.py` | ICA core: fit, transform, reconstruct, save/load with provenance metadata |
| `src/axis_labeler.py` | Automatic axis labeling using WordNet antonym pairs + category seeds |
| `src/semantic_operations.py` | Axis inversion (single/multi/label-group/source-topk), NN search, analogy |
| `src/antonym_classifier.py` | MLP/logistic antonymy classifier on ICA pair features |
| `src/reranker.py` | Logistic reranker over MLP candidates (6 signals) + full-pipeline CV |
| `src/counter_fitting.py` | Mrkšić et al. counter-fitting (experimental; see CF rejection note) |
| `src/qualitative_eval.py` | Canonical cases (king→queen, ...); blind (primary) + oracle protocols |
| `src/quantitative_eval.py` | Oracle axis-inversion 5-fold CV + Google Analogy eval |
| `src/analysis_report.py` | Figures: kurtosis, per-axis success, bias, reconstruction-vs-k curve |
| `src/relevance.py` | Confidence scoring + bias profiles on labeled axes |
| `src/conceptnet_loader.py` | ConceptNet antonym pairs (download/cache/WordNet expansion) |
| `src/contrastive_retrieval.py` | Contrastive retrieval variant (used by `scripts/eval_all_approaches.py`) |
| `src/eval_utils.py` | Shared eval helpers (`unit_rows`, `compute_hits`, `topn_excluding`) |
| `src/generate_paper.py` | Data-driven paper generation: reads results/*.json, warns on missing |
| `run_pipeline.py` | CLI orchestrating phases 1-5 (`--relabel`, `--load-space`, `--counter-fit`) |

### Infrastructure

| Module | Description |
|--------|-------------|
| `src/word2vec_loader.py` | GloVe/fastText/Google News model loading via gensim |
| `src/antonym_loader.py` | WordNet antonym pair extraction (NLTK) |
| `scripts/` | 14 standalone eval/diagnostic scripts (one-off research runs; each is run manually via `pixi run python scripts/<name>.py`) |

### Key Algorithm

```
Phase 1 - ICA Decomposition:
  1. Load GloVe-100 (gensim, 400K words)
  2. Filter to valid English words (top 50K)
  3. Center and apply FastICA (random_state=42) -> 100 components
  4. Auto-label axes using WordNet antonym pair alignment
  5. Save space with provenance (model, counter_fitted, seed, timestamp)

Phase 2/3 - Evaluation:
  - Qualitative: blind protocols (declared-label axes / source top-|z| axis);
    oracle protocol kept as a stated non-deployable reference
  - Quantitative: oracle axis-inversion 5-fold CV; Google analogy dataset

Headline (scripts/eval_reranker.py):
  GloVe-100 -> ICA scores -> MLP on (|s1-s2|, s1*s2) -> top-10 candidates
  -> logistic reranker (mlp_score, ica_cosine, mlp_rank, freq_ratio,
     interaction, inv_rank) -> Hits@{1,5,10}
```

### Data Flow

```
GloVe-100 (gensim) -> ica_transformer.py -> results/ica_space.{npz,json}
WordNet (NLTK)      -> antonym_loader.py -> antonym pairs
                    -> axis_labeler.py   -> results/axis_profiles.json
                    -> qualitative_eval.py  -> results/qualitative_eval.json
                    -> quantitative_eval.py -> results/antonym_retrieval.json
                                            -> results/analogy_eval.json
scripts/eval_reranker.py -> results/reranker_eval.json
scripts/eval_leakfree.py -> results/leakfree_comparison.json
analysis_report.py + relevance.py -> results/analysis_report.json, relevance_analysis.json
                                 -> paper/figures/*.png
generate_paper.py (reads ALL of the above JSONs) -> paper/paper.md
```

**Consistency rule:** every number in `paper/paper.md` must come from a
`results/*.json` file produced on the *same* ICA space. `results/ica_space.json`
records the space's provenance (`counter_fitted`, `model`, `random_state`, ...).
If evals were run against a different space than the cached one, re-run them
(`task run:qualitative`, `run:quantitative`, `eval:reranker`) before regenerating
the paper — mixed-space numbers are invalid.

## Model Options

| Model | Dimensions | Vocab Size | Download Size |
|-------|------------|------------|---------------|
| glove-50 | 50 | 400K | ~70MB |
| glove-100 | 100 | 400K | ~130MB |
| glove-200 | 200 | 400K | ~250MB |
| glove-300 | 300 | 400K | ~380MB |
| google-news | 300 | 3M | ~1.5GB |

GloVe models download automatically via gensim (cached in `~/gensim-data`).

## Dependencies

Managed via `pixi.toml`. Key packages:
- gensim, numpy, scipy, scikit-learn (analysis + ICA)
- nltk (WordNet antonyms)
- matplotlib, pandas (visualization)
- pytest + ruff (dev, run via `task test` / `task lint`)
