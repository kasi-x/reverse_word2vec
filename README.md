# reverse_word2vec — Axis-wise Semantic Inversion via ICA-decomposed Embedding Spaces

Retrieve lexical antonyms (king→queen, hot→cold, ...) from **ICA-decomposed word
embedding spaces**. GloVe-100 is decomposed with FastICA into ~100 statistically
independent semantic axes (gender, sentiment, temperature, size, ...), and several
retrieval approaches are studied on top of the decomposition:

- **Axis inversion** — negate a word's score on a semantic axis, reconstruct the
  vector, and take the nearest neighbour. Interpretable, and evaluated both blind
  and with oracle axis selection.
- **MLP + reranker (deployable pipeline)** — an MLP classifier scores antonymy
  from axis-wise pair features `(|s1−s2| ∥ s1⊙s2)`; a logistic reranker re-orders
  the top-10 candidates using ICA cosine, word frequency, and MLP rank signals.
- **Counter-fitting** (Mrkšić et al., 2016) — investigated and **rejected**: a
  global fit leaks antonym labels into the embedding space, and a leak-free
  per-fold refit collapses retrieval to ~0 Hits@1 (see
  `results/leakfree_comparison.json`). Headline results do **not** use it.

The generated paper lives at [`paper/paper.md`](paper/paper.md); every number in
it is read from a `results/*.json` file, and missing files produce visible
`n/a` placeholders rather than stale or invented numbers.

## Results (GloVe-100 + FastICA, raw space, 1,121 WordNet antonym pairs)

All numbers below come from one consistent, leak-free raw space
(`results/ica_space.json` records `counter_fitted: false`):

| Method | Hits@1 | Hits@5 | Hits@10 |
|--------|--------|--------|---------|
| Blind axis inversion (declared label group) | 0.0% | 26.7% | 33.3% |
| Oracle axis inversion (knows the target; not deployable) | 25.2%±0.03 | 39.1%±0.04 | 45.0%±0.04 |
| MLP classifier (5-fold CV) | 16.7%±0.02 | 31.4%±0.03 | 38.3%±0.04 |
| MLP + Reranker (5-fold CV) | 19.0%±0.03 | 33.2%±0.03 | 38.3%±0.04 |
| Traditional analogy b−a+c (Google Analogy set) | — | — | 57.5% acc |

Key findings after correcting the evaluation:

- With oracle axis selection, interpretable ICA axes carry real antonym signal
  (25.2% Hits@1) — but without the target, blind axis selection is much weaker.
- The deployable MLP + reranker improves over the MLP alone (16.7% → 19.0%)
  without any oracle knowledge.
- An earlier draft reported 37.9% Hits@1 for MLP + reranker, but that number was
  computed on a **globally counter-fitted space**, where antonym labels of the
  whole vocabulary had shaped the embedding geometry (label leakage), and it was
  mixed in one table with raw-space baselines. The corrected, fully consistent
  numbers are the ones above. See `results/leakfree_comparison.json` and the
  paper's Discussion for the counter-fitting analysis.

## Setup

Requires [pixi](https://pixi.sh) and (for the task runner) [go-task](https://taskfile.dev).

```bash
pixi install            # create the environment
task download-models:en # GloVe-100 (~130MB, cached in ~/gensim-data)
task download-analogy   # Google Analogy Dataset -> data/questions-words.txt
```

## Usage

```bash
# Full pipeline: ICA fit → qualitative → quantitative → analysis → paper
task run:all

# Or phase by phase
task run:ica           # Phase 1: FastICA decomposition + axis labeling
task run:qualitative   # Phase 2: blind + oracle qualitative eval
task run:quantitative  # Phase 3: oracle-inversion CV + Google Analogy
task run:analysis      # Phase 4: figures + bias/reconstruction analysis
task paper:generate    # Phase 5: regenerate paper/paper.md from results/*.json

# Re-use the cached ICA space (no refit)
pixi run python run_pipeline.py --load-space --phase 2
task run:relabel       # re-run axis labeling only

# Headline evaluation (MLP + reranker CV and 70/15/15 holdout)
task eval:reranker

# Counter-fitting studies (why CF was rejected)
task eval:leakfree              # leak-free per-fold CF comparison (~0 Hits@1)
task eval:counterfit-separation # global CF cosine shift (paper Method numbers)
```

Other useful flags: `--model glove-300`, `--components 100`, `--vocab-limit`,
`--counter-fit` (experimental CF pipeline).

## Quality gates

```bash
task test    # pytest — fast, synthetic-data unit tests (no model downloads)
task lint    # ruff check
task format  # ruff format
```

## Repository structure

```
run_pipeline.py        # CLI orchestrating phases 1-5 (+ --relabel, --load-space)
src/
  ica_transformer.py   # FastICA fit/transform/reconstruct/save (with provenance)
  axis_labeler.py      # WordNet-based automatic axis labeling
  semantic_operations.py  # axis inversion, sliding, NN search, analogies
  antonym_classifier.py   # MLP/logistic antonymy classifier on ICA features
  reranker.py          # logistic reranker + full-pipeline cross-validation
  counter_fitting.py   # Mrkšić et al. counter-fitting (experimental)
  qualitative_eval.py  # canonical cases; blind + oracle protocols
  quantitative_eval.py # oracle-inversion CV + Google Analogy eval
  analysis_report.py   # figures: kurtosis, per-axis success, bias, recon-k curve
  relevance.py         # confidence scoring + bias profiles
  eval_utils.py        # shared eval helpers (folds, unit_rows, compute_hits, ...)
  generate_paper.py    # data-driven paper.md generation
scripts/               # 15 standalone eval/diagnostic scripts (research runs)
paper/                 # generated paper.md + figures
results/               # eval outputs (JSON, tracked); large ICA .npz + ica_space.json ignored
tests/                 # pytest suite (synthetic data, 52 tests)
```

## Reproducibility notes

- All stochastic stages are seeded (`random_state=42`): FastICA, MLP, logistic
  models, fold shuffling. Counter-fitting is deterministic.
- The cached ICA space records its provenance (`model`, `counter_fitted`,
  `vocab_limit`, `random_state`, `created_at`) in `results/ica_space.json`.
- **Consistency rule**: evaluate everything against the *same* space. If
  `results/ica_space` changes (refit, counter-fitting), re-run the evals
  (`task run:qualitative run:quantitative`, `task eval:reranker`) before
  regenerating the paper — numbers from different spaces must never be mixed.
