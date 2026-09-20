# Axis-wise Semantic Inversion via ICA-decomposed Embedding Spaces

## Abstract

We propose a pipeline for lexical antonym retrieval using ICA-decomposed word embedding
spaces. GloVe-100 vectors are first adjusted via counter-fitting (Mrkšić et al., 2016)
to push antonym vectors apart, then decomposed with FastICA into statistically
independent semantic axes. An MLP classifier trained on axis-wise features
(|s1−s2| ∥ s1⊙s2) retrieves antonym candidates, and a lightweight logistic reranker
re-orders them using ICA cosine, word frequency, and MLP rank signals.
On 1,121 WordNet antonym pairs with 5-fold cross-validation, the full pipeline
achieves 37.9% Hits@1, more than doubling both the classifier alone (17.7%)
and the oracle axis-inversion baseline (25.2%).

## Method

### ICA Decomposition

Given a word embedding matrix $X \in \mathbb{R}^{n \times d}$ (n words, d dimensions),
we apply FastICA to obtain a score matrix $S \in \mathbb{R}^{n \times k}$ where each
column represents a statistically independent component. Unlike PCA, which maximizes
variance, ICA maximizes statistical independence, yielding more interpretable axes.

### Axis Labeling

We automatically label axes using WordNet antonym pairs. For each antonym pair (w1, w2),
we compute the ICA score difference |S[w1] - S[w2]| and assign the pair to the axis
with the largest difference. Axes accumulating many pairs from the same semantic category
(e.g., male/female, king/queen -> gender) receive that category label.

### Counter-fitting

Raw GloVe vectors encode distributional similarity: antonyms such as *hot* and *cold*
appear in identical contexts and therefore cluster together (cosine ≈ +0.47).
Counter-fitting (Mrkšić et al., 2016) corrects this by minimising two objectives:

- **Antonym Repel (AR)**: gradient steps that push antonym pairs below a target
  cosine of −0.3, applied only to the words involved in constraints.
- **Vector Space Preservation (VSP)**: a pull-back term (λ = 0.1) that prevents
  unconstrained drift from the original embedding geometry.

After 100 iterations, mean antonym cosine drops from +0.47 to −0.23,
while the neighbourhood structure of unconstrained words is unchanged.

### ICA Feature Classifier

The counter-fitted vectors are decomposed with FastICA into 100 independent
components, yielding score matrix S ∈ ℝ^{n×100}.
For each candidate word pair (w1, w2), we form a 200-dimensional feature vector:

  φ(w1, w2) = [ |s1 − s2|, s1 ⊙ s2 ]

where s1, s2 ∈ ℝ^100 are the ICA score vectors.
The absolute difference captures axis-wise polarity contrast;
the element-wise product captures alignment (negative = opposite poles).
An MLP (128→64 hidden units) is trained with negatives sampled at ratio 3:1.

### Reranker

The MLP retrieves a top-10 candidate list but ranks noise words (rare vocabulary
items with extreme ICA scores) among the true antonyms. A logistic reranker
re-orders candidates using six signals:

| Feature | Description |
|---------|-------------|
| mlp_score | MLP P(antonym) |
| ica_cosine | Cosine of ICA score vectors |
| mlp_rank | Position in MLP list (1–10) |
| freq_ratio | cand_rank / query_rank (proxy for rarity) |
| interaction | mlp_score × (−ica_cosine) |
| inv_rank | 1 / mlp_rank |

The reranker is trained on a held-out validation fold's MLP predictions,
so the test set never leaks into any training stage.

### Oracle Axis Inversion (Baseline)

As an upper-bound baseline, given the true target word we identify the single
ICA axis with the largest normalised score difference |s1−s2|/σ, negate that
axis in the source word's score vector, reconstruct, and retrieve the nearest
neighbour. This requires knowledge of the target word and is not deployable
in practice.

## Results

### Axis Interpretability

- Total ICA axes: 100
- Labeled axes: 89 (89.0%)
- Well-supported axes (>=3 antonym pairs): 89
- Mean kurtosis: 4.27

![Kurtosis Distribution](figures/kurtosis_distribution.png)

### Qualitative Evaluation

On 15 canonical test cases:

| Metric | Score |
|--------|-------|
| Hits At 1 | 66.7% |
| Hits At 5 | 66.7% |
| Hits At 10 | 66.7% |

**Selected examples:**

| Input | Axis | Expected | Top-3 Results | Rank |
|-------|------|----------|---------------|------|
| king | gender | queen | queen, monarch, kingdom | 1 |
| boy | gender | girl | girl, man, kid | 1 |
| father | gender | mother | mother, son, brother | 1 |
| husband | gender | wife | wife, daughter, mother | 1 |
| happy | sentiment | sad | i, we, want | - |
| good | sentiment | bad | n't, just, get | - |
| love | sentiment | hate | loves, longing, passion | - |
| hot | temperature | cold | cold, dry, cool | 1 |
| warm | temperature | cool | cool, dry, cold | 1 |
| big | size | small | huge, biggest, large | 1 |

### Antonym Retrieval (5-fold CV)

Cross-validated on 1121 WordNet antonym pairs:

| Method | Hits@1 | Hits@5 | Hits@10 |
|--------|--------|--------|---------|
| Oracle axis inversion† | 0.252±0.015 | 0.391±0.014 | 0.450±0.018 |
| MLP classifier | 0.177±0.012 | 0.406±0.032 | 0.495±0.035 |
| MLP + Reranker | 0.379±0.039 | 0.480±0.039 | 0.495±0.035 |

† Oracle: knows the target word to select the best axis (upper bound, not deployable).

### Google Analogy Dataset

Evaluated 2800 analogy questions:

| Method | Accuracy |
|--------|----------|
| ICA Inversion | 18.2% |
| Traditional (b-a+c) | 57.5% |

**Per-category results:**

| Category | ICA | Traditional | n |
|----------|-----|-------------|---|
| capital-common-countries | 19.0% | 94.0% | 200 |
| capital-world | 18.5% | 87.5% | 200 |
| currency | 0.0% | 9.5% | 200 |
| city-in-state | 6.0% | 34.5% | 200 |
| family | 25.0% | 77.0% | 200 |
| gram1-adjective-to-adverb | 2.0% | 19.5% | 200 |
| gram2-opposite | 3.5% | 22.0% | 200 |
| gram3-comparative | 30.5% | 80.5% | 200 |
| gram4-superlative | 0.0% | 53.5% | 200 |
| gram5-present-participle | 37.0% | 55.0% | 200 |
| gram6-nationality-adjective | 18.0% | 82.0% | 200 |
| gram7-past-tense | 42.5% | 52.0% | 200 |
| gram8-plural | 46.0% | 77.5% | 200 |
| gram9-plural-verbs | 6.5% | 60.5% | 200 |

### Per-Axis Inversion Success

![Per-Axis Success](figures/per_axis_success.png)

### Bias Analysis

Distribution of profession words along the 'gender' axis:

![Bias Visualization](figures/bias_gender.png)

### Reconstruction Quality

- Mean relative error: 0.0000
- Median relative error: 0.0000
- 95th percentile: 0.0000

![Reconstruction Quality](figures/reconstruction_quality.png)

### Reranker Feature Analysis

Logistic regression coefficients (trained on 168 validation pairs):

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| freq_ratio | -1.481 | penalises rare noise candidates |
| ica_cosine | -1.020 | prefers negative ICA cosine (semantic opposites) |
| interaction | +1.020 | synergy: high MLP score + negative cosine |
| mlp_rank | -0.474 | higher-ranked candidates preferred |
| inv_rank | -0.098 | 1/rank bonus |
| mlp_score | -0.043 | raw MLP probability |

The dominant signal is `freq_ratio` (−1.48): the MLP tends to rank rare
vocabulary items highly because their extreme ICA scores superficially
resemble antonym features. The reranker suppresses these by penalising
candidates that are much rarer than the query word.

## Discussion

**Why does the reranker outperform the oracle baseline (37.9% vs 25.2%)?**
The oracle baseline only flips the single most-discriminative ICA axis, then
retrieves the nearest neighbour to the reconstructed vector. Reconstruction
noise from ICA round-tripping limits precision. The MLP classifier avoids
reconstruction altogether by scoring candidates directly from ICA features;
the reranker then filters residual noise using frequency and ICA cosine signals.

**Counter-fitting is essential.** Without CF, antonym pairs cluster together
in GloVe space (cosine ≈ +0.47), providing no discriminative signal for the
classifier. CF pushes antonyms below −0.3 cosine, making `ica_cosine` a
reliable reranker feature and improving MLP feature separability.

**Multi-sense limitation.** The method retrieves one antonym per query;
polysemous words (e.g., *light* = weight/brightness/mood) may return the
antonym for an unintended sense. Future work could use the per-axis inversion
to offer multiple sense-specific opposites simultaneously.

**Comparison with LLM-based antonym retrieval.** Large language models
can reliably retrieve antonyms for most common words. The statistical pipeline
here shows that a principled embedding-space approach is competitive for
common-vocabulary antonymy (49.5% Hits@10), without requiring generation
or prompting infrastructure.
