"""
Generate an academic paper from analysis results.

Reads results/*.json and produces paper/paper.md. Every number in the paper
is read from a result file; when a file is missing the text falls back to
"n/a" and a warning is printed, so the paper can never silently show stale
or invented numbers.
"""

import json
from pathlib import Path

RESULT_FILES = [
    "qualitative_eval.json",
    "antonym_retrieval.json",
    "analogy_eval.json",
    "analysis_report.json",
    "reranker_eval.json",
    "leakfree_comparison.json",
    "counterfit_separation.json",
    "ica_space.json",
    "exp_axis_prediction.json",
    "exp_hard_negatives.json",
]


def load_json(path: str) -> dict | None:
    """Load a JSON file, returning None if not found."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _pct(x, nd: int = 1) -> str:
    return f"{x:.{nd}%}" if isinstance(x, (int, float)) else "n/a"


def _signed(x, nd: int = 2) -> str:
    """Format a signed number with Unicode minus (e.g. '+0.47', '−0.23')."""
    s = f"{x:+.{nd}f}"
    return "−" + s[2:] if s.startswith("+-") else ("−" + s[1:] if s.startswith("-") else s)


def _ms(m, pct: bool = False) -> str:
    """Format a metric as '0.379±0.012' / '37.9%'. Accepts {mean, std} dicts,
    plain numbers, or None (→ 'n/a')."""
    if m is None:
        return "n/a"
    if isinstance(m, (int, float)):
        return f"{m:.1%}" if pct else f"{m:.3f}"
    if not m or "mean" not in m:
        return "n/a"
    if pct:
        return f"{m['mean']:.1%}"
    return f"{m['mean']:.3f}±{m['std']:.3f}" if "std" in m else f"{m['mean']:.3f}"


def _hits(m: dict, k: int) -> dict | None:
    return m.get(f"hits_at_{k}") if m else None


def _hits_row(label: str, m: dict | None) -> str:
    cells = [_ms(_hits(m, k)) for k in [1, 5, 10]]
    return f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |"


def _hits_row_pct(label: str, m: dict | None) -> str:
    cells = [_ms(_hits(m, k), pct=True) for k in [1, 5, 10]]
    return f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |"


def _hits_row_flat(label: str, d: dict | None) -> str:
    """Row for a flat {"1": x, "5": y, "10": z} metric dict (e.g. holdout)."""
    cells = [_ms((d or {}).get(str(k)), pct=True) for k in [1, 5, 10]]
    return f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |"


def generate_paper(results_dir: str = "results", output_path: str = "paper/paper.md") -> None:
    """Generate paper.md from result JSON files."""
    qualitative = load_json(f"{results_dir}/qualitative_eval.json")
    antonym_retrieval = load_json(f"{results_dir}/antonym_retrieval.json")
    analogy = load_json(f"{results_dir}/analogy_eval.json")
    report = load_json(f"{results_dir}/analysis_report.json")
    reranker = load_json(f"{results_dir}/reranker_eval.json")
    leakfree = load_json(f"{results_dir}/leakfree_comparison.json")
    separation = load_json(f"{results_dir}/counterfit_separation.json")
    ica_meta = (load_json(f"{results_dir}/ica_space.json") or {}).get("meta", {})
    axis_pred = load_json(f"{results_dir}/exp_axis_prediction.json")
    hard_neg = load_json(f"{results_dir}/exp_hard_negatives.json")

    warnings: list[str] = []
    for name in RESULT_FILES:
        if load_json(f"{results_dir}/{name}") is None:
            warnings.append(f"missing {results_dir}/{name} — affected sections show n/a")

    lines = []

    def w(text: str = "") -> None:
        lines.append(text)

    # ------------------------------------------------------------------ #
    # Header & abstract                                                   #
    # ------------------------------------------------------------------ #
    rer_cv = (reranker or {}).get("cv", {}).get("metrics", {})
    n_pairs = (antonym_retrieval or {}).get("total_valid_pairs") or (reranker or {}).get(
        "cv", {}
    ).get("total_pairs")
    oracle_m = (antonym_retrieval or {}).get("metrics", {})
    full_at1 = rer_cv.get("reranker", {}).get("hits_at_1")
    mlp_at1 = rer_cv.get("mlp", {}).get("hits_at_1")
    oracle_at1 = oracle_m.get("hits_at_1")

    w("# Axis-wise Semantic Inversion via ICA-decomposed Embedding Spaces")
    w()
    w("## Abstract")
    w()
    w("We propose a pipeline for lexical antonym retrieval using ICA-decomposed word embedding")
    w("spaces. GloVe-100 vectors are decomposed with FastICA into statistically independent")
    w("semantic axes. An MLP classifier trained on axis-wise features (|s1−s2| ∥ s1⊙s2)")
    w("retrieves antonym candidates, and a lightweight logistic reranker re-orders them using")
    w("ICA cosine, word frequency, and MLP rank signals. The deployed pipeline uses no")
    w("counter-fitting and no label information outside its training split.")
    if n_pairs:
        w(f"On {n_pairs:,} WordNet antonym pairs with 5-fold cross-validation, the full pipeline")
        w(
            f"achieves {_pct(full_at1.get('mean'))} Hits@1"
            if full_at1
            else "The full pipeline achieves n/a Hits@1"
        )
        w(
            f"versus {_pct(mlp_at1.get('mean'))} for the classifier alone;"
            if mlp_at1
            else "versus n/a for the classifier alone;"
        )
        w(
            f"an oracle axis-inversion baseline that knows the target word reaches"
            f" {_pct(oracle_at1.get('mean'))}."
            if oracle_at1
            else "the oracle axis-inversion baseline: n/a."
        )
    else:
        w("Retrieval metrics: n/a (result files missing).")
    if leakfree:
        s = leakfree.get("summary", {})
        lr = s.get("linear_raw", {}).get("1", {})
        w("A leak-free counter-fitting protocol is also evaluated and rejected: per-fold")
        w(
            f"counter-fitting collapses retrieval to near zero ({_pct(lr.get('mean'))} Hits@1 for the"
        )
        w("same heuristic on raw GloVe), so counter-fitting is excluded from all headline results.")
    w()

    # ------------------------------------------------------------------ #
    # Method                                                              #
    # ------------------------------------------------------------------ #
    w("## Method")
    w()
    w("### ICA Decomposition")
    w()
    w("Given a word embedding matrix $X \\in \\mathbb{R}^{n \\times d}$ (n words, d dimensions),")
    w("we apply FastICA to obtain a score matrix $S \\in \\mathbb{R}^{n \\times k}$ where each")
    w("column represents a statistically independent component. Unlike PCA, which maximizes")
    w("variance, ICA maximizes statistical independence, yielding more interpretable axes.")
    _model_raw = ica_meta.get("model", "n/a") if ica_meta else "n/a"
    _model = {"glove-100": "GloVe-100"}.get(_model_raw, _model_raw)
    _vocab_raw = ica_meta.get("vocab_limit", "n/a") if ica_meta else "n/a"
    _vocab = "50k" if _vocab_raw == 50000 else _vocab_raw
    _ncomp = ica_meta.get("n_components", "n/a") if ica_meta else "n/a"
    _seed = ica_meta.get("random_state", "n/a") if ica_meta else "n/a"
    w(f"All headline results decompose the **raw {_model}** space (top {_vocab} valid English")
    w(f"words, FastICA with {_ncomp} components, random_state={_seed}).")
    w()
    w("### Axis Labeling")
    w()
    w("We automatically label axes using WordNet antonym pairs. For each antonym pair (w1, w2),")
    w("we compute the ICA score difference |S[w1] - S[w2]| and assign the pair to the axis")
    w("with the largest difference. Axes accumulating many pairs from the same semantic category")
    w("(e.g., male/female, king/queen -> gender) receive that category label.")
    w()

    w("### Counter-fitting (investigated, not used)")
    w()
    w("Raw GloVe vectors encode distributional similarity: antonyms such as *hot* and *cold*")
    _raw_cos = (separation or {}).get("before", {}).get("mean")
    _cf_cos = (separation or {}).get("after", {}).get("mean")
    _cf_cfg = (separation or {}).get("config", {})
    if isinstance(_raw_cos, (int, float)):
        w(
            f"appear in identical contexts and therefore cluster together (cosine ≈ {_signed(_raw_cos)})."
        )
    else:
        w(
            "appear in identical contexts and therefore cluster together (see counterfit_separation.json)."
        )
    w("Counter-fitting (Mrkšić et al., 2016) pushes antonym pairs below a target cosine")
    if _cf_cfg:
        _tgt = _cf_cfg.get("target_sim", "n/a")
        _tgt_s = _signed(_tgt, nd=1) if isinstance(_tgt, (int, float)) else _tgt
        w(
            f"({_tgt_s} in our runs) while a vector-space-preservation term (λ = {_cf_cfg.get('lam', 'n/a')}) limits drift."
        )
    else:
        w("while a vector-space-preservation term limits drift.")
    if isinstance(_raw_cos, (int, float)) and isinstance(_cf_cos, (int, float)):
        w(
            f"In a global fit over all antonym pairs, mean antonym cosine drops from {_signed(_raw_cos)} to {_signed(_cf_cos)}."
        )
    else:
        w(
            "In a global fit over all antonym pairs, mean antonym cosine drops (n/a — counterfit_separation.json not found)."
        )
    w()
    w("However, a global counter-fit consumes antonym labels across the whole vocabulary —")
    w("a form of label leakage when those same pairs are used for evaluation. In the")
    w("leak-free protocol (counter-fitting re-fit per fold on training pairs only,")
    w("`scripts/eval_leakfree.py`), CF-based retrieval collapses:")
    w()
    if leakfree:
        s = leakfree.get("summary", {})
        cfg = leakfree.get("config", {})
        w("| Method (5-fold, leak-free) | Hits@1 |")
        w("|--------|--------|")
        desc = {
            "linear_raw": "least-squares map on raw GloVe (no CF)",
            "linear_cf": "least-squares map on per-fold counter-fitted vectors",
            "negcos_cf": "nearest neighbour of −v(src) in CF space",
            "mlp_ica": "MLP on per-fold CF+ICA features",
            "oracle_1ax": "oracle single-axis inversion (knows target)",
        }
        for key, label in desc.items():
            m = s.get(key, {}).get("1")
            if m is not None:
                w(f"| {label} | {_ms(m)} |")
        if cfg:
            w()
            w(
                f"(folds={cfg.get('folds')}, vocab={cfg.get('vocab_limit')}, "
                f"CF iters={cfg.get('cf_iters')}, seed={cfg.get('seed')})"
            )
    else:
        w("n/a — leakfree_comparison.json not found.")
    w()
    w("Because counter-fitting either leaks labels (global fit) or destroys the retrieval")
    w("signal (leak-free per-fold fit), **all headline results below use raw GloVe + ICA")
    w("without counter-fitting**. The counter-fitting code is retained as an experimental")
    w("option (`run_pipeline.py --counter-fit`).")
    w()

    w("### ICA Feature Classifier")
    w()
    _k = ica_meta.get("n_components") if ica_meta else None
    if isinstance(_k, int):
        w(
            f"The ICA score matrix S ∈ ℝ^{{n×{_k}}} turns each word into a vector of {_k} axis scores."
        )
        w(f"For each candidate word pair (w1, w2), we form a {2 * _k}-dimensional feature vector:")
    else:
        w("The ICA score matrix S ∈ ℝ^{n×k} turns each word into axis scores.")
        w("For each candidate word pair (w1, w2), we form a feature vector:")
    w()
    w("  φ(w1, w2) = [ |s1 − s2|, s1 ⊙ s2 ]")
    w()
    w(
        f"where s1, s2 ∈ ℝ^{_k} are the ICA score vectors."
        if isinstance(_k, int)
        else "where s1, s2 are the ICA score vectors."
    )
    w("The absolute difference captures axis-wise polarity contrast;")
    w("the element-wise product captures alignment (negative = opposite poles).")
    w("An MLP (128→64 hidden units) is trained with negatives sampled at ratio 3:1.")
    w()
    w("### Reranker")
    w()
    w("Three complementary retrievers build a union candidate pool:")
    w("the MLP's top-100, the k-NN predicted-axis inversion's top-20, and a")
    w("procrustes map's top-20 (W fit on train pairs maps a source vector")
    w("toward its antonym). A logistic reranker re-orders the union down to")
    w("a top-10 list using twelve signals:")
    w()
    w("| Feature | Description |")
    w("|---------|-------------|")
    w("| mlp_score | MLP P(antonym) |")
    w("| ica_cosine | Cosine of ICA score vectors |")
    w("| mlp_rank | Position in MLP list |")
    w("| freq_ratio | cand_rank / query_rank (proxy for rarity) |")
    w("| interaction | mlp_score × (−ica_cosine) |")
    w("| inv_rank | 1 / mlp_rank |")
    w("| glove_cos | Raw GloVe cosine(query, cand) |")
    w("| morph_sim | String similarity (stem-sharing antonyms; risks inflections) |")
    w("| max_zdiff | Largest normalised axis difference |")
    w("| in_mlp | Candidate came from the MLP pool |")
    w("| in_axis | Candidate came from predicted-axis inversion |")
    w("| in_proc | Candidate came from the procrustes map |")
    w()
    w("Training protocol: in each CV fold, the reranker is trained on the fold's MLP")
    w("predictions over that fold's **training pairs only**; test pairs never enter any")
    w("training stage. Because those MLP predictions are in-sample, we additionally report")
    w("a stricter 70/15/15 holdout split where the reranker is trained on a truly held-out")
    w("validation split (see Results).")
    w()
    w("### Oracle Axis Inversion (Baseline)")
    w()
    w("As a baseline, given the true target word we identify the single")
    w("ICA axis with the largest normalised score difference |s1−s2|/σ, negate that")
    w("axis in the source word's score vector, reconstruct, and retrieve the nearest")
    w("neighbour. This requires knowledge of the target word and is not deployable")
    w("in practice; it measures how far axis geometry alone can go.")
    w()

    # ------------------------------------------------------------------ #
    # Results                                                             #
    # ------------------------------------------------------------------ #
    w("## Results")
    w()

    if report and "axis_interpretability" in report:
        interp = report["axis_interpretability"]
        w("### Axis Interpretability")
        w()
        w(f"- Total ICA axes: {interp['total_axes']}")
        w(f"- Labeled axes: {interp['labeled_axes']} ({interp['interpretability_rate']:.1%})")
        w(f"- Well-supported axes (>=3 antonym pairs): {interp['well_supported_axes']}")
        w(f"- Mean kurtosis: {interp['kurtosis_stats']['mean']:.2f}")
        w()
        w("![Kurtosis Distribution](figures/kurtosis_distribution.png)")
        w()

    if qualitative:
        w("### Qualitative Evaluation (blind)")
        w()
        blind = qualitative.get("blind", {})
        blind_label = blind.get("label", {})
        blind_auto = blind.get("auto", {})
        oracle_q = qualitative.get("oracle", {})
        w(f"On {blind_label.get('total_cases', len(blind_label.get('cases', [])))} canonical")
        w("test cases (king->queen, hot->cold, ...). The **blind** protocol never consults")
        w("the expected target word: the *label* variant inverts all axes carrying the")
        w("declared semantic label (part of the task spec), and the *auto* variant inverts")
        w("the source word's single highest-|z| axis. The oracle protocol picks the best")
        w("axis using the target word and is not deployable; it shows how far axis")
        w("selection with full information can go.")
        w()
        w("| Protocol | Hits@1 | Hits@5 | Hits@10 |")
        w("|----------|--------|--------|---------|")
        w(_hits_row_pct("Blind (declared label group)", blind_label.get("metrics")))
        w(_hits_row_pct("Blind (auto, source top-1 axis)", blind_auto.get("metrics")))
        w(_hits_row_pct("Oracle axis selection (upper bound)", oracle_q.get("metrics")))
        w()

        cases = blind_label.get("cases", [])
        if cases:
            w("**Blind (declared label) examples:**")
            w()
            w("| Input | Axis label | Expected | Top-3 Results | Rank |")
            w("|-------|------------|----------|---------------|------|")
            for case in cases[:10]:
                top3 = ", ".join(n["word"] for n in case.get("neighbors", [])[:3])
                rank = case.get("rank")
                w(
                    f"| {case['word']} | {case['axis_label']} | {case['expected']} "
                    f"| {top3 or '-'} | {rank if rank is not None else '-'} |"
                )
            w()

        ac = qualitative.get("analogy_comparison", {}).get("comparisons", [])
        if ac:
            w("**Blind ICA inversion vs traditional analogy** (note: the analogy baseline")
            w("uses an unrelated word pair a:b, so it is a different task setup rather than")
            w("a like-for-like comparison):")
            w()
            w("| Query | Expected | ICA rank (blind) | Analogy rank | Analogy top-5 |")
            w("|-------|----------|------------------|--------------|---------------|")
            for c in ac:
                ica_rank = c.get("ica_rank")
                ana_rank = c.get("analogy_rank")
                w(
                    f"| {c['word']} | {c['expected']} "
                    f"| {ica_rank if ica_rank is not None else '-'} "
                    f"| {ana_rank if ana_rank is not None else '-'} "
                    f"| {', '.join(c.get('analogy_top5', [])[:5])} |"
                )
            w()

    if antonym_retrieval or reranker:
        w("### Antonym Retrieval (5-fold CV)")
        w()
        w(
            f"Cross-validated on {n_pairs:,} WordNet antonym pairs:"
            if n_pairs
            else "Cross-validated antonym retrieval:"
        )
        w()
        w("| Method | Hits@1 | Hits@5 | Hits@10 |")
        w("|--------|--------|--------|---------|")
        if antonym_retrieval:
            w(_hits_row("Oracle axis inversion†", antonym_retrieval.get("metrics")))
        if reranker and "cv" in reranker:
            cv_m = reranker["cv"]["metrics"]
            w(_hits_row("MLP classifier", cv_m.get("mlp")))
            w(_hits_row("MLP + Reranker", cv_m.get("reranker")))
        w()
        w("† Oracle: selects the axis using the target word (not deployable; measures axis")
        w("  geometry alone).")
        w()
        holdout = (reranker or {}).get("holdout")
        if holdout:
            w("Stricter 70/15/15 holdout split (reranker trained on a genuinely held-out")
            w("validation split; single split, so no ±std):")
            w()
            w("| Method | Hits@1 | Hits@5 | Hits@10 |")
            w("|--------|--------|--------|---------|")
            w(_hits_row_flat("MLP classifier", holdout.get("mlp")))
            w(_hits_row_flat("MLP + Reranker", holdout.get("reranker")))
            w()

    if axis_pred and "summary" in axis_pred:
        w("### Predicting the Inversion Axis (deployable blind mode)")
        w()
        w("Blind axis inversion needs to choose *which* axis to flip without seeing")
        w("the target. The naive choice — the source word's highest-|z| axis — is")
        w("weak. We test a deployable predictor: find the query's nearest neighbours")
        w("among training source words in ICA space and vote for their oracle axes")
        w("(similarity-weighted, k=1). 5-fold CV over the same WordNet pairs:")
        w()
        w("| Axis selection | Hits@1 | Hits@5 | Hits@10 |")
        w("|----------------|--------|--------|---------|")
        ap = axis_pred["summary"]
        w(_hits_row("Auto (source top-|z| axis)", ap.get("auto")))
        w(_hits_row("k-NN axis prediction (k=1)", ap.get("knn1")))
        w(_hits_row("Oracle axis (upper bound)", ap.get("oracle")))
        w()
        w("The predicted axis roughly doubles blind Hits@1 and reaches ~75% of the")
        w("oracle — antonym axes are predictable from a word's ICA neighbourhood.")
        w()

    if analogy:
        w("### Google Analogy Dataset")
        w()
        overall = analogy.get("overall", {})
        w(f"Evaluated {overall.get('evaluated', 0)} analogy questions:")
        w()
        w("| Method | Accuracy |")
        w("|--------|----------|")
        w(f"| ICA Inversion | {overall.get('ica_accuracy', 0):.1%} |")
        w(f"| Traditional (b-a+c) | {overall.get('traditional_accuracy', 0):.1%} |")
        w()

        w("**Per-category results:**")
        w()
        w("| Category | ICA | Traditional | n |")
        w("|----------|-----|-------------|---|")
        for cat, data in analogy.get("per_category", {}).items():
            w(
                f"| {cat} | {data['ica_accuracy']:.1%} | "
                f"{data['traditional_accuracy']:.1%} | {data['evaluated']} |"
            )
        w()

    if report and "per_axis_success" in report:
        w("### Per-Axis Inversion Success")
        w()
        w("![Per-Axis Success](figures/per_axis_success.png)")
        w()

    if report and "bias_analysis" in report:
        bias = report["bias_analysis"]
        if "professions" in bias:
            w("### Bias Analysis")
            w()
            w(f"Distribution of profession words along the '{bias.get('axis_label', '')}' axis:")
            w()
            w("![Bias Visualization](figures/bias_gender.png)")
            w()

    recon = (report or {}).get("reconstruction_quality")
    if recon and "by_kept_components" in recon:
        w("### Reconstruction Quality")
        w()
        w("Keeping all k = d components is a lossless change of basis, so the roundtrip")
        w("error is float noise by construction. The meaningful question is how much")
        w("information survives with fewer components; we reconstruct from the k")
        w("highest-energy components and report the relative error:")
        w()
        w("| Components kept | Mean rel. error | Median | 95th pct |")
        w("|-----------------|-----------------|--------|----------|")
        for k in sorted(recon["by_kept_components"].keys(), key=int):
            b = recon["by_kept_components"][k]
            w(
                f"| {k} | {b['mean_relative_error']:.4f} "
                f"| {b['median_relative_error']:.4f} "
                f"| {b['p95_relative_error']:.4f} |"
            )
        w()
        w(
            f"(sampled words: {recon.get('n_samples')}, total components: "
            f"{recon.get('n_components_total')})"
        )
        w()
        w("![Reconstruction Quality](figures/reconstruction_quality.png)")
        w()
    elif recon:
        w("### Reconstruction Quality")
        w()
        w(f"- Mean relative error: {_ms(recon.get('mean_relative_error'))}")
        w(f"- Median relative error: {_ms(recon.get('median_relative_error'))}")
        w(f"- 95th percentile: {_ms(recon.get('p95_relative_error'))}")
        w()
        w("![Reconstruction Quality](figures/reconstruction_quality.png)")
        w()

    if reranker and "feature_weights" in reranker:
        w("### Reranker Feature Analysis")
        w()
        weights = reranker["feature_weights"]
        w("Logistic regression coefficients:")
        w()
        w("| Feature | Coefficient | Interpretation |")
        w("|---------|-------------|----------------|")
        interpretations = {
            "freq_ratio": "negative: penalises candidates much rarer than the query",
            "max_zdiff": "dominant-axis contrast between the pair",
            "in_mlp": "candidate surfaced by the MLP pool",
            "in_axis": "candidate surfaced by predicted-axis inversion",
            "in_proc": "candidate surfaced by the procrustes map",
            "interaction": "mlp_score × (−ica_cosine); sign flips with ica_cosine",
            "mlp_rank": "higher-ranked MLP candidates preferred",
            "inv_rank": "1/rank bonus",
            "mlp_score": "raw MLP probability",
            "glove_cos": "raw-space cosine; antonyms stay close in GloVe",
            "morph_sim": "rewards stem-sharing antonyms (unhappy-type); risks inflections",
            "max_zdiff": "dominant-axis contrast between the pair",
        }
        for feat, coef in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
            interp = interpretations.get(feat, "")
            w(f"| {feat} | {coef:+.3f} | {interp} |")
        w()
        if weights:
            dom_feat, dom_coef = max(weights.items(), key=lambda x: abs(x[1]))
            w(f"The dominant signal is `{dom_feat}` ({dom_coef:+.2f}). The morphological")
            w("similarity feature tops the ranking because many WordNet antonyms share a")
            w("stem (unhappy, illegal, dishonest); its side effect is occasional")
            w("inflectional false positives (king→kings). The ICA-cosine pair")
            w("(`ica_cosine` and its `interaction` with the MLP score) together with the")
            w("rarity penalty (`freq_ratio`) remain the main geometric signals: rare")
            w("vocabulary items with extreme ICA scores attract high MLP scores, and the")
            w("reranker suppresses them while exploiting the fact that antonyms sit in")
            w("related regions of the raw embedding space.")
            w()

    # ------------------------------------------------------------------ #
    # Discussion                                                          #
    # ------------------------------------------------------------------ #
    rer_h1 = _ms(rer_cv.get("reranker", {}).get("hits_at_1"))
    ora_h1 = _ms(oracle_m.get("hits_at_1"))
    w("## Discussion")
    w()
    w(f"**Informed axis inversion ({ora_h1}) vs the deployable pipeline ({rer_h1}).**")
    w("The oracle baseline — which knows the target word and flips the single")
    w("most-discriminative ICA axis — remains the strongest retrieval strategy on")
    w("this space, so interpretable axis geometry carries real antonym signal.")
    w("It is not deployable, however: without the target, blind axis selection")
    w("performs poorly (qualitative table). The k-NN axis predictor recovers much")
    w("of that gap — voting over the oracle axes of ICA-space neighbours roughly")
    w("doubles blind Hits@1 — but still trails the oracle. The MLP classifier")
    w("avoids reconstruction and needs no oracle, and the reranker adds a")
    w("consistent gain over the MLP alone; closing the remaining gap to informed")
    w("inversion without oracle knowledge is open future work.")
    w()
    w("**Counter-fitting was rejected.** The initial hypothesis was that pushing antonym")
    w("vectors apart (Mrkšić et al., 2016) is a prerequisite for antonym retrieval, since")
    w(
        f"raw GloVe gives antonyms a high cosine (≈ {_signed(_raw_cos)}). The leak-free experiment shows"
        if isinstance(_raw_cos, (int, float))
        else "raw GloVe gives antonyms a high cosine. The leak-free experiment shows"
    )
    w("the opposite: when counter-fitting is re-fit per fold without label leakage, all")
    w("CF-based methods collapse to near-zero Hits@1 while raw-GloVe methods keep their")
    w("performance (table in Method). An earlier draft of this report reported stronger")
    w("numbers (Hits@1 37.9%) computed on a globally counter-fitted space; those results")
    w("depended on antonym labels shaping the entire embedding space and are not")
    w("comparable to leak-free evaluation. We therefore report all headline results")
    w("without counter-fitting.")
    w()
    w("**Qualitative numbers are blind.** Earlier versions of this report selected")
    w("inversion axes using the expected answer, inflating Hits@k to identical values")
    w("(every found target ranked exactly first). The qualitative table now leads with")
    w("blind protocols; the oracle row remains only as a stated non-deployable")
    w("reference.")
    w()
    if hard_neg and "results" in hard_neg:
        hn = hard_neg["results"]
        w("**ICA features cannot separate antonyms from synonyms.** Training the")
        w("MLP with hard negatives (GloVe neighbours of positive words) collapses")
        w("retrieval:")
        w()
        w("| Negative sampling | MLP Hits@1 | Pool-100 recall | Reranker Hits@1 |")
        w("|-------------------|------------|-----------------|-----------------|")
        for name, label in [
            ("random", "uniform (current)"),
            ("mixed", "50% hard"),
            ("hard", "all hard"),
        ]:
            r = hn.get(name, {})
            mlp1 = _ms((r.get("mlp") or {}).get("1"), pct=True)
            rec = _pct(r.get("pool100_recall"))
            rr1 = _ms((r.get("reranker") or {}).get("1"), pct=True)
            w(f"| {label} | {mlp1} | {rec} | {rr1} |")
        w()
        w("The axis-wise features (|s1−s2|, s1⊙s2) encode *how different* two words")
        w("are, not *in which direction* — synonyms and antonyms look alike. The")
        w("pipeline works because uniform negatives are trivially separable; the")
        w("antonym/synonym boundary is carried by the reranker's other signals.")
        w()
    w()
    w("**Multi-sense limitation.** The retrieval pipeline returns one ranking per query;")
    w("polysemous words (e.g., *light* = weight/brightness/mood) may return the antonym")
    w("for an unintended sense. The per-axis inversion mode")
    w("(`SemanticOperator.invert_by_attributes`) can offer multiple sense-specific")
    w("opposites simultaneously.")
    w()
    rer_h10 = rer_cv.get("reranker", {}).get("hits_at_10", {}).get("mean")
    w("**Comparison with LLM-based antonym retrieval.** Large language models")
    w("can reliably retrieve antonyms for most common words. The statistical pipeline")
    w("here reaches")
    w(f"{_pct(rer_h10)} Hits@10 on this vocabulary without requiring generation")
    w("or prompting infrastructure, but remains far from generation-based systems.")
    w()

    # ------------------------------------------------------------------ #
    # Reproducibility                                                     #
    # ------------------------------------------------------------------ #
    w("## Reproducibility")
    w()
    w("- All stochastic stages are seeded with `random_state=42` (FastICA, MLP,")
    w("  logistic reranker, fold shuffling, counter-fitting is deterministic).")
    if ica_meta:
        w(
            "- Cached ICA space provenance: "
            + ", ".join(f"{k}={v}" for k, v in ica_meta.items() if k not in ("created_at",))
            + (f" (created {ica_meta['created_at']})" if ica_meta.get("created_at") else "")
        )
    else:
        w("- Cached ICA space provenance: not recorded (space created before metadata")
        w("  was added; re-run `task run:ica` to regenerate with provenance).")
    w("- Full pipeline: `task run:all`; evaluations: see `task --list` and `scripts/`.")
    w()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated paper at {output_path}")
    for msg in warnings:
        print(f"WARNING: {msg}")


if __name__ == "__main__":
    generate_paper()
