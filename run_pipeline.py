"""
ICA Semantic Inversion Pipeline.

Usage:
    pixi run python run_pipeline.py                    # Run all phases
    pixi run python run_pipeline.py --phase 1          # Phase 1 only (ICA fit + labeling)
    pixi run python run_pipeline.py --phase 3          # Quantitative eval
    pixi run python run_pipeline.py --load-space       # Load cached ICA space
    pixi run python run_pipeline.py --model glove-300  # Use GloVe-300
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICA Semantic Inversion Pipeline")
    parser.add_argument(
        "--model",
        default="glove-100",
        choices=["glove-50", "glove-100", "glove-200", "glove-300", "google-news"],
        help="Word embedding model to use (default: glove-100)",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=None,
        help="Number of ICA components (default: model dimension)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5],
        help="Run a specific phase only (default: all)",
    )
    parser.add_argument(
        "--load-space",
        action="store_true",
        help="Load cached ICA space from results/ica_space instead of fitting",
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Re-run axis labeling on the cached ICA space (no refit) and exit",
    )
    parser.add_argument(
        "--vocab-limit",
        type=int,
        default=50000,
        help="Max vocabulary words for ICA (default: 50000)",
    )
    parser.add_argument(
        "--counter-fit",
        action="store_true",
        help="Apply counter-fitting to push antonym vectors apart before ICA",
    )
    parser.add_argument(
        "--cf-iter",
        type=int,
        default=100,
        help="Counter-fitting iterations (default: 100)",
    )
    parser.add_argument(
        "--cf-target",
        type=float,
        default=-0.3,
        help="Target antonym cosine similarity (default: -0.3)",
    )
    return parser.parse_args()


def load_model(model_name: str):
    """Load word embedding model."""
    from src.word2vec_loader import Word2VecLoader

    loader = Word2VecLoader()
    if model_name == "google-news":
        return loader.load_google_news()
    else:
        dim = int(model_name.split("-")[1])
        return loader.load_glove(dim)


def phase1_ica(model, args) -> tuple:
    """Phase 1: ICA decomposition + axis labeling."""
    from src.antonym_loader import extract_antonym_pairs
    from src.axis_labeler import AxisLabeler
    from src.ica_transformer import ICATransformer, save_ica_space

    print("\n" + "=" * 60)
    print("PHASE 1: ICA Decomposition + Axis Labeling")
    print("=" * 60)

    # Optional counter-fitting before ICA
    if args.counter_fit:
        from src.counter_fitting import CounterFitConfig, CounterFitter

        print("\nApplying counter-fitting...")
        pairs_for_cf = extract_antonym_pairs()
        cfg = CounterFitConfig(
            n_iter=args.cf_iter,
            target_sim=args.cf_target,
            verbose=True,
        )
        fitter = CounterFitter(cfg)
        model = fitter.fit(model, pairs_for_cf)
        print("Counter-fitting complete.\n")

    transformer = ICATransformer()
    space = transformer.fit(
        model,
        n_components=args.components,
        vocab_limit=args.vocab_limit,
    )

    # Save ICA space (with provenance so results are reproducible/traceable)
    save_ica_space(
        space,
        "results/ica_space",
        meta={
            "model": args.model,
            "counter_fitted": bool(args.counter_fit),
            "vocab_limit": args.vocab_limit,
            "n_components": space.n_components,
            "random_state": 42,
        },
    )

    # Load antonym pairs for labeling
    print("\nExtracting WordNet antonym pairs...")
    antonym_pairs = extract_antonym_pairs()
    print(f"Found {len(antonym_pairs)} antonym pairs")

    # Filter to pairs in ICA vocabulary
    valid_pairs = [
        (w1, w2)
        for w1, w2 in antonym_pairs
        if space.score(w1) is not None and space.score(w2) is not None
    ]
    print(f"Valid pairs in ICA space: {len(valid_pairs)}")

    # Label axes
    labeler = AxisLabeler(space)
    profiles = labeler.profile_all_axes()
    profiles = labeler.auto_label(valid_pairs, profiles)
    labeler.print_summary(profiles)
    labeler.save_profiles(profiles, "results/axis_profiles.json")

    return space, profiles, valid_pairs


def phase2_qualitative(model, space, profiles) -> dict:
    """Phase 2: Qualitative evaluation."""
    from src.qualitative_eval import QualitativeEvaluator, save_qualitative_results
    from src.semantic_operations import SemanticOperator

    print("\n" + "=" * 60)
    print("PHASE 2: Qualitative Evaluation")
    print("=" * 60)

    operator = SemanticOperator(model, space, profiles)
    evaluator = QualitativeEvaluator(operator, space, profiles)

    results = evaluator.run_all()

    blind = results["blind"]["label"]
    auto = results["blind"]["auto"]
    oracle = results["oracle"]
    print(f"\nBlind (declared label): {blind['evaluated']} cases evaluated")
    for k_str, val in blind["metrics"].items():
        print(f"  {k_str}: {val:.1%}")
    print(f"Blind (auto, source top-1 axis): {auto['evaluated']} cases evaluated")
    for k_str, val in auto["metrics"].items():
        print(f"  {k_str}: {val:.1%}")
    print(f"Oracle upper bound: {oracle['evaluated']} cases evaluated")
    for k_str, val in oracle["metrics"].items():
        print(f"  {k_str}: {val:.1%}")

    # Compare with traditional analogy (blind ICA side)
    comparison = evaluator.compare_with_analogy()
    results["analogy_comparison"] = comparison

    # Failure analysis on the oracle run (diagnostic only)
    failures = evaluator.failure_analysis(oracle)
    results["failure_analysis"] = failures
    if failures["total_failures"] > 0:
        print(f"\n{failures['total_failures']} oracle failures (diagnostic):")
        for f in failures["failures"][:5]:
            reason = f.get("reason", f"top-3: {f.get('actual_top3', [])}")
            print(f"  {f['word']} -> {f['expected']}: {reason}")

    save_qualitative_results(results, "results/qualitative_eval.json")
    return results


def phase3_quantitative(model, space, antonym_pairs) -> dict:
    """Phase 3: Quantitative evaluation."""
    from src.quantitative_eval import (
        AnalogyEval,
        AntonymRetrievalEval,
        save_quantitative_results,
    )
    from src.semantic_operations import SemanticOperator

    print("\n" + "=" * 60)
    print("PHASE 3: Quantitative Evaluation")
    print("=" * 60)

    operator = SemanticOperator(model, space)  # no profiles needed for quantitative

    # Antonym retrieval with cross-validation
    print("\nAntonym retrieval (5-fold CV):")
    antonym_eval = AntonymRetrievalEval(operator, space)
    cv_results = antonym_eval.cross_validate(antonym_pairs)
    save_quantitative_results(cv_results, "results/antonym_retrieval.json")

    # Google Analogy Dataset (if available)
    analogy_path = Path("data/questions-words.txt")
    if analogy_path.exists():
        print("\nGoogle Analogy Dataset:")
        analogy_eval = AnalogyEval(operator, model, space)
        analogy_results = analogy_eval.evaluate(str(analogy_path))
        save_quantitative_results(analogy_results, "results/analogy_eval.json")
    else:
        print(f"\nSkipping analogy eval: {analogy_path} not found")
        print("Run 'task download-analogy' to download it.")
        analogy_results = None

    return {"antonym_retrieval": cv_results, "analogy": analogy_results}


def phase4_analysis(model, space, profiles, antonym_pairs) -> dict:
    """Phase 4: Analysis and visualization."""
    from src.analysis_report import AnalysisReport
    from src.relevance import RelevanceScorer
    from src.semantic_operations import SemanticOperator

    print("\n" + "=" * 60)
    print("PHASE 4: Analysis & Visualization")
    print("=" * 60)

    operator = SemanticOperator(model, space, profiles)
    report = AnalysisReport(space, model, profiles, operator)
    analysis = report.generate_full_report(antonym_pairs)

    # Relevance analysis (confidence scoring + bias profiles)
    scorer = RelevanceScorer(space, profiles)
    relevance = scorer.generate_report(antonym_pairs)
    analysis["relevance_analysis"] = relevance

    return analysis


def phase5_paper() -> None:
    """Phase 5: Paper generation."""
    from src.generate_paper import generate_paper

    print("\n" + "=" * 60)
    print("PHASE 5: Paper Generation")
    print("=" * 60)

    generate_paper()


def _load_cached_space() -> tuple:
    """Load cached ICA space + profiles + valid pairs (no refit)."""
    from src.antonym_loader import extract_antonym_pairs
    from src.axis_labeler import AxisLabeler
    from src.ica_transformer import load_ica_space

    print("\nLoading cached ICA space...")
    space = load_ica_space("results/ica_space")
    labeler = AxisLabeler(space)
    profiles = labeler.load_profiles("results/axis_profiles.json")
    antonym_pairs = extract_antonym_pairs()
    valid_pairs = [
        (w1, w2)
        for w1, w2 in antonym_pairs
        if space.score(w1) is not None and space.score(w2) is not None
    ]
    return space, profiles, valid_pairs


def relabel_axes() -> None:
    """Re-run axis labeling on the cached ICA space (no refit, no model load)."""
    from src.axis_labeler import AxisLabeler

    print("\n" + "=" * 60)
    print("RELABEL: axis labeling on cached ICA space")
    print("=" * 60)

    space, _, valid_pairs = _load_cached_space()
    print(f"Valid pairs in ICA space: {len(valid_pairs)}")

    labeler = AxisLabeler(space)
    profiles = labeler.profile_all_axes()
    profiles = labeler.auto_label(valid_pairs, profiles)
    labeler.print_summary(profiles)
    labeler.save_profiles(profiles, "results/axis_profiles.json")


def main() -> None:
    args = parse_args()

    if args.relabel:
        relabel_axes()
        return

    print(f"Model: {args.model}")
    print(f"Components: {args.components or 'auto (model dim)'}")
    print(f"Vocab limit: {args.vocab_limit}")

    # Load model
    model = load_model(args.model)

    # Load or fit ICA space
    if args.load_space:
        space, profiles, valid_pairs = _load_cached_space()
    else:
        if args.phase is None or args.phase == 1:
            space, profiles, valid_pairs = phase1_ica(model, args)
        else:
            # Need ICA space for later phases
            space, profiles, valid_pairs = _load_cached_space()

    if args.phase == 1:
        return

    if args.phase is None or args.phase == 2:
        phase2_qualitative(model, space, profiles)
    if args.phase == 2:
        return

    if args.phase is None or args.phase == 3:
        phase3_quantitative(model, space, valid_pairs)
    if args.phase == 3:
        return

    if args.phase is None or args.phase == 4:
        phase4_analysis(model, space, profiles, valid_pairs)
    if args.phase == 4:
        return

    if args.phase is None or args.phase == 5:
        phase5_paper()

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
