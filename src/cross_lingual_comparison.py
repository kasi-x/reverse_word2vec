"""
Cross-lingual comparison of English and Japanese antonym vector spaces.

Compares structural properties:
- SVD variance curves
- Detection accuracy
- Neutral word categories
- Translation pair correspondence

Statistical tests:
- Kolmogorov-Smirnov test on SVD variance curves
- Mann-Whitney U test on score distributions
- Spearman correlation on translation pair scores
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")

# Translation pairs: (English word, Japanese word) for direct comparison
TRANSLATION_PAIRS = [
    # Adjectives
    ("good", "良い", "bad", "悪い"),
    ("big", "大きい", "small", "小さい"),
    ("long", "長い", "short", "短い"),
    ("strong", "強い", "weak", "弱い"),
    ("hot", "暑い", "cold", "寒い"),
    ("light", "明るい", "dark", "暗い"),
    ("rich", "豊か", "poor", "貧しい"),
    ("deep", "深い", "shallow", "浅い"),
    # Verbs
    ("buy", "買う", "sell", "売る"),
    ("win", "勝つ", "lose", "負ける"),
    ("open", "開く", "close", "閉じる"),
    ("increase", "増える", "decrease", "減る"),
    # Nouns
    ("male", "男", "female", "女"),
    ("peace", "平和", "war", "戦争"),
    ("life", "生", "death", "死"),
    ("heaven", "天", "earth", "地"),
]


def load_results(lang: str) -> dict:
    """Load analysis results for a language."""
    path = os.path.join(RESULTS_DIR, f"{lang}_analysis.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results not found: {path}. Run the analysis first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_svd_structure(en: dict, ja: dict) -> dict:
    """Compare SVD variance structure between languages."""
    en_cumvar = np.array(en["svd_analysis"]["cumulative_variance"])
    ja_cumvar = np.array(ja["svd_analysis"]["cumulative_variance"])

    # Normalize to same length for comparison
    min_len = min(len(en_cumvar), len(ja_cumvar))
    en_trimmed = en_cumvar[:min_len]
    ja_trimmed = ja_cumvar[:min_len]

    # KS test on cumulative variance curves
    ks_stat, ks_pvalue = stats.ks_2samp(en_cumvar, ja_cumvar)

    return {
        "en_dims_90pct": en["svd_analysis"]["dims_for_90_pct"],
        "ja_dims_90pct": ja["svd_analysis"]["dims_for_90_pct"],
        "en_dims_95pct": en["svd_analysis"]["dims_for_95_pct"],
        "ja_dims_95pct": ja["svd_analysis"]["dims_for_95_pct"],
        "en_total_directions": en["svd_analysis"]["total_directions"],
        "ja_total_directions": ja["svd_analysis"]["total_directions"],
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "en_cumulative_variance": en_cumvar.tolist(),
        "ja_cumulative_variance": ja_cumvar.tolist(),
    }


def compare_detection_accuracy(en: dict, ja: dict) -> dict:
    """Compare antonym detection accuracy."""
    en_eval = en["comprehensive_evaluation"]
    ja_eval = ja["comprehensive_evaluation"]

    return {
        "en_accuracy_at_1": en_eval["accuracy_at_1"],
        "ja_accuracy_at_1": ja_eval["accuracy_at_1"],
        "en_accuracy_at_5": en_eval["accuracy_at_5"],
        "ja_accuracy_at_5": ja_eval["accuracy_at_5"],
        "en_accuracy_at_10": en_eval["accuracy_at_10"],
        "ja_accuracy_at_10": ja_eval["accuracy_at_10"],
        "en_sample_size": en_eval["sample_size"],
        "ja_sample_size": ja_eval["sample_size"],
    }


def compare_score_distributions(en: dict, ja: dict) -> dict:
    """Compare score distributions using Mann-Whitney U test."""
    en_scores = [p["score"] for p in en["known_pairs_evaluation"]["pair_results"] if p["score"] > 0]
    ja_scores = [p["score"] for p in ja["known_pairs_evaluation"]["pair_results"] if p["score"] > 0]

    if en_scores and ja_scores:
        u_stat, u_pvalue = stats.mannwhitneyu(en_scores, ja_scores, alternative="two-sided")
    else:
        u_stat, u_pvalue = 0.0, 1.0

    return {
        "en_mean_score": float(np.mean(en_scores)) if en_scores else 0,
        "ja_mean_score": float(np.mean(ja_scores)) if ja_scores else 0,
        "en_median_score": float(np.median(en_scores)) if en_scores else 0,
        "ja_median_score": float(np.median(ja_scores)) if ja_scores else 0,
        "mann_whitney_u": u_stat,
        "mann_whitney_pvalue": u_pvalue,
        "en_scores": en_scores,
        "ja_scores": ja_scores,
    }


def analyze_translation_pairs(en: dict, ja: dict) -> dict:
    """Analyze correspondence between translation pairs."""
    en_pair_results = {(p["word1"], p["word2"]): p
                       for p in en["known_pairs_evaluation"]["pair_results"]}

    ja_pair_results = {(p["word1"], p["word2"]): p
                       for p in ja["known_pairs_evaluation"]["pair_results"]}

    correspondences = []
    en_scores = []
    ja_scores = []

    for en_w1, ja_w1, en_w2, ja_w2 in TRANSLATION_PAIRS:
        en_result = en_pair_results.get((en_w1, en_w2))
        ja_result = ja_pair_results.get((ja_w1, ja_w2))

        en_score = en_result["score"] if en_result else None
        ja_score = ja_result["score"] if ja_result else None

        correspondences.append({
            "en_pair": f"{en_w1}/{en_w2}",
            "ja_pair": f"{ja_w1}/{ja_w2}",
            "en_score": en_score,
            "ja_score": ja_score,
            "en_rank": en_result["rank"] if en_result else None,
            "ja_rank": ja_result["rank"] if ja_result else None,
        })

        if en_score is not None and ja_score is not None and en_score > 0 and ja_score > 0:
            en_scores.append(en_score)
            ja_scores.append(ja_score)

    # Spearman correlation
    if len(en_scores) >= 3:
        spearman_r, spearman_p = stats.spearmanr(en_scores, ja_scores)
    else:
        spearman_r, spearman_p = 0.0, 1.0

    return {
        "correspondences": correspondences,
        "n_comparable": len(en_scores),
        "spearman_r": spearman_r,
        "spearman_pvalue": spearman_p,
        "en_scores": en_scores,
        "ja_scores": ja_scores,
    }


def plot_cumulative_variance(svd_comparison: dict, output_path: str):
    """Plot cumulative variance curves for both languages."""
    fig, ax = plt.subplots(figsize=(8, 5))

    en_cv = svd_comparison["en_cumulative_variance"]
    ja_cv = svd_comparison["ja_cumulative_variance"]

    ax.plot(range(1, len(en_cv) + 1), en_cv, label="English (GloVe-100)", color="#2196F3", linewidth=2)
    ax.plot(range(1, len(ja_cv) + 1), ja_cv, label="Japanese (fastText-300)", color="#F44336", linewidth=2)

    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.7, label="90% threshold")
    ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5, label="95% threshold")

    # Mark 90% points
    en_90 = svd_comparison["en_dims_90pct"]
    ja_90 = svd_comparison["ja_dims_90pct"]
    ax.axvline(x=en_90, color="#2196F3", linestyle="--", alpha=0.3)
    ax.axvline(x=ja_90, color="#F44336", linestyle="--", alpha=0.3)
    ax.annotate(f"EN: {en_90}d", (en_90, 0.85), color="#2196F3", fontsize=9)
    ax.annotate(f"JA: {ja_90}d", (ja_90, 0.82), color="#F44336", fontsize=9)

    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("SVD Variance Structure: English vs Japanese Antonym Spaces")
    ax.legend(loc="lower right")
    ax.set_xlim(0, min(200, max(len(en_cv), len(ja_cv))))
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_score_distributions(score_comparison: dict, output_path: str):
    """Plot score distribution histograms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    en_scores = score_comparison["en_scores"]
    ja_scores = score_comparison["ja_scores"]

    ax1.hist(en_scores, bins=20, color="#2196F3", alpha=0.7, edgecolor="white")
    ax1.set_title("English Antonym Scores")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Count")
    ax1.axvline(x=score_comparison["en_mean_score"], color="red", linestyle="--",
                label=f'Mean: {score_comparison["en_mean_score"]:.3f}')
    ax1.legend()

    ax2.hist(ja_scores, bins=20, color="#F44336", alpha=0.7, edgecolor="white")
    ax2.set_title("Japanese Antonym Scores")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Count")
    ax2.axvline(x=score_comparison["ja_mean_score"], color="blue", linestyle="--",
                label=f'Mean: {score_comparison["ja_mean_score"]:.3f}')
    ax2.legend()

    fig.suptitle("Score Distributions: Known Antonym Pairs", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_translation_scatter(translation_analysis: dict, output_path: str):
    """Plot scatter plot of EN vs JA scores for translation pairs."""
    fig, ax = plt.subplots(figsize=(7, 7))

    en_scores = translation_analysis["en_scores"]
    ja_scores = translation_analysis["ja_scores"]
    correspondences = [c for c in translation_analysis["correspondences"]
                       if c["en_score"] and c["ja_score"] and c["en_score"] > 0 and c["ja_score"] > 0]

    ax.scatter(en_scores, ja_scores, c="#673AB7", s=80, alpha=0.7, edgecolors="white", linewidth=0.5)

    # Label points
    for c in correspondences:
        ax.annotate(c["en_pair"], (c["en_score"], c["ja_score"]),
                    fontsize=7, ha="center", va="bottom", alpha=0.8)

    # Diagonal reference line
    lim = [0, max(max(en_scores, default=1), max(ja_scores, default=1)) * 1.05]
    ax.plot(lim, lim, "k--", alpha=0.3, label="y=x")

    r = translation_analysis["spearman_r"]
    p = translation_analysis["spearman_pvalue"]
    ax.set_title(f"Translation Pair Scores (Spearman r={r:.3f}, p={p:.3f})")
    ax.set_xlabel("English Score")
    ax.set_ylabel("Japanese Score")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_accuracy_comparison(accuracy_comparison: dict, output_path: str):
    """Plot accuracy comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ["Hits@1", "Hits@5", "Hits@10"]
    en_vals = [
        accuracy_comparison["en_accuracy_at_1"],
        accuracy_comparison["en_accuracy_at_5"],
        accuracy_comparison["en_accuracy_at_10"],
    ]
    ja_vals = [
        accuracy_comparison["ja_accuracy_at_1"],
        accuracy_comparison["ja_accuracy_at_5"],
        accuracy_comparison["ja_accuracy_at_10"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, en_vals, width, label="English", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + width/2, ja_vals, width, label="Japanese", color="#F44336", alpha=0.8)

    ax.set_ylabel("Accuracy")
    ax.set_title("Antonym Detection Accuracy: English vs Japanese")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 70)
    print("CROSS-LINGUAL COMPARISON: English vs Japanese")
    print("=" * 70)

    # Load results
    print("\n[1/5] Loading results...")
    en = load_results("english")
    ja = load_results("japanese")
    print(f"  English: {en['model']} ({en['model_dimensions']}d, {en['valid_antonym_pairs']} pairs)")
    print(f"  Japanese: {ja['model']} ({ja['model_dimensions']}d, {ja['valid_antonym_pairs']} pairs)")

    # Compare SVD structure
    print("\n[2/5] Comparing SVD structure...")
    svd_comp = compare_svd_structure(en, ja)
    print(f"  English: {svd_comp['en_dims_90pct']} dims for 90% variance "
          f"(of {svd_comp['en_total_directions']} directions)")
    print(f"  Japanese: {svd_comp['ja_dims_90pct']} dims for 90% variance "
          f"(of {svd_comp['ja_total_directions']} directions)")
    print(f"  KS test: statistic={svd_comp['ks_statistic']:.4f}, p={svd_comp['ks_pvalue']:.4f}")

    # Compare detection accuracy
    print("\n[3/5] Comparing detection accuracy...")
    acc_comp = compare_detection_accuracy(en, ja)
    print(f"  English Hits@1: {acc_comp['en_accuracy_at_1']:.1%} (n={acc_comp['en_sample_size']})")
    print(f"  Japanese Hits@1: {acc_comp['ja_accuracy_at_1']:.1%} (n={acc_comp['ja_sample_size']})")

    # Compare score distributions
    print("\n[4/5] Comparing score distributions...")
    score_comp = compare_score_distributions(en, ja)
    print(f"  English mean score: {score_comp['en_mean_score']:.3f}")
    print(f"  Japanese mean score: {score_comp['ja_mean_score']:.3f}")
    print(f"  Mann-Whitney U: statistic={score_comp['mann_whitney_u']:.1f}, "
          f"p={score_comp['mann_whitney_pvalue']:.4f}")

    # Translation pair analysis
    print("\n[5/5] Analyzing translation pairs...")
    trans_comp = analyze_translation_pairs(en, ja)
    print(f"  Comparable pairs: {trans_comp['n_comparable']}")
    print(f"  Spearman correlation: r={trans_comp['spearman_r']:.3f}, "
          f"p={trans_comp['spearman_pvalue']:.4f}")

    print("\n  Translation pair details:")
    for c in trans_comp["correspondences"]:
        en_s = f"{c['en_score']:.3f}" if c["en_score"] else "N/A"
        ja_s = f"{c['ja_score']:.3f}" if c["ja_score"] else "N/A"
        en_r = f"R{c['en_rank']}" if c["en_rank"] else "N/F"
        ja_r = f"R{c['ja_rank']}" if c["ja_rank"] else "N/F"
        print(f"    {c['en_pair']:20} ({en_s}, {en_r}) | "
              f"{c['ja_pair']:12} ({ja_s}, {ja_r})")

    # Generate plots
    print("\n[Generating plots...]")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plot_cumulative_variance(svd_comp, os.path.join(FIGURES_DIR, "svd_variance_comparison.png"))
    plot_score_distributions(score_comp, os.path.join(FIGURES_DIR, "score_distributions.png"))
    plot_accuracy_comparison(acc_comp, os.path.join(FIGURES_DIR, "accuracy_comparison.png"))

    if trans_comp["n_comparable"] >= 3:
        plot_translation_scatter(trans_comp, os.path.join(FIGURES_DIR, "translation_scatter.png"))

    # Save comparison results
    comparison_results = {
        "svd_comparison": {
            k: v for k, v in svd_comp.items()
            if k not in ("en_cumulative_variance", "ja_cumulative_variance")
        },
        "accuracy_comparison": acc_comp,
        "score_comparison": {
            k: v for k, v in score_comp.items()
            if k not in ("en_scores", "ja_scores")
        },
        "translation_pairs": {
            k: v for k, v in trans_comp.items()
            if k not in ("en_scores", "ja_scores")
        },
    }

    output_path = os.path.join(RESULTS_DIR, "cross_lingual_comparison.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    print(f"\nComparison results saved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'Metric':<30} {'English':>10} {'Japanese':>10}")
    print(f"  {'─'*30} {'─'*10} {'─'*10}")
    print(f"  {'Model dimensions':<30} {en['model_dimensions']:>10} {ja['model_dimensions']:>10}")
    print(f"  {'Antonym pairs':<30} {en['valid_antonym_pairs']:>10} {ja['valid_antonym_pairs']:>10}")
    print(f"  {'Antonym directions':<30} {en['n_antonym_directions']:>10} {ja['n_antonym_directions']:>10}")
    print(f"  {'Dims for 90% variance':<30} {svd_comp['en_dims_90pct']:>10} {svd_comp['ja_dims_90pct']:>10}")
    print(f"  {'Hits@1 (comprehensive)':<30} {acc_comp['en_accuracy_at_1']:>10.1%} {acc_comp['ja_accuracy_at_1']:>10.1%}")
    print(f"  {'Hits@10 (comprehensive)':<30} {acc_comp['en_accuracy_at_10']:>10.1%} {acc_comp['ja_accuracy_at_10']:>10.1%}")
    print(f"  {'Mean antonym score':<30} {score_comp['en_mean_score']:>10.3f} {score_comp['ja_mean_score']:>10.3f}")
    print(f"  {'Query time (ms)':<30} {en['query_time_ms']:>10.1f} {ja['query_time_ms']:>10.1f}")
    print()


if __name__ == "__main__":
    main()
