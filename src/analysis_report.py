"""
Analysis and visualization for ICA-decomposed embedding spaces.

Generates figures, computes interpretability metrics, and performs
bias analysis on ICA axes.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors

from src.axis_labeler import AxisProfile
from src.ica_transformer import ICASpace, ICATransformer
from src.semantic_operations import SemanticOperator


class AnalysisReport:
    """Generates analysis figures and metrics for ICA spaces."""

    def __init__(
        self,
        space: ICASpace,
        model: KeyedVectors,
        profiles: list[AxisProfile],
        operator: SemanticOperator,
        fig_dir: str = "paper/figures",
        results_dir: str = "results",
    ):
        self.space = space
        self.model = model
        self.profiles = profiles
        self.operator = operator
        self.transformer = ICATransformer()
        self.fig_dir = Path(fig_dir)
        self.results_dir = Path(results_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def axis_interpretability(self) -> dict:
        """
        Measure how many axes are interpretable.

        An axis is "interpretable" if it has a non-empty label and
        at least 3 associated antonym pairs.
        """
        labeled = [p for p in self.profiles if p.label]
        with_pairs = [p for p in labeled if len(p.antonym_pairs) >= 3]

        # Kurtosis distribution
        kurtosis_values = [p.kurtosis for p in self.profiles]

        result = {
            "total_axes": self.space.n_components,
            "labeled_axes": len(labeled),
            "well_supported_axes": len(with_pairs),
            "interpretability_rate": len(labeled) / self.space.n_components,
            "kurtosis_stats": {
                "mean": float(np.mean(kurtosis_values)),
                "median": float(np.median(kurtosis_values)),
                "max": float(np.max(kurtosis_values)),
                "min": float(np.min(kurtosis_values)),
            },
            "labeled_axis_details": [
                {
                    "idx": p.axis_idx,
                    "label": p.label,
                    "kurtosis": p.kurtosis,
                    "n_pairs": len(p.antonym_pairs),
                }
                for p in labeled
            ],
        }

        # Plot kurtosis distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        sorted_kurt = sorted(kurtosis_values, reverse=True)
        colors = [
            "#2196F3" if p.label else "#BDBDBD"
            for p in sorted(self.profiles, key=lambda p: p.kurtosis, reverse=True)
        ]
        ax.bar(range(len(sorted_kurt)), sorted_kurt, color=colors, width=1.0)
        ax.set_xlabel("Axis (sorted by kurtosis)")
        ax.set_ylabel("Kurtosis")
        ax.set_title("ICA Axis Kurtosis (blue = labeled)")
        fig.tight_layout()
        fig.savefig(self.fig_dir / "kurtosis_distribution.png", dpi=150)
        plt.close(fig)
        print("Saved kurtosis distribution plot")

        return result

    def per_axis_inversion_success(
        self,
        antonym_pairs: list[tuple[str, str]],
        top_n: int = 10,
    ) -> dict:
        """
        Compute inversion success rate per labeled axis.

        For each labeled axis, finds pairs where that axis has the largest score difference,
        then checks if inversion retrieves the antonym in top-n.
        """
        axis_results = {}

        for profile in self.profiles:
            if not profile.label or len(profile.antonym_pairs) < 3:
                continue

            pairs_to_test = profile.antonym_pairs[:50]  # cap for speed
            hits = 0
            tested = 0

            for w1, w2 in pairs_to_test:
                if self.space.score(w1) is None or self.space.score(w2) is None:
                    continue
                neighbors = self.operator.axis_invert(w1, profile.axis_idx, top_n=top_n)
                neighbor_words = [n.word for n in neighbors]
                tested += 1
                if w2 in neighbor_words:
                    hits += 1

            if tested > 0:
                axis_results[profile.label] = {
                    "axis_idx": profile.axis_idx,
                    "tested": tested,
                    "hits": hits,
                    "success_rate": hits / tested,
                }

        # Plot
        if axis_results:
            labels = list(axis_results.keys())
            rates = [axis_results[label]["success_rate"] for label in labels]
            counts = [axis_results[label]["tested"] for label in labels]

            fig, ax = plt.subplots(figsize=(12, 5))
            bars = ax.bar(range(len(labels)), rates, color="#4CAF50")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Inversion Success Rate (Hits@10)")
            ax.set_title("Per-Axis Antonym Inversion Success")
            ax.set_ylim(0, 1.0)
            for bar, count in zip(bars, counts, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            fig.tight_layout()
            fig.savefig(self.fig_dir / "per_axis_success.png", dpi=150)
            plt.close(fig)
            print("Saved per-axis success plot")

        return axis_results

    def zero_point_analysis(self, axis_indices: list[int] | None = None, top_n: int = 20) -> dict:
        """
        Find words near the zero point on specific axes (semantically neutral words).
        """
        if axis_indices is None:
            axis_indices = [p.axis_idx for p in self.profiles if p.label][:5]

        results = {}
        for axis_idx in axis_indices:
            scores = self.space.S[:, axis_idx]
            abs_scores = np.abs(scores)
            neutral_indices = np.argsort(abs_scores)[:top_n]
            neutral_words = [
                {"word": self.space.words[i], "score": float(scores[i])} for i in neutral_indices
            ]

            profile = next((p for p in self.profiles if p.axis_idx == axis_idx), None)
            label = profile.label if profile else ""
            results[f"axis_{axis_idx}_{label}"] = {
                "axis_idx": axis_idx,
                "label": label,
                "neutral_words": neutral_words,
            }

        return results

    def bias_visualization(self, axis_label: str = "gender") -> dict:
        """
        Visualize profession words along a semantic axis (e.g., gender bias).
        """
        # Find the gender axis
        profile = None
        for p in self.profiles:
            if p.label == axis_label:
                profile = p
                break
        if profile is None:
            return {"error": f"No axis labeled '{axis_label}'"}

        axis_idx = profile.axis_idx

        # Common profession words
        professions = [
            "doctor",
            "nurse",
            "engineer",
            "teacher",
            "scientist",
            "artist",
            "lawyer",
            "chef",
            "pilot",
            "mechanic",
            "librarian",
            "programmer",
            "professor",
            "athlete",
            "surgeon",
            "therapist",
            "architect",
            "dentist",
            "pharmacist",
            "accountant",
            "journalist",
            "musician",
            "carpenter",
            "electrician",
            "plumber",
            "secretary",
            "receptionist",
            "manager",
            "director",
            "president",
        ]

        scores = []
        valid_profs = []
        for prof in professions:
            s = self.space.score(prof)
            if s is not None:
                scores.append(float(s[axis_idx]))
                valid_profs.append(prof)

        if not scores:
            return {"error": "No profession words found in ICA space"}

        # Sort by score
        order = np.argsort(scores)
        sorted_profs = [valid_profs[i] for i in order]
        sorted_scores = [scores[i] for i in order]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#E91E63" if s > 0 else "#2196F3" for s in sorted_scores]
        ax.barh(range(len(sorted_profs)), sorted_scores, color=colors)
        ax.set_yticks(range(len(sorted_profs)))
        ax.set_yticklabels(sorted_profs, fontsize=9)
        ax.set_xlabel(f"ICA Score on '{axis_label}' axis")
        ax.set_title(f"Profession Scores on '{axis_label}' Axis")
        ax.axvline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(self.fig_dir / f"bias_{axis_label}.png", dpi=150)
        plt.close(fig)
        print(f"Saved bias visualization for '{axis_label}' axis")

        return {
            "axis_label": axis_label,
            "axis_idx": axis_idx,
            "professions": [
                {"word": p, "score": s} for p, s in zip(sorted_profs, sorted_scores, strict=False)
            ],
        }

    def reconstruction_quality(
        self,
        n_sample: int = 1000,
        component_grid: tuple[int, ...] = (5, 10, 25, 50, 75, 100),
    ) -> dict:
        """
        Measure how much information each number of ICA components retains.

        Keeping all k = d components is a lossless change of basis, so the
        roundtrip error is float noise by construction. To make the metric
        meaningful we reconstruct from only the k highest-energy components
        (energy = contribution variance across the vocabulary) and report
        the relative error for each k.
        """
        rng = np.random.RandomState(42)
        indices = rng.choice(
            len(self.space.words), size=min(n_sample, len(self.space.words)), replace=False
        )

        S = self.space.S[indices]  # (n_sample, n_components)
        A = self.space.mixing_matrix  # (d, n_components)
        mean_vec = self.space.mean_vector
        orig_rows = [self.model.key_to_index[self.space.words[i]] for i in indices]
        originals = self.model.vectors[np.asarray(orig_rows)].astype(np.float64)
        orig_norms = np.maximum(np.linalg.norm(originals, axis=1), 1e-10)

        # Energy of each component = mean squared contribution over sampled words
        energy = np.mean((S**2) * np.sum(A**2, axis=0)[None, :], axis=0)
        order = np.argsort(energy)[::-1]

        grid = [k for k in component_grid if k <= self.space.n_components]
        by_kept = {}
        for k in grid:
            keep = order[:k]
            S_k = np.zeros_like(S)
            S_k[:, keep] = S[:, keep]
            reconstructed = S_k @ A.T + mean_vec
            errors = np.linalg.norm(originals - reconstructed, axis=1) / orig_norms
            by_kept[k] = {
                "mean_relative_error": float(np.mean(errors)),
                "median_relative_error": float(np.median(errors)),
                "p95_relative_error": float(np.percentile(errors, 95)),
            }

        result = {
            "n_samples": len(indices),
            "n_components_total": self.space.n_components,
            "note": (
                "k = n_components_total is a lossless change of basis "
                "(error is float noise by construction)"
            ),
            "by_kept_components": by_kept,
        }

        # Plot the error curve
        ks = sorted(by_kept.keys())
        means = [by_kept[k]["mean_relative_error"] for k in ks]
        medians = [by_kept[k]["median_relative_error"] for k in ks]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ks, means, "o-", color="#FF9800", label="Mean relative error")
        ax.plot(ks, medians, "s--", color="#9C27B0", label="Median relative error")
        ax.axhline(
            by_kept[max(ks)]["mean_relative_error"],
            color="red",
            linestyle=":",
            label="k = d baseline (≈0, lossless by construction)",
        )
        ax.set_xlabel("Number of ICA components kept")
        ax.set_ylabel("Relative reconstruction error")
        ax.set_title(f"Reconstruction Error vs Components Kept (n={len(indices)})")
        ax.set_xticks(ks)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.fig_dir / "reconstruction_quality.png", dpi=150)
        plt.close(fig)
        print("Saved reconstruction quality plot")

        return result

    def generate_full_report(self, antonym_pairs: list[tuple[str, str]]) -> dict:
        """Run all analyses and save results."""
        print("\n=== Analysis Report ===\n")

        print("1. Axis interpretability...")
        interp = self.axis_interpretability()

        print("2. Per-axis inversion success...")
        per_axis = self.per_axis_inversion_success(antonym_pairs)

        print("3. Zero-point analysis...")
        zero = self.zero_point_analysis()

        print("4. Bias visualization...")
        bias = self.bias_visualization("gender")

        print("5. Reconstruction quality...")
        recon = self.reconstruction_quality()

        report = {
            "axis_interpretability": interp,
            "per_axis_success": per_axis,
            "zero_point_analysis": zero,
            "bias_analysis": bias,
            "reconstruction_quality": recon,
        }

        path = self.results_dir / "analysis_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved full analysis report to {path}")

        return report
