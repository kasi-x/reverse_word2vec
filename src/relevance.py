"""
Relevance scoring for word-axis relationships in ICA spaces.

Computes how strongly each word engages with each ICA axis (z-score),
enabling confidence-based inversion filtering and bias analysis.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.axis_labeler import AxisProfile
from src.ica_transformer import ICASpace


@dataclass
class WordAxisScore:
    """A word's engagement with a single axis."""

    axis_idx: int
    label: str
    raw_score: float  # S[w, k]
    z_score: float  # |S[w, k]| / std(axis_k)
    pole: str  # "positive" or "negative"


@dataclass
class InversionConfidence:
    """Confidence assessment for inverting a word on an axis."""

    word: str
    axis_idx: int
    axis_label: str
    z_score: float  # how strongly the word loads on this axis
    axis_rank: int  # rank of this axis among all axes for this word (1 = strongest)
    kurtosis: float  # axis structuredness
    confidence: str  # "high", "medium", "low"


class RelevanceScorer:
    """Computes word-axis relevance scores from ICA decomposition."""

    # z-score thresholds for confidence levels
    Z_HIGH = 2.0
    Z_MEDIUM = 1.0

    def __init__(self, space: ICASpace, profiles: list[AxisProfile] | None = None):
        self.space = space
        self.profiles = profiles or []
        self._label_map = {p.axis_idx: p.label for p in self.profiles}
        self._kurtosis_map = {p.axis_idx: p.kurtosis for p in self.profiles}

        # Precompute axis statistics
        self._axis_std = np.std(space.S, axis=0)
        self._axis_std = np.maximum(self._axis_std, 1e-8)
        # Full z-score matrix: (n_words, n_components)
        self._z_matrix = np.abs(space.S) / self._axis_std

    def word_profile(self, word: str, top_n: int = 10) -> list[WordAxisScore]:
        """
        Get a word's top axes ranked by z-score.

        Returns the axes where this word has the strongest engagement,
        i.e., where it deviates most from the population mean.
        """
        idx = self.space.word_to_idx.get(word)
        if idx is None:
            return []

        scores = self.space.S[idx]
        z_scores = self._z_matrix[idx]
        ranked = np.argsort(z_scores)[::-1][:top_n]

        result = []
        for k in ranked:
            result.append(
                WordAxisScore(
                    axis_idx=int(k),
                    label=self._label_map.get(int(k), ""),
                    raw_score=float(scores[k]),
                    z_score=float(z_scores[k]),
                    pole="positive" if scores[k] > 0 else "negative",
                )
            )
        return result

    def inversion_confidence(self, word: str, axis_idx: int) -> InversionConfidence | None:
        """
        Assess confidence of inverting a word on a specific axis.

        High confidence: z >= 2.0 (word is 2+ std from center on this axis)
        Medium confidence: 1.0 <= z < 2.0
        Low confidence: z < 1.0 (word barely engages with this axis)
        """
        idx = self.space.word_to_idx.get(word)
        if idx is None:
            return None

        z = float(self._z_matrix[idx, axis_idx])

        # Rank: where does this axis fall among all axes for this word?
        all_z = self._z_matrix[idx]
        rank = int((all_z >= z).sum())  # 1-indexed rank

        kurtosis = self._kurtosis_map.get(axis_idx, 0.0)

        if z >= self.Z_HIGH:
            confidence = "high"
        elif z >= self.Z_MEDIUM:
            confidence = "medium"
        else:
            confidence = "low"

        return InversionConfidence(
            word=word,
            axis_idx=axis_idx,
            axis_label=self._label_map.get(axis_idx, ""),
            z_score=z,
            axis_rank=rank,
            kurtosis=kurtosis,
            confidence=confidence,
        )

    def confident_axes(self, word: str, min_z: float | None = None) -> list[InversionConfidence]:
        """
        Find all axes where a word has high enough engagement to invert confidently.

        Args:
            word: Target word.
            min_z: Minimum z-score threshold. Defaults to Z_MEDIUM.

        Returns:
            List of InversionConfidence, sorted by z-score descending.
        """
        if min_z is None:
            min_z = self.Z_MEDIUM

        idx = self.space.word_to_idx.get(word)
        if idx is None:
            return []

        z_scores = self._z_matrix[idx]
        mask = z_scores >= min_z
        axes = np.where(mask)[0]

        results = []
        for k in axes:
            conf = self.inversion_confidence(word, int(k))
            if conf is not None:
                results.append(conf)

        results.sort(key=lambda c: c.z_score, reverse=True)
        return results

    def axis_extremes(self, axis_idx: int, n: int = 20) -> dict:
        """
        Find the most extreme words on both poles of an axis.

        Returns:
            Dict with 'positive' and 'negative' lists of (word, z_score, raw_score).
        """
        scores = self.space.S[:, axis_idx]
        z_scores = self._z_matrix[:, axis_idx]

        # Positive pole (high raw score)
        pos_indices = np.argsort(scores)[-n:][::-1]
        # Negative pole (low raw score)
        neg_indices = np.argsort(scores)[:n]

        def make_entries(indices):
            return [
                {
                    "word": self.space.words[i],
                    "z_score": float(z_scores[i]),
                    "raw_score": float(scores[i]),
                }
                for i in indices
            ]

        return {
            "axis_idx": axis_idx,
            "label": self._label_map.get(axis_idx, ""),
            "positive": make_entries(pos_indices),
            "negative": make_entries(neg_indices),
        }

    def bias_scores(
        self,
        words: list[str],
        axis_idx: int,
    ) -> list[dict]:
        """
        Compute bias scores (signed z-scores) for a set of words on an axis.

        Unlike axis_extremes (which scans the full vocabulary), this evaluates
        a specific word list — useful for measuring bias in profession words, etc.
        """
        results = []
        for word in words:
            idx = self.space.word_to_idx.get(word)
            if idx is None:
                continue
            raw = float(self.space.S[idx, axis_idx])
            signed_z = raw / float(self._axis_std[axis_idx])
            results.append(
                {
                    "word": word,
                    "raw_score": raw,
                    "signed_z": signed_z,
                    "abs_z": abs(signed_z),
                }
            )
        results.sort(key=lambda x: x["signed_z"])
        return results

    def confidence_stats(
        self,
        antonym_pairs: list[tuple[str, str]],
    ) -> dict:
        """
        Compute confidence statistics across antonym pairs.

        For each pair, finds the best axis and reports its confidence level.
        Shows what fraction of pairs can be handled at each confidence level.
        """
        z_scores = []
        by_confidence = {"high": 0, "medium": 0, "low": 0}

        for w1, w2 in antonym_pairs:
            s1 = self.space.score(w1)
            s2 = self.space.score(w2)
            if s1 is None or s2 is None:
                continue

            diff = np.abs(s1 - s2)
            norm_diff = diff / self._axis_std
            best_z = float(np.max(norm_diff))
            z_scores.append(best_z)

            if best_z >= self.Z_HIGH:
                by_confidence["high"] += 1
            elif best_z >= self.Z_MEDIUM:
                by_confidence["medium"] += 1
            else:
                by_confidence["low"] += 1

        total = len(z_scores)
        if total == 0:
            return {"error": "no valid pairs"}

        z_arr = np.array(z_scores)
        return {
            "total_pairs": total,
            "by_confidence": by_confidence,
            "by_confidence_pct": {k: v / total for k, v in by_confidence.items()},
            "z_score_stats": {
                "mean": float(np.mean(z_arr)),
                "median": float(np.median(z_arr)),
                "std": float(np.std(z_arr)),
                "p25": float(np.percentile(z_arr, 25)),
                "p75": float(np.percentile(z_arr, 75)),
                "p90": float(np.percentile(z_arr, 90)),
            },
            "z_scores": z_scores,  # for plotting
        }

    # ── Visualization ──────────────────────────────────────────────

    def plot_word_axis_heatmap(
        self,
        words: list[str],
        axis_indices: list[int] | None = None,
        path: str = "paper/figures/relevance_heatmap.png",
    ) -> None:
        """
        Plot a heatmap of signed z-scores for selected words × axes.

        If axis_indices is None, auto-selects the top axes that have
        the highest max z-score across the given words.
        """
        # Filter to valid words
        valid = [(w, self.space.word_to_idx[w]) for w in words if w in self.space.word_to_idx]
        if not valid:
            return
        word_labels, word_indices = zip(*valid, strict=True)

        if axis_indices is None:
            # Pick axes with highest max engagement across these words
            sub_z = self._z_matrix[list(word_indices)]  # (n_words, n_components)
            max_per_axis = sub_z.max(axis=0)
            axis_indices = list(np.argsort(max_per_axis)[-15:][::-1])

        # Build signed z matrix for display
        data = np.zeros((len(word_labels), len(axis_indices)))
        for i, widx in enumerate(word_indices):
            for j, aidx in enumerate(axis_indices):
                raw = self.space.S[widx, aidx]
                data[i, j] = raw / self._axis_std[aidx]

        axis_labels = []
        for aidx in axis_indices:
            label = self._label_map.get(aidx, "")
            axis_labels.append(f"{aidx}:{label}" if label else str(aidx))

        fig, ax = plt.subplots(
            figsize=(max(8, len(axis_indices) * 0.7), max(4, len(word_labels) * 0.4))
        )
        vmax = max(abs(data.min()), abs(data.max()), 1.0)
        im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(axis_labels)))
        ax.set_xticklabels(axis_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(word_labels)))
        ax.set_yticklabels(word_labels, fontsize=9)
        ax.set_title("Word-Axis Relevance (signed z-score)")

        # Annotate cells
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if abs(val) > 0.5:
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8, label="signed z-score")
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved relevance heatmap to {path}")

    def plot_confidence_distribution(
        self,
        antonym_pairs: list[tuple[str, str]],
        path: str = "paper/figures/confidence_distribution.png",
    ) -> dict:
        """
        Plot the distribution of confidence z-scores across antonym pairs.

        Shows what fraction of pairs fall into high/medium/low confidence bands.
        """
        stats = self.confidence_stats(antonym_pairs)
        if "error" in stats:
            return stats

        z_scores = np.array(stats["z_scores"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: histogram with confidence bands
        ax = axes[0]
        ax.hist(z_scores, bins=50, color="#78909C", edgecolor="white", alpha=0.8)
        ax.axvline(
            self.Z_HIGH,
            color="#4CAF50",
            linewidth=2,
            linestyle="--",
            label=f"high (z>{self.Z_HIGH})",
        )
        ax.axvline(
            self.Z_MEDIUM,
            color="#FF9800",
            linewidth=2,
            linestyle="--",
            label=f"medium (z>{self.Z_MEDIUM})",
        )
        ax.set_xlabel("Best-axis z-score per antonym pair")
        ax.set_ylabel("Count")
        ax.set_title("Confidence Distribution")
        ax.legend()

        # Right: pie chart of confidence levels
        ax = axes[1]
        sizes = [stats["by_confidence"][k] for k in ["high", "medium", "low"]]
        labels = [
            f"High ({stats['by_confidence_pct']['high']:.0%})",
            f"Medium ({stats['by_confidence_pct']['medium']:.0%})",
            f"Low ({stats['by_confidence_pct']['low']:.0%})",
        ]
        colors = ["#4CAF50", "#FF9800", "#F44336"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90)
        ax.set_title(f"Confidence Levels (n={stats['total_pairs']})")

        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved confidence distribution to {path}")

        # Remove raw z_scores from returned stats (too large for JSON)
        stats_copy = {k: v for k, v in stats.items() if k != "z_scores"}
        return stats_copy

    def plot_bias_profile(
        self,
        words: list[str],
        axis_idx: int,
        path: str | None = None,
    ) -> list[dict]:
        """
        Plot signed z-scores for a word list on a specific axis.

        Returns the bias scores for further use.
        """
        scores = self.bias_scores(words, axis_idx)
        if not scores:
            return []

        label = self._label_map.get(axis_idx, f"axis_{axis_idx}")
        if path is None:
            path = f"paper/figures/bias_profile_{label}.png"

        fig, ax = plt.subplots(figsize=(10, max(4, len(scores) * 0.3)))
        word_labels = [s["word"] for s in scores]
        z_values = [s["signed_z"] for s in scores]
        colors = ["#E91E63" if z > 0 else "#2196F3" for z in z_values]

        ax.barh(range(len(word_labels)), z_values, color=colors)
        ax.set_yticks(range(len(word_labels)))
        ax.set_yticklabels(word_labels, fontsize=9)
        ax.set_xlabel(f"Signed z-score on '{label}' axis")
        ax.set_title(f"Bias Profile: '{label}' axis (z-score)")
        ax.axvline(0, color="black", linewidth=0.5)

        # Mark confidence thresholds
        ax.axvline(self.Z_HIGH, color="#4CAF50", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.axvline(-self.Z_HIGH, color="#4CAF50", linewidth=0.8, linestyle=":", alpha=0.5)

        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved bias profile to {path}")

        return scores

    # ── JSON export ────────────────────────────────────────────────

    def generate_report(
        self,
        antonym_pairs: list[tuple[str, str]],
        sample_words: list[str] | None = None,
        fig_dir: str = "paper/figures",
        results_dir: str = "results",
    ) -> dict:
        """
        Run full relevance analysis: confidence stats, heatmap, bias profiles.

        Returns report dict and saves figures + JSON.
        """
        print("\n=== Relevance Analysis ===\n")

        if sample_words is None:
            sample_words = [
                "king",
                "queen",
                "man",
                "woman",
                "hot",
                "cold",
                "warm",
                "cool",
                "good",
                "bad",
                "happy",
                "sad",
                "big",
                "small",
                "huge",
                "tiny",
                "doctor",
                "nurse",
                "engineer",
                "teacher",
                "alive",
                "dead",
                "young",
                "old",
            ]

        # 1. Confidence distribution
        print("1. Confidence distribution across antonym pairs...")
        conf_stats = self.plot_confidence_distribution(
            antonym_pairs,
            path=f"{fig_dir}/confidence_distribution.png",
        )

        # 2. Word-axis heatmap
        print("2. Word-axis relevance heatmap...")
        self.plot_word_axis_heatmap(
            sample_words,
            path=f"{fig_dir}/relevance_heatmap.png",
        )

        # 3. Bias profiles for labeled axes
        print("3. Bias profiles on labeled axes...")
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
        bias_results = {}
        for profile in self.profiles:
            if profile.label in ("gender", "sentiment", "age"):
                scores = self.plot_bias_profile(
                    professions,
                    profile.axis_idx,
                    path=f"{fig_dir}/bias_profile_{profile.label}.png",
                )
                bias_results[profile.label] = scores

        # 4. Sample word profiles
        print("4. Sample word profiles...")
        word_profiles = {}
        for word in sample_words[:10]:
            profile = self.word_profile(word, top_n=5)
            if profile:
                word_profiles[word] = [
                    {
                        "axis": s.axis_idx,
                        "label": s.label,
                        "z_score": round(s.z_score, 3),
                        "raw_score": round(s.raw_score, 3),
                        "pole": s.pole,
                    }
                    for s in profile
                ]

        report = {
            "confidence_stats": conf_stats,
            "bias_profiles": {
                label: [{"word": s["word"], "signed_z": round(s["signed_z"], 3)} for s in scores]
                for label, scores in bias_results.items()
            },
            "word_profiles": word_profiles,
        }

        # Save JSON
        results_path = Path(results_dir) / "relevance_analysis.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved relevance analysis to {results_path}")

        return report
