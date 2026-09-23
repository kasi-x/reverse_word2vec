"""
Qualitative evaluation of ICA axis-wise semantic inversion.

Tests canonical examples (king->queen, hot->cold, etc.) and compares
ICA inversion with traditional vector analogy.

Evaluation modes:
- Blind (primary): invert the axes of the declared semantic label, without
  ever consulting the expected target word for axis selection. Also reports
  a fully automatic variant (flip the source word's highest-|z| axis).
- Oracle (upper bound): use the known (word, expected) pair to find the
  best axis. Not deployable; reported only as an upper bound.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.axis_labeler import AxisProfile
from src.ica_transformer import ICASpace, ICATransformer
from src.semantic_operations import SemanticOperator


@dataclass
class TestCase:
    """A single qualitative test case."""

    word: str
    axis_label: str
    expected: str
    operation: str = "invert"  # "invert" or "analogy"
    # For analogy: a->b as word->expected
    analogy_a: str = ""
    analogy_b: str = ""


# Canonical test cases for axis-wise inversion
CANONICAL_TEST_CASES = [
    # Gender axis
    TestCase("king", "gender", "queen", analogy_a="man", analogy_b="woman"),
    TestCase("boy", "gender", "girl", analogy_a="man", analogy_b="woman"),
    TestCase("father", "gender", "mother", analogy_a="man", analogy_b="woman"),
    TestCase("husband", "gender", "wife", analogy_a="man", analogy_b="woman"),
    # Sentiment axis
    TestCase("happy", "sentiment", "sad"),
    TestCase("good", "sentiment", "bad"),
    TestCase("love", "sentiment", "hate"),
    # Temperature axis
    TestCase("hot", "temperature", "cold"),
    TestCase("warm", "temperature", "cool"),
    # Size axis
    TestCase("big", "size", "small"),
    TestCase("huge", "size", "tiny"),
    # Direction axis
    TestCase("up", "direction", "down"),
    TestCase("high", "direction", "low"),
    # Activity axis
    TestCase("alive", "activity", "dead"),
    TestCase("begin", "activity", "end"),
]


class QualitativeEvaluator:
    """Runs qualitative tests and compares ICA inversion with baselines."""

    def __init__(
        self,
        operator: SemanticOperator,
        space: ICASpace,
        profiles: list[AxisProfile],
    ):
        self.operator = operator
        self.space = space
        self.profiles = profiles
        self.transformer = ICATransformer()
        self._label_to_axes = self._build_label_map()
        self._profile_by_axis = {p.axis_idx: p for p in profiles}
        self._axis_std = np.maximum(np.std(space.S, axis=0), 1e-8)

    def _build_label_map(self) -> dict[str, list[int]]:
        """Map labels to axis indices."""
        mapping: dict[str, list[int]] = {}
        for p in self.profiles:
            if p.label:
                mapping.setdefault(p.label, []).append(p.axis_idx)
        return mapping

    def _find_axis_for_label(self, label: str) -> int | None:
        """Find the best axis index for a semantic label."""
        axes = self._label_to_axes.get(label, [])
        if not axes:
            return None
        best = max(
            axes,
            key=lambda a: len(
                self._profile_by_axis[a].antonym_pairs if a in self._profile_by_axis else []
            ),
        )
        return best

    def _find_oracle_axis(self, word: str, expected: str) -> int | None:
        """Find the best axis by looking at the actual word pair's score difference,
        normalized by axis standard deviation."""
        s1 = self.space.score(word)
        s2 = self.space.score(expected)
        if s1 is None or s2 is None:
            return None
        diff = np.abs(s1 - s2)
        # Normalize by axis std to avoid picking axes with naturally large scales
        normalized_diff = diff / self._axis_std
        return int(np.argmax(normalized_diff))

    def _find_oracle_top_axes(self, word: str, expected: str, n: int = 5) -> list[int]:
        """Find the top-n axes with largest normalized score difference."""
        s1 = self.space.score(word)
        s2 = self.space.score(expected)
        if s1 is None or s2 is None:
            return []
        diff = np.abs(s1 - s2)
        normalized_diff = diff / self._axis_std
        return list(np.argsort(normalized_diff)[-n:][::-1])

    def _rank_of(self, neighbors, expected: str) -> int | None:
        """1-based rank of `expected` in a neighbor list, or None."""
        for i, n in enumerate(neighbors):
            if n.word == expected:
                return i + 1
        return None

    @staticmethod
    def _hits_from(rank: int | None, hits: dict[int, int]) -> None:
        if rank is not None:
            for k in [1, 5, 10]:
                if rank <= k:
                    hits[k] += 1

    def _label_group(self, label: str) -> list[int]:
        """All axis indices carrying a semantic label."""
        return self._label_to_axes.get(label, [])

    def run_blind(self, top_n: int = 10) -> dict:
        """
        Blind evaluation (primary metric): never consults the expected word.

        Two variants:
          - label: flip all axes carrying the test case's declared semantic
            label (e.g. "gender"). The label is part of the task spec.
          - auto: flip only the single axis where the SOURCE word has the
            highest |z-score|. No labels, no oracle.

        Returns:
            Dict with per-case results and aggregate Hits@k per variant.
        """
        label_results = []
        auto_results = []
        label_hits = {1: 0, 5: 0, 10: 0}
        auto_hits = {1: 0, 5: 0, 10: 0}
        label_total = 0
        auto_total = 0

        for tc in CANONICAL_TEST_CASES:
            if self.space.score(tc.word) is None:
                label_results.append(
                    {
                        "word": tc.word,
                        "expected": tc.expected,
                        "axis_label": tc.axis_label,
                        "status": "word_not_found",
                        "neighbors": [],
                    }
                )
                continue

            # Variant 1: declared-label group inversion
            axes = self._label_group(tc.axis_label)
            if axes:
                neighbors = self.operator.multi_axis_invert(tc.word, axes, top_n=top_n)
                rank = self._rank_of(neighbors, tc.expected)
                self._hits_from(rank, label_hits)
                label_total += 1
                label_results.append(
                    {
                        "word": tc.word,
                        "expected": tc.expected,
                        "axis_label": tc.axis_label,
                        "axes": [int(a) for a in axes],
                        "rank": rank,
                        "neighbors": [
                            {"word": n.word, "sim": round(n.similarity, 4)} for n in neighbors[:10]
                        ],
                    }
                )
            else:
                label_results.append(
                    {
                        "word": tc.word,
                        "expected": tc.expected,
                        "axis_label": tc.axis_label,
                        "status": "no_labeled_axis",
                        "neighbors": [],
                    }
                )

            # Variant 2: fully automatic source-top-1 axis inversion
            result = self.operator.invert_by_source_topk(tc.word, k=1, top_n=top_n)
            if result.neighbors:
                rank = self._rank_of(result.neighbors, tc.expected)
                self._hits_from(rank, auto_hits)
                auto_total += 1
                auto_results.append(
                    {
                        "word": tc.word,
                        "expected": tc.expected,
                        "axis": int(result.axes_flipped[0]) if result.axes_flipped else None,
                        "rank": rank,
                        "neighbors": [
                            {"word": n.word, "sim": round(n.similarity, 4)}
                            for n in result.neighbors[:10]
                        ],
                    }
                )

        return {
            "label": {
                "total_cases": len(CANONICAL_TEST_CASES),
                "evaluated": label_total,
                "metrics": {
                    f"hits_at_{k}": label_hits[k] / label_total if label_total > 0 else 0.0
                    for k in [1, 5, 10]
                },
                "cases": label_results,
            },
            "auto": {
                "total_cases": len(CANONICAL_TEST_CASES),
                "evaluated": auto_total,
                "metrics": {
                    f"hits_at_{k}": auto_hits[k] / auto_total if auto_total > 0 else 0.0
                    for k in [1, 5, 10]
                },
                "cases": auto_results,
            },
        }

    def run_oracle(self, top_n: int = 10) -> dict:
        """
        Oracle evaluation (upper bound): use the known (word, expected) pair
        to select the best axis among the top-5 candidates.

        The target word is consulted, so these numbers are NOT deployable
        performance; they bound what axis inversion can achieve.
        """
        results = []
        hits = {1: 0, 5: 0, 10: 0}
        total = 0

        for tc in CANONICAL_TEST_CASES:
            top_axes = self._find_oracle_top_axes(tc.word, tc.expected, n=5)
            labeled_axis = self._find_axis_for_label(tc.axis_label)

            if not top_axes:
                results.append(
                    {
                        "word": tc.word,
                        "expected": tc.expected,
                        "axis_label": tc.axis_label,
                        "status": "word_not_found",
                        "neighbors": [],
                    }
                )
                continue

            # Try each candidate axis and pick the best
            best_rank = None
            best_axis = top_axes[0]
            best_neighbors = []

            for axis in top_axes:
                neighbors = self.operator.axis_invert(tc.word, axis, top_n=top_n)
                rank = self._rank_of(neighbors, tc.expected)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_axis = axis
                    best_neighbors = neighbors

            # Also try multi-axis inversion with top-3 axes
            multi_neighbors = self.operator.multi_axis_invert(tc.word, top_axes[:3], top_n=top_n)
            multi_rank = self._rank_of(multi_neighbors, tc.expected)
            if multi_rank is not None and (best_rank is None or multi_rank < best_rank):
                best_rank = multi_rank

            # If no axis found the expected word, just use the top oracle axis
            if not best_neighbors:
                best_neighbors = self.operator.axis_invert(tc.word, top_axes[0], top_n=top_n)

            total += 1
            self._hits_from(best_rank, hits)

            results.append(
                {
                    "word": tc.word,
                    "expected": tc.expected,
                    "axis_label": tc.axis_label,
                    "oracle_axis": int(best_axis),
                    "labeled_axis": int(labeled_axis) if labeled_axis is not None else None,
                    "candidate_axes": [int(a) for a in top_axes],
                    "best_rank": best_rank,
                    "multi_axis_rank": multi_rank,
                    "neighbors": [
                        {"word": n.word, "sim": round(n.similarity, 4)} for n in best_neighbors[:10]
                    ],
                    "multi_neighbors": [
                        {"word": n.word, "sim": round(n.similarity, 4)}
                        for n in multi_neighbors[:10]
                    ],
                }
            )

        metrics = {}
        for k in [1, 5, 10]:
            metrics[f"hits_at_{k}"] = hits[k] / total if total > 0 else 0.0

        return {
            "total_cases": len(CANONICAL_TEST_CASES),
            "evaluated": total,
            "metrics": metrics,
            "cases": results,
        }

    def run_all(self, top_n: int = 10) -> dict:
        """Run both blind (primary) and oracle (upper bound) evaluations."""
        return {
            "blind": self.run_blind(top_n),
            "oracle": self.run_oracle(top_n),
        }

    def compare_with_analogy(self, top_n: int = 10) -> dict:
        """
        Compare blind ICA inversion results with traditional vector analogy.

        The ICA side inverts the axes of the declared semantic label (blind);
        the analogy side uses a:b :: c:? with an unrelated word pair, so this
        comparison illustrates the difference in task setup rather than a
        like-for-like accuracy contest.
        """
        comparisons = []

        for tc in CANONICAL_TEST_CASES:
            if not tc.analogy_a or not tc.analogy_b:
                continue

            # Blind ICA inversion: flip the declared label's axes
            axes = self._label_group(tc.axis_label)
            ica_neighbors = []
            ica_rank = None
            if axes:
                ica_neighbors = self.operator.multi_axis_invert(tc.word, axes, top_n=top_n)
                ica_rank = self._rank_of(ica_neighbors, tc.expected)

            # Traditional analogy: a->b as word->?
            analogy_neighbors = self.operator.analogy_traditional(
                tc.analogy_a, tc.analogy_b, tc.word, top_n=top_n
            )
            analogy_rank = self._rank_of(analogy_neighbors, tc.expected)

            comparisons.append(
                {
                    "word": tc.word,
                    "expected": tc.expected,
                    "analogy": f"{tc.analogy_a}:{tc.analogy_b}::{tc.word}:?",
                    "ica_rank": ica_rank,
                    "analogy_rank": analogy_rank,
                    "ica_top5": [n.word for n in ica_neighbors[:5]],
                    "analogy_top5": [n.word for n in analogy_neighbors[:5]],
                }
            )

        return {"comparisons": comparisons}

    def failure_analysis(self, oracle_results: dict) -> dict:
        """Analyze failures from run_oracle() results (informative for axis
        coverage even though oracle numbers are an upper bound)."""
        failures = []
        for case in oracle_results["cases"]:
            if case.get("best_rank") is None and case.get("status") != "word_not_found":
                comparison = self.operator.compare_words(case["word"], case["expected"])
                failures.append(
                    {
                        "word": case["word"],
                        "expected": case["expected"],
                        "candidate_axes": case.get("candidate_axes", []),
                        "actual_top3": [n["word"] for n in case.get("neighbors", [])[:3]],
                        "multi_top3": [n["word"] for n in case.get("multi_neighbors", [])[:3]],
                        "word_comparison": comparison.get("top_diffs", [])[:5],
                    }
                )
            elif case.get("status") == "word_not_found":
                failures.append(
                    {
                        "word": case["word"],
                        "expected": case["expected"],
                        "reason": "word not in ICA space",
                    }
                )

        return {"failures": failures, "total_failures": len(failures)}


def save_qualitative_results(results: dict, path: str) -> None:
    """Save qualitative evaluation results to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved qualitative results to {path}")
