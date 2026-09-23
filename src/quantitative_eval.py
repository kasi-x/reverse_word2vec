"""
Quantitative evaluation of ICA axis-wise semantic inversion.

- Antonym retrieval evaluation with 5-fold cross-validation
- Google Analogy Dataset evaluation comparing ICA inversion vs traditional analogy
"""

import json
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

from src.eval_utils import make_folds
from src.ica_transformer import ICASpace, ICATransformer
from src.semantic_operations import SemanticOperator


class AntonymRetrievalEval:
    """Evaluate antonym retrieval via ICA axis inversion with cross-validation."""

    def __init__(self, operator: SemanticOperator, space: ICASpace):
        self.operator = operator
        self.space = space
        self._axis_std = np.maximum(np.std(space.S, axis=0), 1e-8)

    def select_best_axis(self, w1: str, w2: str) -> int | None:
        """
        Oracle axis selection: the axis with the largest std-normalised
        absolute score difference |S[w1] - S[w2]| / σ.

        Requires the target word, so this is a non-deployable upper-bound
        baseline (same normalisation as QualitativeEvaluator and eval_leakfree).
        """
        s1 = self.space.score(w1)
        s2 = self.space.score(w2)
        if s1 is None or s2 is None:
            return None
        return int(np.argmax(np.abs(s1 - s2) / self._axis_std))

    def evaluate_pairs(
        self,
        pairs: list[tuple[str, str]],
        top_n: int = 10,
    ) -> dict:
        """
        Oracle evaluation of antonym retrieval on a set of pairs.

        For each pair (w1, w2):
          1. Select the axis from the pair itself (oracle: knows the target)
          2. Invert w1 on that axis and check if w2 appears in top-k

        Args:
            pairs: Pairs to evaluate.
            top_n: Max rank to check.

        Returns:
            Dict with Hits@1/5/10 and per-pair details.
        """
        hits = {1: 0, 5: 0, 10: 0}
        evaluated = 0
        details = []

        for w1, w2 in pairs:
            if self.space.score(w1) is None or self.space.score(w2) is None:
                continue

            axis = self.select_best_axis(w1, w2)

            if axis is None:
                continue

            # Try both directions and take best rank
            best_rank = None
            for source, target in [(w1, w2), (w2, w1)]:
                neighbors = self.operator.axis_invert(source, axis, top_n=top_n)
                for i, n in enumerate(neighbors):
                    if n.word == target:
                        rank = i + 1
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                        break

            evaluated += 1
            for k in [1, 5, 10]:
                if best_rank is not None and best_rank <= k:
                    hits[k] += 1

            details.append(
                {
                    "w1": w1,
                    "w2": w2,
                    "axis": axis,
                    "best_rank": best_rank,
                }
            )

        metrics = {}
        for k in [1, 5, 10]:
            metrics[f"hits_at_{k}"] = hits[k] / evaluated if evaluated > 0 else 0.0

        return {
            "evaluated": evaluated,
            "metrics": metrics,
            "details": details,
        }

    def cross_validate(
        self,
        pairs: list[tuple[str, str]],
        n_folds: int = 5,
        top_n: int = 10,
    ) -> dict:
        """
        5-fold cross-validation of antonym retrieval (oracle axis selection).

        Splits are shuffled with seed 42; each test pair selects its own
        axis (knows the target), so this measures axis geometry, not a
        deployable method.
        """
        # Filter to pairs in vocabulary
        valid_pairs = [
            (w1, w2)
            for w1, w2 in pairs
            if self.space.score(w1) is not None and self.space.score(w2) is not None
        ]

        folds = make_folds(len(valid_pairs), n_folds, seed=42)

        fold_results = []
        for fold_idx in range(n_folds):
            test_pairs = [valid_pairs[i] for i in folds[fold_idx]]

            result = self.evaluate_pairs(test_pairs, top_n=top_n)
            fold_results.append(result)
            print(
                f"  Fold {fold_idx + 1}/{n_folds}: "
                f"Hits@1={result['metrics']['hits_at_1']:.3f}, "
                f"Hits@5={result['metrics']['hits_at_5']:.3f}, "
                f"Hits@10={result['metrics']['hits_at_10']:.3f} "
                f"({result['evaluated']} pairs)"
            )

        # Aggregate
        agg_metrics = {}
        for k_str in ["hits_at_1", "hits_at_5", "hits_at_10"]:
            values = [r["metrics"][k_str] for r in fold_results]
            agg_metrics[k_str] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "per_fold": values,
            }

        return {
            "n_folds": n_folds,
            "total_valid_pairs": len(valid_pairs),
            "metrics": agg_metrics,
        }


class AnalogyEval:
    """Evaluate on the Google Analogy Dataset (questions-words.txt)."""

    def __init__(self, operator: SemanticOperator, model: KeyedVectors, space: ICASpace):
        self.operator = operator
        self.model = model
        self.space = space
        self.transformer = ICATransformer()

    def load_analogy_dataset(self, path: str) -> dict[str, list[tuple[str, str, str, str]]]:
        """
        Load Google Analogy Dataset.

        Returns:
            Dict mapping category name to list of (a, b, c, d) tuples.
        """
        categories: dict[str, list[tuple[str, str, str, str]]] = {}
        current_cat = ""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(":"):
                    current_cat = line[2:]
                    categories[current_cat] = []
                elif line:
                    parts = line.lower().split()
                    if len(parts) == 4:
                        categories[current_cat].append(tuple(parts))
        return categories

    def evaluate(self, dataset_path: str, max_per_category: int = 200) -> dict:
        """
        Evaluate ICA inversion vs traditional analogy on the Google dataset.

        For each analogy a:b::c:d:
        - Traditional: find nearest to b - a + c
        - ICA: find axis where a and b differ most, invert c on that axis
        """
        categories = self.load_analogy_dataset(dataset_path)
        results = {}

        for cat, quads in categories.items():
            ica_correct = 0
            trad_correct = 0
            evaluated = 0

            for a, b, c, d in quads[:max_per_category]:
                # Skip if any word not in vocab
                if any(w not in self.model for w in [a, b, c, d]):
                    continue

                evaluated += 1

                # Traditional analogy
                trad_neighbors = self.operator.analogy_traditional(a, b, c, top_n=5)
                if trad_neighbors and trad_neighbors[0].word == d:
                    trad_correct += 1

                # ICA-based: find axis differentiating a and b, invert c on it
                s_a = self.transformer.transform_word(a, self.model, self.space)
                s_b = self.transformer.transform_word(b, self.model, self.space)
                if s_a is not None and s_b is not None:
                    diff = np.abs(s_a - s_b)
                    best_axis = int(np.argmax(diff))
                    ica_neighbors = self.operator.axis_invert(c, best_axis, top_n=5, exclude={a, b})
                    if ica_neighbors and ica_neighbors[0].word == d:
                        ica_correct += 1

            if evaluated > 0:
                results[cat] = {
                    "evaluated": evaluated,
                    "ica_accuracy": ica_correct / evaluated,
                    "traditional_accuracy": trad_correct / evaluated,
                    "ica_correct": ica_correct,
                    "traditional_correct": trad_correct,
                }
                print(
                    f"  {cat}: ICA={ica_correct}/{evaluated} "
                    f"({ica_correct / evaluated:.1%}), "
                    f"Trad={trad_correct}/{evaluated} "
                    f"({trad_correct / evaluated:.1%})"
                )

        # Overall
        total_eval = sum(r["evaluated"] for r in results.values())
        total_ica = sum(r["ica_correct"] for r in results.values())
        total_trad = sum(r["traditional_correct"] for r in results.values())

        return {
            "per_category": results,
            "overall": {
                "evaluated": total_eval,
                "ica_accuracy": total_ica / total_eval if total_eval > 0 else 0,
                "traditional_accuracy": total_trad / total_eval if total_eval > 0 else 0,
            },
        }


def save_quantitative_results(results: dict, path: str) -> None:
    """Save quantitative evaluation results to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved quantitative results to {path}")
