"""
Main experiment runner for Word2Vec antonym space analysis.
"""
import os
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm

from antonym_loader import (
    extract_antonym_pairs,
    extract_antonym_pairs_by_pos,
    group_antonyms_by_word,
    save_antonym_pairs,
    load_antonym_pairs,
)
from word2vec_loader import Word2VecLoader
from antonym_space import AntonymSpace, create_antonym_space_from_model


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    model_type: str = 'glove-100'  # 'google-news', 'glove-50/100/200/300'
    method: str = 'greedy'  # 'greedy' or 'svd'
    max_axes: Optional[int] = 100
    min_orthogonality: float = 0.1  # For greedy method
    explained_variance_ratio: float = 0.95  # For SVD method
    output_dir: str = 'results'


@dataclass
class ExperimentResults:
    """Container for experiment results."""
    config: dict
    num_antonym_pairs: int
    num_valid_pairs: int
    num_axes_created: int
    coverage_stats: dict
    near_zero_words: List[Tuple[str, float]]
    discovered_antonyms: Dict[str, List[Tuple[str, float]]]
    timestamp: str


class Experiment:
    """Main experiment class."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.loader = Word2VecLoader()
        self.model = None
        self.antonym_pairs = []
        self.known_antonyms = {}
        self.space: Optional[AntonymSpace] = None

    def setup(self):
        """Load data and model."""
        print("=" * 60)
        print("EXPERIMENT SETUP")
        print("=" * 60)

        # Load antonym pairs from WordNet
        print("\n[1/2] Loading antonym pairs from WordNet...")
        self.antonym_pairs = extract_antonym_pairs()
        self.known_antonyms = group_antonyms_by_word()
        print(f"  Loaded {len(self.antonym_pairs)} antonym pairs")
        print(f"  Covering {len(self.known_antonyms)} unique words")

        # Load Word2Vec model
        print(f"\n[2/2] Loading Word2Vec model: {self.config.model_type}")
        if self.config.model_type == 'google-news':
            self.model = self.loader.load_google_news()
        elif self.config.model_type.startswith('glove-'):
            dim = int(self.config.model_type.split('-')[1])
            self.model = self.loader.load_glove(dim)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        print(f"  Model loaded: {len(self.model)} words, "
              f"dim={self.model.vector_size}")

    def run(self) -> ExperimentResults:
        """Run the main experiment."""
        print("\n" + "=" * 60)
        print("RUNNING EXPERIMENT")
        print("=" * 60)

        # Filter antonym pairs to those in vocabulary
        print("\n[1/5] Filtering antonym pairs to vocabulary...")
        valid_pairs = [
            (w1, w2) for w1, w2 in self.antonym_pairs
            if self.loader.has_word(w1) and self.loader.has_word(w2)
        ]
        print(f"  Valid pairs: {len(valid_pairs)} / {len(self.antonym_pairs)}")

        # Build antonym space
        print(f"\n[2/5] Building antonym space (method={self.config.method})...")
        if self.config.method == 'greedy':
            self.space = create_antonym_space_from_model(
                self.model,
                valid_pairs,
                method='greedy',
                max_axes=self.config.max_axes,
                min_orthogonality=self.config.min_orthogonality,
            )
        else:
            self.space = create_antonym_space_from_model(
                self.model,
                valid_pairs,
                method='svd',
                max_axes=self.config.max_axes,
                explained_variance_ratio=self.config.explained_variance_ratio,
            )
        print(f"  Created {len(self.space.axes)} axes")

        # Print axis info
        print("\n  Sample axes:")
        for i, axis in enumerate(self.space.axes[:10]):
            print(f"    Axis {i}: {axis.positive_word} <-> {axis.negative_word} "
                  f"(sep={axis.separation:.3f})")

        # Analyze coverage
        print("\n[3/5] Analyzing space coverage...")
        coverage_stats = self.space.analyze_axis_coverage()
        print(f"  Explained variance ratio: {coverage_stats['explained_ratio']:.4f}")

        # Find near-zero words
        print("\n[4/5] Finding words near zero (semantic neutrals)...")
        near_zero = self.space.find_near_zero_words(top_n=100)
        print(f"  Found {len(near_zero)} words near zero")
        print("\n  Top 20 neutral words:")
        for word, dist, _ in near_zero[:20]:
            print(f"    {word}: {dist:.4f}")

        # Discover potential antonyms for words without known antonyms
        print("\n[5/5] Discovering potential antonyms...")
        discovered = self._discover_antonyms(n_words=50, top_k=5)

        results = ExperimentResults(
            config=asdict(self.config),
            num_antonym_pairs=len(self.antonym_pairs),
            num_valid_pairs=len(valid_pairs),
            num_axes_created=len(self.space.axes),
            coverage_stats=coverage_stats,
            near_zero_words=[(w, d) for w, d, _ in near_zero],
            discovered_antonyms=discovered,
            timestamp=datetime.now().isoformat(),
        )

        return results

    def _discover_antonyms(
        self,
        n_words: int = 50,
        top_k: int = 5
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Try to discover antonyms for words that don't have known antonyms.
        """
        # Find words without known antonyms
        words_without_antonyms = [
            word for word in list(self.model.key_to_index.keys())[:10000]
            if word.lower() not in self.known_antonyms
            and word.isalpha()
            and len(word) > 2
        ]

        print(f"  Analyzing {n_words} words without known antonyms...")

        discovered = {}
        sample_words = words_without_antonyms[:n_words]

        for word in tqdm(sample_words, desc="  Discovering"):
            potential = self.space.find_potential_antonyms(
                word,
                candidates=list(self.model.key_to_index.keys())[:50000],
                top_n=top_k
            )
            if potential:
                discovered[word] = potential

        # Print some discoveries
        print("\n  Sample discovered potential antonyms:")
        for word, antonyms in list(discovered.items())[:10]:
            ant_str = ", ".join([f"{a}({s:.2f})" for a, s in antonyms[:3]])
            print(f"    {word}: {ant_str}")

        return discovered

    def save_results(self, results: ExperimentResults, filename: str = None):
        """Save results to file."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.config.output_dir, filename)

        # Convert numpy types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_to_native(item) for item in obj]
            return obj

        # Convert to JSON-serializable format
        data = convert_to_native({
            'config': results.config,
            'num_antonym_pairs': results.num_antonym_pairs,
            'num_valid_pairs': results.num_valid_pairs,
            'num_axes_created': results.num_axes_created,
            'coverage_stats': results.coverage_stats,
            'near_zero_words': results.near_zero_words,
            'discovered_antonyms': results.discovered_antonyms,
            'timestamp': results.timestamp,
        })

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {filepath}")

    def save_space(self, filename: str = 'antonym_space.json'):
        """Save the antonym space for later analysis."""
        if self.space is None:
            raise RuntimeError("No space to save. Run experiment first.")

        os.makedirs(self.config.output_dir, exist_ok=True)
        filepath = os.path.join(self.config.output_dir, filename)

        data = {
            'basis_matrix': self.space.basis_matrix.tolist(),
            'axes': [(a.positive_word, a.negative_word, float(a.separation))
                     for a in self.space.axes],
            'config': asdict(self.config),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f)

        print(f"Space saved to: {filepath}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Word2Vec Antonym Space Experiment'
    )
    parser.add_argument(
        '--model', type=str, default='glove-100',
        choices=['google-news', 'glove-50', 'glove-100', 'glove-200', 'glove-300'],
        help='Word embedding model to use'
    )
    parser.add_argument(
        '--method', type=str, default='greedy',
        choices=['greedy', 'svd'],
        help='Method for building antonym axes'
    )
    parser.add_argument(
        '--max-axes', type=int, default=100,
        help='Maximum number of antonym axes to create'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        model_type=args.model,
        method=args.method,
        max_axes=args.max_axes,
        output_dir=args.output_dir,
    )

    experiment = Experiment(config)
    experiment.setup()
    results = experiment.run()
    experiment.save_results(results)
    experiment.save_space()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
