"""
Experiment 4: Deep Analysis of Antonym Dimensions

Goal: Understand the structure of antonym space
- How many dimensions are needed?
- What do the principal directions represent?
- Are there universal semantic categories?

Approach:
1. SVD analysis of antonym directions
2. Interpretation of principal components
3. Clustering to find semantic categories
"""
import sys
sys.path.insert(0, '..')

import numpy as np
from numpy.linalg import norm, svd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from collections import defaultdict
from typing import List, Tuple, Dict
import time

from antonym_loader import extract_antonym_pairs
from word2vec_loader import Word2VecLoader


class DimensionAnalyzer:
    """
    Analyze the dimensional structure of antonym space.
    """

    def __init__(self, model, antonym_pairs: List[Tuple[str, str]]):
        self.model = model
        self.dim = model.vector_size

        print("Building dimension analyzer...")

        # Build antonym direction matrix
        self.directions = []
        self.pair_labels = []

        for w1, w2 in antonym_pairs:
            if w1 in model and w2 in model:
                d = model[w1] - model[w2]
                d_norm = norm(d)
                if d_norm > 1e-6:
                    self.directions.append(d / d_norm)
                    self.pair_labels.append((w1, w2))

        self.D = np.array(self.directions, dtype=np.float32)
        print(f"  {len(self.directions)} antonym directions")

        # Perform SVD
        print("  Computing SVD...")
        self.U, self.S, self.Vh = svd(self.D, full_matrices=False)
        self.principal_directions = self.Vh  # (k, dim)

        # Compute variance explained
        total_var = np.sum(self.S ** 2)
        self.variance_explained = self.S ** 2 / total_var
        self.cumulative_variance = np.cumsum(self.variance_explained)

        print(f"  Top 5 singular values: {self.S[:5]}")

    def get_dimension_for_variance(self, target_variance: float = 0.9) -> int:
        """Get number of dimensions needed to explain target variance."""
        return np.searchsorted(self.cumulative_variance, target_variance) + 1

    def get_representative_pairs(self, component_idx: int, top_n: int = 5) -> List[Tuple[Tuple[str, str], float]]:
        """
        Get the antonym pairs most aligned with a principal component.
        """
        if component_idx >= len(self.Vh):
            return []

        direction = self.Vh[component_idx]

        # Compute alignment of each antonym pair with this direction
        alignments = self.D @ direction  # (n_pairs,)

        # Get top pairs (both positive and negative alignment)
        pos_indices = np.argsort(alignments)[::-1][:top_n]
        neg_indices = np.argsort(alignments)[:top_n]

        results = []
        for idx in pos_indices:
            results.append((self.pair_labels[idx], float(alignments[idx])))
        for idx in neg_indices:
            if idx not in pos_indices:
                results.append((self.pair_labels[idx], float(alignments[idx])))

        return results

    def interpret_component(self, component_idx: int, model) -> Dict:
        """
        Try to interpret a principal component.
        """
        direction = self.Vh[component_idx]

        # Find words most aligned with this direction
        words = list(model.key_to_index.keys())[:50000]
        valid_words = [w for w in words if w.isalpha() and w == w.lower() and 3 <= len(w) <= 12]

        projections = []
        for word in valid_words:
            v = model[word]
            proj = np.dot(v, direction)
            projections.append((word, proj))

        # Sort by projection
        projections.sort(key=lambda x: x[1])

        # Get representative pairs
        pairs = self.get_representative_pairs(component_idx, top_n=5)

        return {
            'component': component_idx,
            'variance_explained': float(self.variance_explained[component_idx]),
            'cumulative_variance': float(self.cumulative_variance[component_idx]),
            'negative_pole_words': projections[:10],
            'positive_pole_words': projections[-10:][::-1],
            'representative_pairs': pairs
        }

    def cluster_directions(self, n_clusters: int = 15) -> Dict:
        """
        Cluster antonym directions to find semantic categories.
        """
        print(f"  Clustering into {n_clusters} categories...")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.D)

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(self.pair_labels[i])

        # Compute cluster centroids and characteristics
        cluster_info = {}
        for label, pairs in clusters.items():
            centroid = kmeans.cluster_centers_[label]
            centroid = centroid / (norm(centroid) + 1e-10)

            cluster_info[label] = {
                'n_pairs': len(pairs),
                'example_pairs': pairs[:5],
                'centroid': centroid
            }

        return cluster_info


def run_dimension_experiment():
    """Run the dimension analysis experiment."""
    print("=" * 70)
    print("EXPERIMENT 4: Dimension Analysis")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    antonym_pairs = extract_antonym_pairs()
    loader = Word2VecLoader()
    model = loader.load_glove(100)

    # Build analyzer
    print("\n[2/3] Building analyzer...")
    analyzer = DimensionAnalyzer(model, antonym_pairs)

    # Dimension analysis
    print("\n[3/3] Analyzing dimensions...")

    print("\n" + "=" * 70)
    print("VARIANCE ANALYSIS")
    print("=" * 70)

    print("\n  Dimensions needed to explain variance:")
    for target in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        n_dim = analyzer.get_dimension_for_variance(target)
        print(f"    {target*100:.0f}%: {n_dim} dimensions")

    print(f"\n  Total antonym directions: {len(analyzer.directions)}")
    print(f"  Original embedding dimension: {analyzer.dim}")

    # Top principal components
    print("\n" + "=" * 70)
    print("PRINCIPAL COMPONENTS INTERPRETATION")
    print("=" * 70)

    for i in range(10):
        interp = analyzer.interpret_component(i, model)

        print(f"\nComponent {i + 1} (explains {interp['variance_explained']*100:.2f}% variance):")

        # Show representative pairs
        pairs = interp['representative_pairs']
        if pairs:
            print(f"  Representative antonym pairs:")
            for (w1, w2), align in pairs[:3]:
                print(f"    {w1}/{w2} (alignment: {align:+.2f})")

        # Show pole words
        pos_words = [w for w, p in interp['positive_pole_words'][:5]]
        neg_words = [w for w, p in interp['negative_pole_words'][:5]]
        print(f"  Negative pole: {', '.join(neg_words)}")
        print(f"  Positive pole: {', '.join(pos_words)}")

    # Clustering
    print("\n" + "=" * 70)
    print("SEMANTIC CATEGORIES (Clustering)")
    print("=" * 70)

    cluster_info = analyzer.cluster_directions(n_clusters=12)

    sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['n_pairs'], reverse=True)

    for label, info in sorted_clusters:
        print(f"\nCategory {label + 1} ({info['n_pairs']} pairs):")
        for w1, w2 in info['example_pairs']:
            print(f"    {w1} ↔ {w2}")

    # Variance decay plot (text-based)
    print("\n" + "=" * 70)
    print("VARIANCE DECAY (Scree Plot)")
    print("=" * 70)

    print("\n  Component | Variance | Cumulative | Bar")
    print("  " + "-" * 55)
    for i in range(min(20, len(analyzer.variance_explained))):
        var = analyzer.variance_explained[i]
        cum = analyzer.cumulative_variance[i]
        bar = "█" * int(var * 100)
        print(f"  {i+1:9d} | {var*100:7.2f}% | {cum*100:9.2f}% | {bar}")

    return analyzer


if __name__ == "__main__":
    run_dimension_experiment()
