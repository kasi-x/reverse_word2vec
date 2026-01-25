"""
Automatic Antonym Axis Discovery

Goal: Automatically discover all semantic axes for a word without
relying on WordNet or any predefined antonym list.

Approach:
1. Find candidates with antonym-like relationships (using global antonym directions)
2. Cluster the difference vectors to find distinct semantic axes
3. Interpret each cluster as a different meaning dimension
"""
import sys
sys.path.insert(0, '..')

import numpy as np
from numpy.linalg import norm, svd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass

from antonym_loader import extract_antonym_pairs, group_antonyms_by_word
from word2vec_loader import Word2VecLoader


@dataclass
class DiscoveredAxis:
    """Represents an automatically discovered semantic axis."""
    axis_id: int
    direction: np.ndarray
    representative_antonym: str
    representative_score: float
    cluster_words: List[Tuple[str, float]]  # (word, score)
    interpretation: Optional[str] = None


class AutoAxisDiscoverer:
    """
    Automatically discovers semantic axes for any word.
    """

    def __init__(self, model, antonym_pairs: List[Tuple[str, str]]):
        self.model = model
        self.dim = model.vector_size
        self.known_antonyms = group_antonyms_by_word()

        print("Building auto axis discoverer...")

        # Build global antonym direction matrix
        self.directions = []
        self.direction_pairs = []
        for w1, w2 in antonym_pairs:
            if w1 in model and w2 in model:
                d = model[w1] - model[w2]
                d_norm = norm(d)
                if d_norm > 1e-6:
                    self.directions.append(d / d_norm)
                    self.direction_pairs.append((w1, w2))

        self.D = np.array(self.directions, dtype=np.float32)
        print(f"  {len(self.directions)} global antonym directions")

        # Vocabulary
        self.vocab = [
            w for w in model.key_to_index
            if w.isalpha() and w == w.lower() and 3 <= len(w) <= 15
        ][:50000]
        print(f"  {len(self.vocab)} candidate words")

    def find_antonym_candidates(self, word: str, top_n: int = 200,
                                  min_score: float = 0.4) -> List[Dict]:
        """
        Find words that have antonym-like relationships with the target word.
        """
        if word not in self.model:
            return []

        v_word = self.model[word]
        candidates = []

        for cand in self.vocab:
            if cand == word:
                continue

            v_cand = self.model[cand]
            diff = v_word - v_cand
            diff_norm = norm(diff)
            if diff_norm < 1e-6:
                continue

            diff_normalized = diff / diff_norm

            # Antonym score: how well does this direction align with known antonym directions?
            antonym_score = np.max(np.abs(self.D @ diff_normalized))

            # Semantic relatedness (cosine)
            cosine = np.dot(v_word, v_cand) / (norm(v_word) * norm(v_cand))

            if antonym_score >= min_score:
                candidates.append({
                    'word': cand,
                    'antonym_score': float(antonym_score),
                    'cosine': float(cosine),
                    'diff_vector': diff_normalized,
                    'distance': float(diff_norm)
                })

        # Sort by antonym score
        candidates.sort(key=lambda x: x['antonym_score'], reverse=True)
        return candidates[:top_n]

    def cluster_axes(self, word: str, candidates: List[Dict],
                      n_clusters: int = None,
                      method: str = 'auto') -> List[DiscoveredAxis]:
        """
        Cluster candidate difference vectors to find distinct axes.
        """
        if len(candidates) < 3:
            return []

        # Stack difference vectors
        diff_vectors = np.array([c['diff_vector'] for c in candidates])

        # Determine optimal number of clusters
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(diff_vectors, max_k=10)

        print(f"  Clustering into {n_clusters} axes...")

        # Cluster
        if method == 'kmeans' or method == 'auto':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(diff_vectors)
            centers = clusterer.cluster_centers_
        elif method == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(diff_vectors)
            # Compute centers manually
            centers = []
            for i in range(n_clusters):
                mask = labels == i
                if np.sum(mask) > 0:
                    center = np.mean(diff_vectors[mask], axis=0)
                    center = center / (norm(center) + 1e-10)
                    centers.append(center)
                else:
                    centers.append(np.zeros(self.dim))
            centers = np.array(centers)

        # Build discovered axes
        axes = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_candidates = [c for c, m in zip(candidates, mask) if m]

            if not cluster_candidates:
                continue

            # Find representative (highest antonym score in cluster)
            cluster_candidates.sort(key=lambda x: x['antonym_score'], reverse=True)
            representative = cluster_candidates[0]

            # Axis direction (cluster center, normalized)
            direction = centers[cluster_id]
            direction = direction / (norm(direction) + 1e-10)

            axes.append(DiscoveredAxis(
                axis_id=cluster_id,
                direction=direction,
                representative_antonym=representative['word'],
                representative_score=representative['antonym_score'],
                cluster_words=[(c['word'], c['antonym_score']) for c in cluster_candidates[:20]]
            ))

        # Sort by representative score
        axes.sort(key=lambda x: x.representative_score, reverse=True)

        return axes

    def _find_optimal_clusters(self, vectors: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using silhouette score."""
        if len(vectors) < 4:
            return min(2, len(vectors))

        best_k = 2
        best_score = -1

        for k in range(2, min(max_k + 1, len(vectors))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(vectors)
                score = silhouette_score(vectors, labels)

                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                pass

        return best_k

    def interpret_axis(self, word: str, axis: DiscoveredAxis,
                        top_n: int = 10) -> Dict:
        """
        Interpret what a discovered axis represents.
        """
        v_word = self.model[word]
        direction = axis.direction

        # Find words at each pole
        projections = []
        for w in self.vocab:
            if w == word:
                continue
            v = self.model[w]
            proj = np.dot(v - v_word, direction)
            projections.append((w, proj))

        projections.sort(key=lambda x: x[1])

        negative_pole = projections[:top_n]  # Antonym side
        positive_pole = projections[-top_n:][::-1]  # Word side

        # Find neutral words
        neutral = sorted(projections, key=lambda x: abs(x[1]))[:top_n]

        return {
            'axis_id': axis.axis_id,
            'representative': axis.representative_antonym,
            f'{word}_pole': positive_pole,
            'antonym_pole': negative_pole,
            'neutral': neutral
        }

    def discover_all_axes(self, word: str,
                           min_candidates: int = 50,
                           min_score: float = 0.4,
                           n_clusters: int = None) -> List[DiscoveredAxis]:
        """
        Main method: Discover all semantic axes for a word.
        """
        print(f"\nDiscovering axes for '{word}'...")

        # Step 1: Find antonym candidates
        print("  Finding antonym candidates...")
        candidates = self.find_antonym_candidates(word, top_n=200, min_score=min_score)
        print(f"  Found {len(candidates)} candidates")

        if len(candidates) < 5:
            print("  Not enough candidates for clustering")
            return []

        # Step 2: Cluster to find axes
        axes = self.cluster_axes(word, candidates, n_clusters=n_clusters)
        print(f"  Discovered {len(axes)} axes")

        return axes

    def compare_with_known(self, word: str, axes: List[DiscoveredAxis]) -> Dict:
        """
        Compare discovered axes with known WordNet antonyms.
        """
        known = self.known_antonyms.get(word, set())

        comparison = {
            'word': word,
            'known_antonyms': list(known),
            'discovered_axes': [],
            'coverage': 0,
            'new_discoveries': []
        }

        for axis in axes:
            axis_words = set(w for w, _ in axis.cluster_words)

            # Check if this axis covers any known antonym
            covered_known = axis_words & known
            is_known = len(covered_known) > 0 or axis.representative_antonym in known

            comparison['discovered_axes'].append({
                'representative': axis.representative_antonym,
                'matches_known': is_known,
                'covered_antonyms': list(covered_known),
                'top_words': [w for w, _ in axis.cluster_words[:5]]
            })

            if not is_known:
                comparison['new_discoveries'].append(axis.representative_antonym)

        # Coverage: how many known antonyms are covered?
        all_discovered = set()
        for axis in axes:
            all_discovered.update(w for w, _ in axis.cluster_words)
            all_discovered.add(axis.representative_antonym)

        if known:
            comparison['coverage'] = len(all_discovered & known) / len(known)

        return comparison


def run_auto_discovery():
    """Run automatic axis discovery experiment."""
    print("=" * 70)
    print("AUTOMATIC ANTONYM AXIS DISCOVERY")
    print("=" * 70)

    # Load
    print("\n[1/3] Loading data...")
    antonym_pairs = extract_antonym_pairs()
    loader = Word2VecLoader()
    model = loader.load_glove(100)

    # Build discoverer
    print("\n[2/3] Building discoverer...")
    discoverer = AutoAxisDiscoverer(model, antonym_pairs)

    # Test words
    print("\n[3/3] Discovering axes...")

    test_words = ['light', 'right', 'good', 'hot', 'fast', 'hard', 'cold', 'high', 'long', 'open']

    all_results = {}

    for word in test_words:
        print("\n" + "=" * 70)
        print(f"WORD: {word}")
        print("=" * 70)

        known = discoverer.known_antonyms.get(word, set())
        print(f"Known antonyms (WordNet): {known}")

        # Discover axes
        axes = discoverer.discover_all_axes(word, min_score=0.35)

        if not axes:
            print("No axes discovered")
            continue

        all_results[word] = axes

        # Display each axis
        for i, axis in enumerate(axes[:8]):  # Top 8 axes
            print(f"\n{'─' * 60}")
            print(f"AXIS {i+1}: {word} ↔ {axis.representative_antonym}")
            print(f"Score: {axis.representative_score:.3f}")
            print(f"{'─' * 60}")

            # Show cluster words
            print(f"  Cluster members ({len(axis.cluster_words)} words):")
            for w, score in axis.cluster_words[:10]:
                marker = "★" if w in known else " "
                print(f"    {marker} {w:15s} ({score:.2f})")

            # Interpret
            interp = discoverer.interpret_axis(word, axis)
            print(f"\n  Antonym pole (toward '{axis.representative_antonym}'):")
            for w, proj in interp['antonym_pole'][:5]:
                print(f"      {w:15s} ({proj:+.2f})")
            print(f"\n  '{word}' pole:")
            for w, proj in interp[f'{word}_pole'][:5]:
                print(f"      {w:15s} ({proj:+.2f})")

        # Compare with known
        comparison = discoverer.compare_with_known(word, axes)
        print(f"\n{'─' * 60}")
        print("COMPARISON WITH WORDNET")
        print(f"{'─' * 60}")
        print(f"  Known coverage: {comparison['coverage']*100:.0f}%")
        print(f"  New discoveries: {comparison['new_discoveries'][:5]}")

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY: AUTO-DISCOVERED vs KNOWN AXES")
    print("=" * 70)

    print("\n  Word       | Known   | Discovered | New      | Coverage")
    print("  " + "-" * 60)

    for word in test_words:
        known = discoverer.known_antonyms.get(word, set())
        axes = all_results.get(word, [])
        comparison = discoverer.compare_with_known(word, axes) if axes else {'coverage': 0, 'new_discoveries': []}

        print(f"  {word:10s} | {len(known):7d} | {len(axes):10d} | "
              f"{len(comparison['new_discoveries']):8d} | {comparison['coverage']*100:6.0f}%")

    return discoverer, all_results


def detailed_analysis(word: str):
    """Detailed analysis for a single word."""
    print(f"\n{'=' * 70}")
    print(f"DETAILED AXIS ANALYSIS: {word}")
    print(f"{'=' * 70}")

    antonym_pairs = extract_antonym_pairs()
    loader = Word2VecLoader()
    model = loader.load_glove(100)

    discoverer = AutoAxisDiscoverer(model, antonym_pairs)

    # Discover with more candidates
    print("\nFinding candidates...")
    candidates = discoverer.find_antonym_candidates(word, top_n=300, min_score=0.3)
    print(f"Found {len(candidates)} antonym candidates")

    # Show top candidates before clustering
    print("\nTop 30 antonym candidates (before clustering):")
    for i, c in enumerate(candidates[:30], 1):
        known = discoverer.known_antonyms.get(word, set())
        marker = "★" if c['word'] in known else " "
        print(f"  {i:2d}. {marker} {c['word']:15s} "
              f"(antonym: {c['antonym_score']:.2f}, sim: {c['cosine']:.2f})")

    # Try different cluster numbers
    print("\n" + "─" * 60)
    print("CLUSTERING ANALYSIS")
    print("─" * 60)

    for n_clusters in [3, 5, 7, 10]:
        print(f"\n--- {n_clusters} clusters ---")
        axes = discoverer.cluster_axes(word, candidates, n_clusters=n_clusters)

        for axis in axes:
            members = [w for w, _ in axis.cluster_words[:5]]
            print(f"  {axis.representative_antonym:15s}: {', '.join(members)}")

    return discoverer


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        detailed_analysis(sys.argv[1])
    else:
        run_auto_discovery()
