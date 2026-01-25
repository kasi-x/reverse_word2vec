"""
Advanced Antonym Analysis with Semantic Clustering and Bidirectional Verification.

Improvements over the basic approach:
1. Cluster antonym directions into semantic categories (valence, size, speed, etc.)
2. Bidirectional verification (if A→B then B→A)
3. Semantic relatedness filter (antonyms should be related concepts)
4. Interpretable results showing which dimensions differ
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from numpy.linalg import norm, svd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
from tqdm import tqdm

from antonym_loader import extract_antonym_pairs, group_antonyms_by_word
from word2vec_loader import Word2VecLoader


def is_valid_word(word: str) -> bool:
    """Filter out noise tokens."""
    if not word.isalpha():
        return False
    if word != word.lower():
        return False
    if len(word) < 3 or len(word) > 15:
        return False
    # Filter common noise patterns
    if word.endswith('bb') or word.startswith('afp'):
        return False
    return True


@dataclass
class SemanticCluster:
    """A cluster of related antonym directions."""
    id: int
    name: str  # Auto-generated or manual label
    centroid: np.ndarray
    example_pairs: List[Tuple[str, str]]
    direction_indices: List[int]


@dataclass
class AntonymCandidate:
    """A potential antonym with detailed scoring."""
    word: str
    total_score: float
    bidirectional_score: float  # How well the reverse also works
    relatedness: float  # Cosine similarity in original space
    opposing_clusters: List[Tuple[int, float]]  # Which semantic dimensions differ


class AdvancedAntonymAnalyzer:
    """
    Advanced analyzer with clustering, bidirectional verification, and interpretability.
    """

    def __init__(self, model, antonym_pairs: List[Tuple[str, str]], n_clusters: int = 10):
        self.model = model
        self.dim = model.vector_size
        self.n_clusters = n_clusters

        # Build antonym direction matrix
        self.directions = []
        self.pair_labels = []
        self.pair_to_idx = {}

        print("Building antonym direction matrix...")
        for w1, w2 in tqdm(antonym_pairs, desc="Processing pairs"):
            if w1 in model and w2 in model:
                v1, v2 = model[w1], model[w2]
                d = v1 - v2
                d_norm = norm(d)
                if d_norm > 1e-6:
                    idx = len(self.directions)
                    self.directions.append(d / d_norm)
                    self.pair_labels.append((w1, w2))
                    self.pair_to_idx[(w1, w2)] = idx
                    self.pair_to_idx[(w2, w1)] = idx

        self.D = np.array(self.directions)
        print(f"  {len(self.directions)} valid antonym directions")

        # Cluster the antonym directions
        self._cluster_directions()

        # Precompute word vectors for valid words
        self.valid_words = [w for w in model.key_to_index.keys() if is_valid_word(w)]
        print(f"  {len(self.valid_words)} valid words")

    def _cluster_directions(self):
        """Cluster antonym directions into semantic categories."""
        print(f"Clustering into {self.n_clusters} semantic categories...")

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.D)

        self.clusters: List[SemanticCluster] = []

        for i in range(self.n_clusters):
            indices = np.where(labels == i)[0]
            centroid = kmeans.cluster_centers_[i]
            centroid = centroid / (norm(centroid) + 1e-10)

            # Get example pairs for this cluster
            examples = [self.pair_labels[idx] for idx in indices[:5]]

            # Auto-generate cluster name from examples
            name = self._generate_cluster_name(examples)

            self.clusters.append(SemanticCluster(
                id=i,
                name=name,
                centroid=centroid,
                example_pairs=examples,
                direction_indices=indices.tolist()
            ))

        # Print cluster summary
        print("\nDiscovered semantic categories:")
        for c in self.clusters:
            examples_str = ", ".join([f"{p[0]}/{p[1]}" for p in c.example_pairs[:3]])
            print(f"  [{c.id}] {c.name}: {examples_str}")

    def _generate_cluster_name(self, examples: List[Tuple[str, str]]) -> str:
        """Generate a descriptive name for a cluster based on examples."""
        if not examples:
            return "unknown"

        # Use the first example as the name
        w1, w2 = examples[0]
        return f"{w1[:8]}/{w2[:8]}"

    def get_cluster_projection(self, word: str) -> Optional[np.ndarray]:
        """Project word onto each cluster centroid."""
        if word not in self.model:
            return None
        v = self.model[word]
        projections = np.array([np.dot(v, c.centroid) for c in self.clusters])
        return projections

    def find_antonyms_advanced(
        self,
        word: str,
        candidates: Optional[List[str]] = None,
        top_n: int = 10,
        min_relatedness: float = 0.1,
        require_bidirectional: bool = True
    ) -> List[AntonymCandidate]:
        """
        Find antonyms with advanced scoring and verification.

        Args:
            word: Query word
            candidates: Candidate words (None = use all valid words)
            top_n: Number of results
            min_relatedness: Minimum cosine similarity in original space
            require_bidirectional: Require bidirectional verification
        """
        if word not in self.model:
            return []

        v_word = self.model[word]
        word_cluster_proj = self.get_cluster_projection(word)

        if candidates is None:
            candidates = self.valid_words[:50000]

        results = []

        for cand in candidates:
            if cand == word or cand not in self.model:
                continue

            v_cand = self.model[cand]

            # 1. Semantic relatedness (cosine similarity)
            relatedness = np.dot(v_word, v_cand) / (norm(v_word) * norm(v_cand) + 1e-10)

            # Skip if too unrelated (antonyms should be semantically related)
            if relatedness < min_relatedness:
                continue

            # 2. Compute difference and align with antonym directions
            diff = v_word - v_cand
            diff_norm = norm(diff)
            if diff_norm < 1e-6:
                continue
            diff_normalized = diff / diff_norm

            # Score based on alignment with antonym directions
            alignments = self.D @ diff_normalized
            forward_score = np.max(np.abs(alignments))

            # 3. Bidirectional verification
            if require_bidirectional:
                diff_reverse = v_cand - v_word
                diff_reverse_normalized = diff_reverse / norm(diff_reverse)
                reverse_alignments = self.D @ diff_reverse_normalized
                reverse_score = np.max(np.abs(reverse_alignments))
                bidirectional_score = min(forward_score, reverse_score)
            else:
                bidirectional_score = forward_score

            # 4. Find which clusters show opposition
            cand_cluster_proj = self.get_cluster_projection(cand)
            cluster_diffs = word_cluster_proj - cand_cluster_proj
            opposing_clusters = [
                (i, float(cluster_diffs[i]))
                for i in np.argsort(-np.abs(cluster_diffs))[:3]
            ]

            # 5. Combined score (penalize low relatedness slightly)
            total_score = bidirectional_score * (0.5 + 0.5 * relatedness)

            results.append(AntonymCandidate(
                word=cand,
                total_score=total_score,
                bidirectional_score=bidirectional_score,
                relatedness=relatedness,
                opposing_clusters=opposing_clusters
            ))

        # Sort by total score
        results.sort(key=lambda x: x.total_score, reverse=True)
        return results[:top_n]

    def find_neutral_words_by_cluster(
        self,
        words: Optional[List[str]] = None,
        top_n: int = 50
    ) -> List[Tuple[str, float, np.ndarray]]:
        """
        Find words that are neutral across all semantic clusters.
        Returns words with their neutrality score and cluster projections.
        """
        if words is None:
            words = self.valid_words[:50000]

        results = []
        for word in tqdm(words, desc="Finding neutrals"):
            proj = self.get_cluster_projection(word)
            if proj is not None:
                neutrality = norm(proj)
                results.append((word, neutrality, proj))

        results.sort(key=lambda x: x[1])
        return results[:top_n]

    def analyze_word_profile(self, word: str) -> Optional[Dict]:
        """
        Get detailed semantic profile of a word across antonym clusters.
        """
        if word not in self.model:
            return None

        proj = self.get_cluster_projection(word)
        if proj is None:
            return None

        profile = {
            'word': word,
            'cluster_projections': {}
        }

        for i, (cluster, p) in enumerate(zip(self.clusters, proj)):
            direction = "+" if p > 0 else "-"
            example = cluster.example_pairs[0] if cluster.example_pairs else ("?", "?")
            pole = example[0] if p > 0 else example[1]

            profile['cluster_projections'][cluster.name] = {
                'value': float(p),
                'magnitude': abs(float(p)),
                'direction': direction,
                'pole_word': pole
            }

        return profile

    def discover_antonym_pairs(
        self,
        words: List[str],
        threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """
        Discover new antonym pairs from a list of words.
        Uses bidirectional verification and clustering coherence.
        """
        discovered = []
        checked = set()

        for word in tqdm(words, desc="Discovering pairs"):
            if word not in self.model:
                continue

            candidates = self.find_antonyms_advanced(
                word,
                candidates=words,
                top_n=5,
                require_bidirectional=True
            )

            for cand in candidates:
                pair = tuple(sorted([word, cand.word]))
                if pair in checked:
                    continue
                checked.add(pair)

                if cand.bidirectional_score >= threshold:
                    discovered.append((word, cand.word, cand.bidirectional_score))

        # Remove duplicates and sort
        discovered = list(set(discovered))
        discovered.sort(key=lambda x: x[2], reverse=True)

        return discovered


def main():
    print("=" * 70)
    print("ADVANCED ANTONYM ANALYSIS")
    print("with Clustering, Bidirectional Verification, and Interpretability")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    antonym_pairs = extract_antonym_pairs()
    known_antonyms = group_antonyms_by_word()

    # Load model
    print("\n[2/3] Loading model...")
    loader = Word2VecLoader()
    model = loader.load_glove(100)

    # Build analyzer
    print("\n[3/3] Building advanced analyzer...")
    analyzer = AdvancedAntonymAnalyzer(model, antonym_pairs, n_clusters=12)

    # Test on known antonym pairs
    print("\n" + "=" * 70)
    print("TEST 1: Verification on Known Antonym Pairs")
    print("=" * 70)

    test_pairs = [
        ("good", "bad"), ("happy", "sad"), ("love", "hate"),
        ("hot", "cold"), ("light", "dark"), ("big", "small"),
        ("fast", "slow"), ("young", "old"), ("rich", "poor"),
        ("strong", "weak"), ("open", "close"), ("life", "death")
    ]

    print("\n{:<10} {:<10} {:>8} {:>8} {:>10}  {}".format(
        "Word", "Expected", "Found?", "Rank", "Score", "Top 3 Discovered"
    ))
    print("-" * 70)

    for w1, w2 in test_pairs:
        if w1 not in model:
            continue

        results = analyzer.find_antonyms_advanced(w1, top_n=20)
        found_words = [r.word for r in results]

        if w2 in found_words:
            rank = found_words.index(w2) + 1
            score = results[rank-1].total_score
            status = "✓"
        else:
            rank = "-"
            score = 0
            status = "✗"

        top3 = ", ".join(found_words[:3])
        print(f"{w1:<10} {w2:<10} {status:>8} {str(rank):>8} {score:>10.3f}  {top3}")

    # Analyze semantic profiles
    print("\n" + "=" * 70)
    print("TEST 2: Semantic Profiles of Words")
    print("=" * 70)

    profile_words = ["happy", "sad", "big", "small", "good", "evil"]
    for word in profile_words:
        profile = analyzer.analyze_word_profile(word)
        if profile:
            print(f"\n{word.upper()}:")
            # Show top 3 strongest dimensions
            dims = list(profile['cluster_projections'].items())
            dims.sort(key=lambda x: x[1]['magnitude'], reverse=True)
            for name, data in dims[:4]:
                print(f"  {name:<20}: {data['value']:+.3f} (→ {data['pole_word']})")

    # Find neutral words
    print("\n" + "=" * 70)
    print("TEST 3: Semantically Neutral Words")
    print("=" * 70)

    neutrals = analyzer.find_neutral_words_by_cluster(top_n=30)
    print("\nTop 20 neutral words (low projection on all antonym dimensions):")
    for i, (word, score, _) in enumerate(neutrals[:20]):
        print(f"  {i+1:2d}. {word:<20} (neutrality: {score:.4f})")

    # Discover antonyms for abstract concepts
    print("\n" + "=" * 70)
    print("TEST 4: Antonym Discovery for Abstract Concepts")
    print("=" * 70)

    abstract_words = [
        "technology", "science", "democracy", "philosophy", "history",
        "nature", "culture", "time", "space", "power", "freedom",
        "truth", "beauty", "wisdom", "chaos", "peace", "war"
    ]

    print("\nDiscovered antonyms with detailed scoring:")
    for word in abstract_words:
        if word not in model:
            continue

        results = analyzer.find_antonyms_advanced(word, top_n=5)
        if results:
            print(f"\n{word.upper()}:")
            for r in results[:3]:
                # Show which dimensions differ most
                dims = [f"{analyzer.clusters[c[0]].name}({c[1]:+.2f})"
                        for c in r.opposing_clusters[:2]]
                dims_str = ", ".join(dims)
                print(f"  → {r.word:<15} score={r.total_score:.3f} "
                      f"rel={r.relatedness:.2f} [{dims_str}]")

    # Discover new antonym pairs
    print("\n" + "=" * 70)
    print("TEST 5: Discover New Antonym Pairs")
    print("=" * 70)

    # Use common adjectives
    common_words = [
        w for w in analyzer.valid_words[:5000]
        if len(w) >= 4 and w not in known_antonyms
    ][:500]

    print(f"\nSearching among {len(common_words)} words without known antonyms...")
    new_pairs = analyzer.discover_antonym_pairs(common_words, threshold=0.75)

    print(f"\nTop 20 discovered potential antonym pairs:")
    for w1, w2, score in new_pairs[:20]:
        print(f"  {w1:<15} ↔ {w2:<15} (score: {score:.3f})")


if __name__ == "__main__":
    main()
