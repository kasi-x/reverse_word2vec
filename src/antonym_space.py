"""
Antonym Space Construction and Analysis.

This module implements the core algorithm for constructing a semantic space
based on antonym relationships using linear algebra operations.

Mathematical Framework:
-----------------------
Given word vectors from Word2Vec and a set of antonym pairs, we construct
a new coordinate system where each antonym pair defines an axis.

For an antonym pair (w+, w-):
  - Direction vector: d = normalize(vec(w+) - vec(w-))
  - Projection of any word w onto this axis: proj = dot(vec(w), d)
  - Sign indicates which "pole" the word is closer to

The algorithm uses Gram-Schmidt orthogonalization to ensure the antonym
axes form an orthonormal basis, maximizing the information captured by
each new dimension.
"""
import numpy as np
from numpy.linalg import norm, qr, svd, lstsq
from scipy.linalg import orth
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class AntonymAxis:
    """Represents a single antonym axis in the semantic space."""
    positive_word: str
    negative_word: str
    direction: np.ndarray  # Normalized direction vector
    midpoint: np.ndarray   # Midpoint between the two words
    separation: float      # Distance between the words


class AntonymSpace:
    """
    Constructs and analyzes a semantic space based on antonym relationships.

    This class implements an orthogonal projection-based approach where:
    1. Each antonym pair defines a semantic direction
    2. Directions are orthogonalized to form a basis
    3. Words are projected onto this basis for analysis
    """

    def __init__(self, word_vectors: Dict[str, np.ndarray], dimension: int):
        """
        Initialize the antonym space.

        Args:
            word_vectors: Dictionary mapping words to their embedding vectors.
            dimension: Dimension of the word vectors.
        """
        self.word_vectors = word_vectors
        self.original_dimension = dimension
        self.axes: List[AntonymAxis] = []
        self.basis_matrix: Optional[np.ndarray] = None  # Orthonormal basis
        self.used_pairs: List[Tuple[str, str]] = []

    def _get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get word vector, return None if not found."""
        return self.word_vectors.get(word)

    def compute_antonym_direction(
        self, word1: str, word2: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Compute the direction vector for an antonym pair.

        Returns:
            Tuple of (direction, midpoint, separation) or None if words not found.
        """
        v1 = self._get_vector(word1)
        v2 = self._get_vector(word2)

        if v1 is None or v2 is None:
            return None

        # Direction from word2 to word1
        direction = v1 - v2
        separation = norm(direction)

        if separation < 1e-10:
            return None

        direction = direction / separation
        midpoint = (v1 + v2) / 2

        return direction, midpoint, separation

    def build_axes_greedy(
        self,
        antonym_pairs: List[Tuple[str, str]],
        max_axes: Optional[int] = None,
        min_orthogonality: float = 0.1
    ) -> int:
        """
        Build orthogonal antonym axes using a greedy approach.

        Algorithm:
        1. For each antonym pair, compute direction vector
        2. Project out components along existing axes (Gram-Schmidt)
        3. If residual is large enough, add as new axis

        Args:
            antonym_pairs: List of (word1, word2) antonym pairs.
            max_axes: Maximum number of axes to create.
            min_orthogonality: Minimum norm of orthogonal component to accept.

        Returns:
            Number of axes created.
        """
        if max_axes is None:
            max_axes = min(len(antonym_pairs), self.original_dimension)

        directions = []

        for w1, w2 in tqdm(antonym_pairs, desc="Building axes"):
            if len(self.axes) >= max_axes:
                break

            result = self.compute_antonym_direction(w1, w2)
            if result is None:
                continue

            direction, midpoint, separation = result

            # Gram-Schmidt: project out existing directions
            if directions:
                basis = np.array(directions).T  # d x k matrix
                # Compute projection onto span of existing directions
                proj = basis @ (basis.T @ direction)
                orthogonal = direction - proj
            else:
                orthogonal = direction

            orthogonal_norm = norm(orthogonal)

            if orthogonal_norm >= min_orthogonality:
                # Normalize and add
                orthogonal = orthogonal / orthogonal_norm
                directions.append(orthogonal)
                self.axes.append(AntonymAxis(
                    positive_word=w1,
                    negative_word=w2,
                    direction=orthogonal,
                    midpoint=midpoint,
                    separation=separation
                ))
                self.used_pairs.append((w1, w2))

        if directions:
            self.basis_matrix = np.array(directions).T  # d x k matrix

        return len(self.axes)

    def build_axes_svd(
        self,
        antonym_pairs: List[Tuple[str, str]],
        max_axes: Optional[int] = None,
        explained_variance_ratio: float = 0.95
    ) -> int:
        """
        Build antonym axes using SVD for optimal dimensionality reduction.

        This approach:
        1. Collects all valid antonym direction vectors
        2. Uses SVD to find the principal directions
        3. Selects axes that capture most variance

        Args:
            antonym_pairs: List of antonym pairs.
            max_axes: Maximum number of axes (if None, determined by variance).
            explained_variance_ratio: Minimum cumulative variance to explain.

        Returns:
            Number of axes created.
        """
        # Collect all antonym directions
        direction_data = []

        for w1, w2 in tqdm(antonym_pairs, desc="Computing directions"):
            result = self.compute_antonym_direction(w1, w2)
            if result is not None:
                direction, midpoint, separation = result
                direction_data.append((w1, w2, direction, midpoint, separation))

        if not direction_data:
            return 0

        # Stack directions into matrix
        D = np.array([d[2] for d in direction_data])  # n x d matrix

        # SVD of direction matrix
        U, S, Vh = svd(D, full_matrices=False)

        # Determine number of components
        total_variance = np.sum(S ** 2)
        cumulative_variance = np.cumsum(S ** 2) / total_variance

        if max_axes is None:
            n_components = np.searchsorted(
                cumulative_variance, explained_variance_ratio
            ) + 1
            n_components = min(n_components, len(S))
        else:
            n_components = min(max_axes, len(S))

        # Principal directions are rows of Vh
        self.basis_matrix = Vh[:n_components].T  # d x k matrix

        # Associate each axis with the most aligned antonym pair
        for i in range(n_components):
            principal_dir = Vh[i]
            # Find most aligned original direction
            alignments = np.abs(D @ principal_dir)
            best_idx = np.argmax(alignments)
            w1, w2, _, midpoint, separation = direction_data[best_idx]

            # Ensure direction aligns with the original pair
            if np.dot(D[best_idx], principal_dir) < 0:
                principal_dir = -principal_dir

            self.axes.append(AntonymAxis(
                positive_word=w1,
                negative_word=w2,
                direction=principal_dir,
                midpoint=midpoint,
                separation=separation
            ))
            self.used_pairs.append((w1, w2))

        return n_components

    def project_word(self, word: str) -> Optional[np.ndarray]:
        """
        Project a word onto the antonym space.

        Returns:
            k-dimensional vector of projections onto each antonym axis.
        """
        if self.basis_matrix is None:
            raise RuntimeError("Axes not built. Call build_axes_* first.")

        vec = self._get_vector(word)
        if vec is None:
            return None

        # Project onto antonym basis: B^T @ v
        return self.basis_matrix.T @ vec

    def project_all_words(
        self, words: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Project all words (or specified subset) onto antonym space.

        Args:
            words: List of words to project. If None, use all words.

        Returns:
            Dictionary mapping words to their projections.
        """
        if words is None:
            words = list(self.word_vectors.keys())

        projections = {}
        for word in tqdm(words, desc="Projecting words"):
            proj = self.project_word(word)
            if proj is not None:
                projections[word] = proj

        return projections

    def compute_residual(self, word: str) -> Optional[np.ndarray]:
        """
        Compute the residual vector (component not captured by antonym axes).

        Args:
            word: Word to analyze.

        Returns:
            Residual vector in original space.
        """
        if self.basis_matrix is None:
            raise RuntimeError("Axes not built.")

        vec = self._get_vector(word)
        if vec is None:
            return None

        # Project and reconstruct: B @ B^T @ v
        projection = self.basis_matrix @ (self.basis_matrix.T @ vec)
        residual = vec - projection

        return residual

    def find_near_zero_words(
        self,
        words: Optional[List[str]] = None,
        threshold: float = 0.1,
        top_n: int = 100
    ) -> List[Tuple[str, float, np.ndarray]]:
        """
        Find words whose projections are near the zero vector.

        These are words that are "neutral" with respect to all antonym axes.

        Args:
            words: Words to search (None = all words).
            threshold: Maximum norm to consider "near zero".
            top_n: Return top N closest words to zero.

        Returns:
            List of (word, norm, projection) tuples sorted by norm.
        """
        projections = self.project_all_words(words)

        near_zero = []
        for word, proj in projections.items():
            proj_norm = norm(proj)
            near_zero.append((word, proj_norm, proj))

        near_zero.sort(key=lambda x: x[1])

        return near_zero[:top_n]

    def find_potential_antonyms(
        self,
        word: str,
        candidates: Optional[List[str]] = None,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find potential antonyms for a word based on opposite positions
        in the antonym space.

        Args:
            word: Word to find antonyms for.
            candidates: Candidate words (None = all words).
            top_n: Number of candidates to return.

        Returns:
            List of (candidate_word, score) tuples.
        """
        proj = self.project_word(word)
        if proj is None:
            return []

        if candidates is None:
            candidates = list(self.word_vectors.keys())

        # Find words with opposite projections
        scores = []
        for candidate in candidates:
            if candidate == word:
                continue
            cand_proj = self.project_word(candidate)
            if cand_proj is None:
                continue

            # Score based on negative correlation of projections
            # Higher (more positive) score = more opposite
            score = -np.dot(proj, cand_proj) / (norm(proj) * norm(cand_proj) + 1e-10)
            scores.append((candidate, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def analyze_axis_coverage(self) -> Dict[str, float]:
        """
        Analyze how much of the vocabulary is covered by the antonym space.

        Returns:
            Dictionary with coverage statistics.
        """
        if self.basis_matrix is None:
            raise RuntimeError("Axes not built.")

        total_variance = 0.0
        explained_variance = 0.0

        for word, vec in tqdm(
            self.word_vectors.items(),
            desc="Analyzing coverage"
        ):
            vec_norm_sq = np.dot(vec, vec)
            total_variance += vec_norm_sq

            proj = self.basis_matrix.T @ vec
            proj_norm_sq = np.dot(proj, proj)
            explained_variance += proj_norm_sq

        return {
            'total_variance': total_variance,
            'explained_variance': explained_variance,
            'explained_ratio': explained_variance / total_variance,
            'num_axes': len(self.axes),
            'num_words': len(self.word_vectors),
        }


def create_antonym_space_from_model(
    model,  # gensim KeyedVectors
    antonym_pairs: List[Tuple[str, str]],
    method: str = 'greedy',
    **kwargs
) -> AntonymSpace:
    """
    Convenience function to create AntonymSpace from a gensim model.

    Args:
        model: Gensim KeyedVectors object.
        antonym_pairs: List of antonym pairs.
        method: 'greedy' or 'svd'.
        **kwargs: Additional arguments for the build method.

    Returns:
        Configured AntonymSpace object.
    """
    # Extract word vectors as dictionary
    word_vectors = {word: model[word] for word in model.key_to_index}

    space = AntonymSpace(word_vectors, model.vector_size)

    if method == 'greedy':
        space.build_axes_greedy(antonym_pairs, **kwargs)
    elif method == 'svd':
        space.build_axes_svd(antonym_pairs, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    return space
