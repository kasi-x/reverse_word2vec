"""
Visualization utilities for antonym space analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Tuple, Dict, Optional

from antonym_space import AntonymSpace


def plot_antonym_axes_2d(
    space: AntonymSpace,
    words: List[str],
    axes_to_show: Tuple[int, int] = (0, 1),
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
):
    """
    Plot words on a 2D plane defined by two antonym axes.

    Args:
        space: The AntonymSpace object.
        words: Words to plot.
        axes_to_show: Tuple of (axis_x, axis_y) indices.
        figsize: Figure size.
        save_path: Path to save figure (optional).
    """
    ax_x, ax_y = axes_to_show

    if ax_x >= len(space.axes) or ax_y >= len(space.axes):
        raise ValueError(f"Axis index out of range. Max: {len(space.axes)-1}")

    projections = space.project_all_words(words)

    x_axis = space.axes[ax_x]
    y_axis = space.axes[ax_y]

    fig, ax = plt.subplots(figsize=figsize)

    xs = []
    ys = []
    labels = []

    for word, proj in projections.items():
        xs.append(proj[ax_x])
        ys.append(proj[ax_y])
        labels.append(word)

    ax.scatter(xs, ys, alpha=0.6)

    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), fontsize=8, alpha=0.7)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel(f"Axis {ax_x}: {x_axis.negative_word} ← → {x_axis.positive_word}")
    ax.set_ylabel(f"Axis {ax_y}: {y_axis.negative_word} ← → {y_axis.positive_word}")
    ax.set_title("Words in Antonym Space")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_near_zero_words(
    near_zero: List[Tuple[str, float, np.ndarray]],
    space: AntonymSpace,
    n_words: int = 50,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
):
    """
    Visualize words near the zero vector using PCA.

    Args:
        near_zero: List of (word, distance, projection) tuples.
        space: The AntonymSpace object.
        n_words: Number of words to plot.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    words = [w for w, d, p in near_zero[:n_words]]
    projections = np.array([p for w, d, p in near_zero[:n_words]])
    distances = [d for w, d, p in near_zero[:n_words]]

    if projections.shape[1] > 2:
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(projections)
    else:
        coords_2d = projections

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=distances,
        cmap='viridis_r',
        alpha=0.7,
        s=100
    )

    for i, word in enumerate(words):
        ax.annotate(
            word,
            (coords_2d[i, 0], coords_2d[i, 1]),
            fontsize=8,
            alpha=0.8
        )

    plt.colorbar(scatter, label='Distance from Zero')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_title("Words Near Zero in Antonym Space (PCA projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_word_profile(
    word: str,
    space: AntonymSpace,
    top_n_axes: int = 15,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot a word's projection profile across top antonym axes.

    Args:
        word: Word to analyze.
        space: The AntonymSpace object.
        top_n_axes: Number of top axes to show.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    proj = space.project_word(word)
    if proj is None:
        print(f"Word '{word}' not found in vocabulary")
        return

    n_axes = min(top_n_axes, len(space.axes))
    indices = np.argsort(np.abs(proj))[::-1][:n_axes]

    values = proj[indices]
    labels = [
        f"{space.axes[i].negative_word}|{space.axes[i].positive_word}"
        for i in indices
    ]

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['green' if v > 0 else 'red' for v in values]
    bars = ax.barh(range(n_axes), values, color=colors, alpha=0.7)

    ax.set_yticks(range(n_axes))
    ax.set_yticklabels(labels)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Projection Value")
    ax.set_title(f"Semantic Profile: '{word}'")

    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_antonym_pair_comparison(
    word1: str,
    word2: str,
    space: AntonymSpace,
    top_n_axes: int = 10,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
):
    """
    Compare two words' profiles across antonym axes.

    Args:
        word1, word2: Words to compare.
        space: The AntonymSpace object.
        top_n_axes: Number of axes to show.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    proj1 = space.project_word(word1)
    proj2 = space.project_word(word2)

    if proj1 is None or proj2 is None:
        print(f"One or both words not found in vocabulary")
        return

    diff = np.abs(proj1 - proj2)
    indices = np.argsort(diff)[::-1][:top_n_axes]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(top_n_axes)
    width = 0.35

    vals1 = proj1[indices]
    vals2 = proj2[indices]

    labels = [
        f"{space.axes[i].negative_word}|{space.axes[i].positive_word}"
        for i in indices
    ]

    ax.bar(x - width/2, vals1, width, label=word1, alpha=0.7)
    ax.bar(x + width/2, vals2, width, label=word2, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel("Projection Value")
    ax.set_title(f"Comparison: '{word1}' vs '{word2}'")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_coverage_breakdown(
    space: AntonymSpace,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot cumulative variance explained by antonym axes.

    Args:
        space: The AntonymSpace object.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    if space.basis_matrix is None:
        print("No basis matrix available")
        return

    # Sample words and compute per-axis variance
    sample_size = min(10000, len(space.word_vectors))
    sample_words = list(space.word_vectors.keys())[:sample_size]

    projections = []
    for word in sample_words:
        proj = space.project_word(word)
        if proj is not None:
            projections.append(proj)

    projections = np.array(projections)
    variances = np.var(projections, axis=0)

    cumulative = np.cumsum(variances) / np.sum(variances)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.bar(range(len(variances)), variances, alpha=0.7)
    ax1.set_xlabel("Axis Index")
    ax1.set_ylabel("Variance")
    ax1.set_title("Variance per Antonym Axis")

    ax2.plot(range(len(cumulative)), cumulative, 'b-', linewidth=2)
    ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90%')
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
    ax2.set_xlabel("Number of Axes")
    ax2.set_ylabel("Cumulative Variance Ratio")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
