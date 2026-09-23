"""
ICA (Independent Component Analysis) transformation of word embedding spaces.

Decomposes word vectors into statistically independent components,
each potentially corresponding to an interpretable semantic axis.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import FastICA


@dataclass
class ICASpace:
    """Container for an ICA-transformed embedding space."""

    words: list[str]
    word_to_idx: dict[str, int]
    S: np.ndarray  # (n_words, n_components) ICA score matrix
    mixing_matrix: np.ndarray  # (n_features, n_components) A in X = S @ A.T
    unmixing_matrix: np.ndarray  # (n_components, n_features) W in S = X @ W.T
    mean_vector: np.ndarray  # (n_features,) mean subtracted before ICA
    n_components: int
    meta: dict = field(default_factory=dict)  # provenance (model, seed, CF, ...)

    def score(self, word: str) -> np.ndarray | None:
        """Get ICA score vector for a word. Returns None if not found."""
        idx = self.word_to_idx.get(word)
        if idx is None:
            return None
        return self.S[idx]

    def top_words(self, axis: int, n: int = 10, positive: bool = True) -> list[str]:
        """Get top-n words on positive or negative end of an axis."""
        col = self.S[:, axis]
        if positive:
            indices = np.argsort(col)[-n:][::-1]
        else:
            indices = np.argsort(col)[:n]
        return [self.words[i] for i in indices]


class ICATransformer:
    """Fits ICA on word embeddings and provides transform/reconstruct."""

    def fit(
        self,
        model: KeyedVectors,
        n_components: int | None = None,
        vocab_limit: int = 50000,
        random_state: int = 42,
        max_iter: int = 1000,
    ) -> ICASpace:
        """
        Fit ICA on the top vocab_limit words from the model.

        Args:
            model: Gensim KeyedVectors model.
            n_components: Number of ICA components. Defaults to model dimension.
            vocab_limit: Max words to include.
            random_state: For reproducibility.
            max_iter: Max iterations for FastICA.

        Returns:
            ICASpace with fitted parameters.
        """
        words = []
        indices = []
        # index_to_key position == row index in model.vectors
        for pos, word in enumerate(model.index_to_key[:vocab_limit]):
            if is_valid_english_word(word):
                words.append(word)
                indices.append(pos)

        X = model.vectors[np.asarray(indices)].astype(np.float64)
        mean_vec = X.mean(axis=0)
        X_centered = X - mean_vec

        if n_components is None:
            n_components = X.shape[1]

        print(f"Fitting ICA: {len(words)} words, {X.shape[1]}d -> {n_components} components...")
        ica = FastICA(
            n_components=n_components,
            random_state=random_state,
            max_iter=max_iter,
            whiten="unit-variance",
        )
        S = ica.fit_transform(X_centered)

        word_to_idx = {w: i for i, w in enumerate(words)}
        space = ICASpace(
            words=words,
            word_to_idx=word_to_idx,
            S=S,
            mixing_matrix=ica.mixing_,
            unmixing_matrix=ica.components_,
            mean_vector=mean_vec,
            n_components=n_components,
        )
        print(f"ICA fit complete. Score matrix shape: {S.shape}")
        return space

    def transform_word(self, word: str, model: KeyedVectors, space: ICASpace) -> np.ndarray | None:
        """Transform a single word (possibly OOV for the ICA space) into ICA scores."""
        if word in space.word_to_idx:
            return space.S[space.word_to_idx[word]]
        if word not in model:
            return None
        vec = model[word].astype(np.float64) - space.mean_vector
        return vec @ space.unmixing_matrix.T

    def reconstruct(self, ica_scores: np.ndarray, space: ICASpace) -> np.ndarray:
        """Reconstruct an original-space vector from ICA scores."""
        return ica_scores @ space.mixing_matrix.T + space.mean_vector


def save_ica_space(space: ICASpace, path: str, meta: dict | None = None) -> None:
    """
    Save ICA space to disk (npz for arrays, json for metadata).

    Args:
        space: The fitted ICASpace.
        path: Base path (suffixes .npz/.json are appended).
        meta: Provenance metadata (model name, counter_fitted, vocab_limit,
            random_state, ...). A creation timestamp is added automatically.
    """
    base = Path(path)
    np.savez_compressed(
        str(base.with_suffix(".npz")),
        S=space.S,
        mixing_matrix=space.mixing_matrix,
        unmixing_matrix=space.unmixing_matrix,
        mean_vector=space.mean_vector,
    )
    provenance = {"created_at": datetime.now(UTC).isoformat()}
    provenance.update(meta or {})
    provenance.update(space.meta)
    meta_out = {
        "words": space.words,
        "n_components": space.n_components,
        "meta": provenance,
    }
    with open(str(base.with_suffix(".json")), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False)
    print(f"Saved ICA space to {base.with_suffix('.npz')} + {base.with_suffix('.json')}")


def load_ica_space(path: str) -> ICASpace:
    """Load ICA space from disk."""
    base = Path(path)
    data = np.load(str(base.with_suffix(".npz")))
    with open(str(base.with_suffix(".json")), encoding="utf-8") as f:
        meta = json.load(f)

    words = meta["words"]
    word_to_idx = {w: i for i, w in enumerate(words)}
    space = ICASpace(
        words=words,
        word_to_idx=word_to_idx,
        S=data["S"],
        mixing_matrix=data["mixing_matrix"],
        unmixing_matrix=data["unmixing_matrix"],
        mean_vector=data["mean_vector"],
        n_components=meta["n_components"],
        meta=meta.get("meta", {}),
    )
    print(f"Loaded ICA space: {len(words)} words, {space.n_components} components")
    return space


# Word-level filter matching the ascii-latin pattern used in GloVe
_VALID_WORD_RE = re.compile(r"^[a-z][a-z'-]*[a-z]$|^[a-z]$")


def is_valid_english_word(word: str) -> bool:
    """Return True if word looks like a real English word (lowercase, no digits/punctuation)."""
    return bool(_VALID_WORD_RE.match(word)) and "--" not in word and "''" not in word
