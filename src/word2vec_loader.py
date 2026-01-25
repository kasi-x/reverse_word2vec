"""
Word2Vec model loader with support for various pre-trained models.
"""
import os
from typing import Optional

import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api


class Word2VecLoader:
    """Loader for Word2Vec models."""

    def __init__(self):
        self.model: Optional[KeyedVectors] = None
        self.model_name: Optional[str] = None

    def load_google_news(self, use_api: bool = True) -> KeyedVectors:
        """
        Load Google News pre-trained Word2Vec model.

        Args:
            use_api: If True, use gensim's downloader API.
                     If False, expect manual download.

        Returns:
            KeyedVectors model.
        """
        print("Loading Google News Word2Vec model (this may take a while)...")

        if use_api:
            self.model = api.load('word2vec-google-news-300')
        else:
            # Manual loading from file
            model_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'data',
                'GoogleNews-vectors-negative300.bin.gz'
            )
            self.model = KeyedVectors.load_word2vec_format(
                model_path, binary=True
            )

        self.model_name = 'google-news-300'
        print(f"Loaded model with {len(self.model)} words, "
              f"dimension={self.model.vector_size}")
        return self.model

    def load_glove(self, dimensions: int = 100) -> KeyedVectors:
        """
        Load GloVe pre-trained model via gensim API.

        Args:
            dimensions: Vector dimensions (50, 100, 200, 300).

        Returns:
            KeyedVectors model.
        """
        valid_dims = [50, 100, 200, 300]
        if dimensions not in valid_dims:
            raise ValueError(f"dimensions must be one of {valid_dims}")

        model_name = f'glove-wiki-gigaword-{dimensions}'
        print(f"Loading {model_name}...")
        self.model = api.load(model_name)
        self.model_name = model_name
        print(f"Loaded model with {len(self.model)} words")
        return self.model

    def load_from_file(self, filepath: str, binary: bool = True) -> KeyedVectors:
        """
        Load a Word2Vec model from a file.

        Args:
            filepath: Path to the model file.
            binary: Whether the file is in binary format.

        Returns:
            KeyedVectors model.
        """
        print(f"Loading model from {filepath}...")
        self.model = KeyedVectors.load_word2vec_format(filepath, binary=binary)
        self.model_name = os.path.basename(filepath)
        print(f"Loaded model with {len(self.model)} words, "
              f"dimension={self.model.vector_size}")
        return self.model

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector for a word, returning None if not found."""
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_* first.")
        try:
            return self.model[word]
        except KeyError:
            return None

    def has_word(self, word: str) -> bool:
        """Check if word exists in vocabulary."""
        if self.model is None:
            return False
        return word in self.model

    def get_dimension(self) -> int:
        """Get vector dimension of loaded model."""
        if self.model is None:
            raise RuntimeError("No model loaded.")
        return self.model.vector_size


def list_available_models() -> list:
    """List available pre-trained models in gensim."""
    return list(api.info()['models'].keys())


if __name__ == "__main__":
    print("Available models:")
    for m in list_available_models():
        print(f"  - {m}")
