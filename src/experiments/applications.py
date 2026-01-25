"""
Applications of Antonym Vectors

Beyond discovering new antonym relationships, antonym vectors have many
practical applications in NLP.
"""
import sys
sys.path.insert(0, '..')

import numpy as np
from numpy.linalg import norm, svd
from typing import List, Dict, Tuple

from antonym_loader import extract_antonym_pairs
from word2vec_loader import Word2VecLoader


class AntonymApplications:
    """
    Various applications of antonym direction vectors.
    """

    def __init__(self, model, antonym_pairs: List[Tuple[str, str]]):
        self.model = model
        self.dim = model.vector_size

        # Build antonym direction matrix
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

        # Vocabulary
        self.vocab = [
            w for w in model.key_to_index
            if w.isalpha() and w == w.lower() and 3 <= len(w) <= 15
        ][:30000]

    # =================================================================
    # Application 1: Sentiment Analysis
    # =================================================================

    def create_sentiment_axis(self, positive: str = 'good', negative: str = 'bad') -> np.ndarray:
        """Create a sentiment axis from positive and negative poles."""
        direction = self.model[positive] - self.model[negative]
        return direction / norm(direction)

    def measure_word_sentiment(self, word: str, axis: np.ndarray) -> float:
        """Measure a word's sentiment on the given axis."""
        if word not in self.model:
            return 0.0
        return float(np.dot(self.model[word], axis))

    def measure_text_sentiment(self, text: str, axis: np.ndarray) -> Dict:
        """Analyze sentiment of a text."""
        words = [w.lower().strip('.,!?') for w in text.split()]
        valid_words = [w for w in words if w in self.model]

        if not valid_words:
            return {'score': 0.0, 'label': 'unknown', 'words': []}

        vectors = [self.model[w] for w in valid_words]
        avg_vec = np.mean(vectors, axis=0)
        score = float(np.dot(avg_vec, axis))

        if score > 0.3:
            label = 'positive'
        elif score < -0.3:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            'score': score,
            'label': label,
            'words': [(w, self.measure_word_sentiment(w, axis)) for w in valid_words]
        }

    # =================================================================
    # Application 2: Synonym vs Antonym Distinction
    # =================================================================

    def classify_word_relation(self, word1: str, word2: str) -> Dict:
        """
        Classify the relationship between two words.

        Uses both cosine similarity and antonym alignment to distinguish:
        - Synonyms: high similarity, low antonym score
        - Antonyms: high similarity, high antonym score
        - Related: moderate similarity
        - Unrelated: low similarity
        """
        if word1 not in self.model or word2 not in self.model:
            return {'relation': 'unknown'}

        v1 = self.model[word1]
        v2 = self.model[word2]

        # Cosine similarity
        cosine = np.dot(v1, v2) / (norm(v1) * norm(v2))

        # Antonym score
        diff = v1 - v2
        diff_norm = norm(diff)
        if diff_norm > 1e-6:
            antonym_score = np.max(np.abs(self.D @ (diff / diff_norm)))
        else:
            antonym_score = 0

        # Classification
        if cosine > 0.7 and antonym_score < 0.5:
            relation = 'synonym'
        elif cosine > 0.3 and antonym_score > 0.6:
            relation = 'antonym'
        elif cosine < 0.3:
            relation = 'unrelated'
        else:
            relation = 'related'

        return {
            'relation': relation,
            'cosine_similarity': float(cosine),
            'antonym_score': float(antonym_score)
        }

    # =================================================================
    # Application 3: Bias Detection
    # =================================================================

    def create_bias_axis(self, group_a: str, group_b: str) -> np.ndarray:
        """Create a bias detection axis (e.g., he-she for gender)."""
        direction = self.model[group_a] - self.model[group_b]
        return direction / norm(direction)

    def measure_bias(self, words: List[str], axis: np.ndarray) -> List[Tuple[str, float]]:
        """Measure bias of words on the given axis."""
        results = []
        for w in words:
            if w in self.model:
                proj = float(np.dot(self.model[w], axis))
                results.append((w, proj))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def detect_gender_bias(self, words: List[str]) -> List[Tuple[str, float]]:
        """Detect gender bias in occupation words."""
        axis = self.create_bias_axis('he', 'she')
        return self.measure_bias(words, axis)

    # =================================================================
    # Application 4: Intensity Gradation
    # =================================================================

    def create_intensity_scale(self, positive: str, negative: str) -> List[Tuple[str, float]]:
        """
        Create an intensity scale between two poles.

        Returns words ordered from negative pole to positive pole.
        """
        direction = self.model[positive] - self.model[negative]
        direction = direction / norm(direction)
        midpoint = (self.model[positive] + self.model[negative]) / 2
        pole_dist = norm(self.model[positive] - midpoint)

        projections = []
        for w in self.vocab:
            if w in self.model:
                proj = np.dot(self.model[w] - midpoint, direction) / pole_dist
                projections.append((w, float(proj)))

        projections.sort(key=lambda x: x[1])
        return projections

    def get_intensity_order(self, words: List[str], positive: str, negative: str) -> List[Tuple[str, float]]:
        """Get the intensity ordering of specific words on a scale."""
        scale = self.create_intensity_scale(positive, negative)
        word_set = set(words)
        return [(w, p) for w, p in scale if w in word_set]

    # =================================================================
    # Application 5: Semantic Space Analysis
    # =================================================================

    def analyze_antonym_space(self) -> Dict:
        """
        Analyze the structure of antonym space using SVD.
        """
        U, S, Vh = svd(self.D, full_matrices=False)

        total_var = np.sum(S ** 2)
        var_explained = S ** 2 / total_var
        cumvar = np.cumsum(var_explained)

        # Find dimensions needed for 90% variance
        dims_90 = np.searchsorted(cumvar, 0.9) + 1

        return {
            'n_directions': len(self.D),
            'embedding_dim': self.dim,
            'singular_values': S[:20].tolist(),
            'variance_explained': var_explained[:20].tolist(),
            'cumulative_variance': cumvar[:20].tolist(),
            'dims_for_90_percent': int(dims_90)
        }

    # =================================================================
    # Application 6: Neutrality Measurement
    # =================================================================

    def measure_neutrality(self, word: str) -> float:
        """
        Measure how neutral a word is on antonym dimensions.

        Lower score = more neutral (less "charged" semantically).
        """
        if word not in self.model:
            return None
        v = self.model[word]
        projections = self.D @ v
        return float(np.sqrt(np.sum(projections ** 2)))

    def compare_neutrality(self, word_groups: Dict[str, List[str]]) -> Dict:
        """
        Compare neutrality across different word groups.
        """
        results = {}
        for group_name, words in word_groups.items():
            scores = []
            for w in words:
                score = self.measure_neutrality(w)
                if score is not None:
                    scores.append((w, score))

            if scores:
                avg = np.mean([s for _, s in scores])
                results[group_name] = {
                    'average': float(avg),
                    'words': sorted(scores, key=lambda x: x[1])
                }

        return results


def run_all_applications():
    """Demonstrate all applications."""
    print("=" * 70)
    print("ANTONYM VECTOR APPLICATIONS")
    print("=" * 70)

    # Load
    print("\nLoading data...")
    antonym_pairs = extract_antonym_pairs()
    loader = Word2VecLoader()
    model = loader.load_glove(100)

    app = AntonymApplications(model, antonym_pairs)

    # Application 1: Sentiment
    print("\n" + "=" * 70)
    print("APPLICATION 1: Sentiment Analysis")
    print("=" * 70)

    sentiment_axis = app.create_sentiment_axis('good', 'bad')

    texts = [
        "This movie is absolutely wonderful and amazing",
        "The food was okay but nothing special",
        "That was the worst experience ever",
    ]

    for text in texts:
        result = app.measure_text_sentiment(text, sentiment_axis)
        print(f"\n  \"{text}\"")
        print(f"  → {result['label'].upper()} (score: {result['score']:.2f})")

    # Application 2: Synonym vs Antonym
    print("\n" + "=" * 70)
    print("APPLICATION 2: Synonym vs Antonym Distinction")
    print("=" * 70)

    pairs = [
        ('happy', 'joyful'),
        ('happy', 'sad'),
        ('big', 'large'),
        ('big', 'small'),
        ('hot', 'warm'),
        ('hot', 'cold'),
    ]

    for w1, w2 in pairs:
        result = app.classify_word_relation(w1, w2)
        print(f"  {w1} - {w2}: {result['relation'].upper()}")

    # Application 3: Bias
    print("\n" + "=" * 70)
    print("APPLICATION 3: Gender Bias Detection")
    print("=" * 70)

    occupations = ['doctor', 'nurse', 'engineer', 'teacher', 'ceo', 'secretary']
    biases = app.detect_gender_bias(occupations)

    print("\n  Occupation      | Gender Bias")
    print("  " + "-" * 35)
    for word, score in biases:
        direction = "Male ←" if score > 0 else "→ Female"
        print(f"  {word:15s} | {direction} ({score:+.2f})")

    # Application 4: Intensity
    print("\n" + "=" * 70)
    print("APPLICATION 4: Intensity Gradation")
    print("=" * 70)

    temp_words = ['freezing', 'cold', 'cool', 'warm', 'hot', 'boiling']
    temp_order = app.get_intensity_order(temp_words, 'hot', 'cold')

    print("\n  Temperature scale:")
    for w, pos in temp_order:
        print(f"    {w:12s}: {pos:+.2f}")

    # Application 5: Space Analysis
    print("\n" + "=" * 70)
    print("APPLICATION 5: Semantic Space Analysis")
    print("=" * 70)

    analysis = app.analyze_antonym_space()
    print(f"\n  Antonym directions: {analysis['n_directions']}")
    print(f"  Embedding dimension: {analysis['embedding_dim']}")
    print(f"  Dimensions for 90% variance: {analysis['dims_for_90_percent']}")

    # Application 6: Neutrality
    print("\n" + "=" * 70)
    print("APPLICATION 6: Neutrality Measurement")
    print("=" * 70)

    word_groups = {
        'Emotion': ['happy', 'sad', 'angry', 'love'],
        'Function': ['the', 'and', 'but', 'however'],
        'Concrete': ['table', 'chair', 'book', 'car'],
    }

    neutrality = app.compare_neutrality(word_groups)

    for group, data in neutrality.items():
        print(f"\n  {group} (avg: {data['average']:.1f}):")
        for w, s in data['words'][:4]:
            print(f"    {w:12s}: {s:.1f}")

    return app


if __name__ == "__main__":
    run_all_applications()
