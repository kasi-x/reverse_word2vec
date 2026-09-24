"""
Experiment: hard negatives for the MLP antonym classifier.

Current training samples negatives uniformly at random — trivially
separable, so the MLP never learns the antonym/synonym boundary.
Nguyen et al. (2016) showed antonym detection needs negatives drawn
from the query's semantic neighbourhood.

Variants (same 70/15/15 split, seed 42, same MLP arch):
  random : 3:1 uniform negatives (current baseline)
  mixed  : 1.5:1 uniform + 1.5:1 hard (GloVe top-50 neighbours)
  hard   : 3:1 hard negatives only

Reports MLP-only Hits@k, pool-100 recall, and reranker Hits@k
(9-feature, pool=100 — the current pipeline).

Usage:
    pixi run python scripts/exp_hard_negatives.py
"""

import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from src.antonym_classifier import AntonymClassifier
from src.antonym_loader import extract_antonym_pairs
from src.eval_utils import compute_hits
from src.ica_transformer import load_ica_space
from src.reranker import Reranker
from src.word2vec_loader import Word2VecLoader

POOL_N = 100


class HardNegClassifier(AntonymClassifier):
    """AntonymClassifier with GloVe-neighbour hard negatives."""

    def __init__(self, space, model, hard_frac: float, hard_topn: int = 50):
        super().__init__(space, "mlp")
        self.model = model
        self.hard_frac = hard_frac
        self.hard_topn = hard_topn

    def _build_dataset(self, antonym_pairs, neg_ratio=3.0, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)

        X_pos, X_neg = [], []
        valid_pairs = []
        for w1, w2 in antonym_pairs:
            s1, s2 = self.space.score(w1), self.space.score(w2)
            if s1 is not None and s2 is not None:
                feat = self._features(s1, s2)
                X_pos.append(feat)
                X_pos.append(feat)
                valid_pairs.append((w1, w2))

        antonym_set = set(antonym_pairs) | {(w2, w1) for w1, w2 in antonym_pairs}
        n_neg = int(len(X_pos) * neg_ratio)
        n_hard = int(n_neg * self.hard_frac)

        # Hard negatives: GloVe neighbours of each positive's words
        hard_pool = []
        for w1, w2 in valid_pairs:
            for w in (w1, w2):
                if w in self.model:
                    for nb, _ in self.model.most_similar(w, topn=self.hard_topn):
                        if (w, nb) not in antonym_set and (nb, w) not in antonym_set:
                            hard_pool.append((w, nb))
        rng.shuffle(hard_pool)

        for w1, w2 in hard_pool:
            if len(X_neg) >= n_hard:
                break
            s1, s2 = self.space.score(w1), self.space.score(w2)
            if s1 is None or s2 is None:
                continue
            X_neg.append(self._features(s1, s2))

        # Fill the rest with uniform negatives
        words = self.space.words
        attempts = 0
        while len(X_neg) < n_neg and attempts < n_neg * 10:
            attempts += 1
            i, j = rng.randint(0, len(words), size=2)
            w1, w2 = words[i], words[j]
            if w1 == w2 or (w1, w2) in antonym_set:
                continue
            s1, s2 = self.space.score(w1), self.space.score(w2)
            if s1 is None or s2 is None:
                continue
            X_neg.append(self._features(s1, s2))

        X = np.array(X_pos + X_neg, dtype=np.float32)
        y = np.array([1] * len(X_pos) + [0] * len(X_neg), dtype=np.int32)
        return X, y


def main():
    t0 = time.time()
    print("Loading model and ICA space...")
    model = Word2VecLoader().load_glove(100)
    space = load_ica_space("results/ica_space")

    pairs = [
        (a, b)
        for a, b in extract_antonym_pairs()
        if space.score(a) is not None and space.score(b) is not None
    ]
    rng = np.random.RandomState(42)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n = len(pairs)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    train_pairs = [pairs[i] for i in idx[:n_train]]
    val_pairs = [pairs[i] for i in idx[n_train : n_train + n_val]]
    test_pairs = [pairs[i] for i in idx[n_train + n_val :]]
    print(f"Split: {len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}")

    variants = {
        "random": 0.0,
        "mixed": 0.5,
        "hard": 1.0,
    }
    results = {}

    for name, frac in variants.items():
        print(f"\n=== negatives: {name} (hard_frac={frac}) ===")
        mlp = HardNegClassifier(space, model, frac)
        mlp.fit(train_pairs, neg_ratio=3.0, rng=np.random.RandomState(42))

        mlp_hits, _ = compute_hits(
            test_pairs, lambda s, _m=mlp: [w for w, _ in _m.retrieve(s, top_n=10)]
        )
        # pool-100 recall
        rec = 0
        for w1, w2 in test_pairs:
            if any(
                tgt in [w for w, _ in mlp.retrieve(src, top_n=POOL_N)]
                for src, tgt in [(w1, w2), (w2, w1)]
            ):
                rec += 1
        recall = rec / len(test_pairs)

        rr = Reranker(space, model)
        rr.fit(val_pairs, mlp, top_n=POOL_N)

        def fn(src, _m=mlp, _r=rr):
            cands = _m.retrieve(src, top_n=POOL_N)
            return [w for w, _ in _r.rerank(src, cands, top_n=10)]

        rr_hits, _ = compute_hits(test_pairs, fn)
        results[name] = {
            "mlp": {str(k): v for k, v in mlp_hits.items()},
            "reranker": {str(k): v for k, v in rr_hits.items()},
            "pool100_recall": recall,
        }
        print(
            f"  MLP      @1={mlp_hits[1]:.1%} @5={mlp_hits[5]:.1%} @10={mlp_hits[10]:.1%}"
            f"  pool100={recall:.1%}"
        )
        print(
            f"  Reranker @1={rr_hits[1]:.1%} @5={rr_hits[5]:.1%} @10={rr_hits[10]:.1%}"
        )

    out = {"results": results, "elapsed_s": time.time() - t0}
    with open("results/exp_hard_negatives.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/exp_hard_negatives.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
