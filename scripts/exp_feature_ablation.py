"""
Experiment: ablate new reranker features and source parameters.

Variants (each = base pipeline + one change):
  base        : current 14-feature reranker, k=1 axis, aux=20
  dom_flip    : + dom_axis_flip feature (opposite sign on query's dominant axis)
  n_flipped   : + n_axes_flipped feature (count of large-sign-flip axes)
  both_feat   : + both new features
  knn5        : k=5 majority-vote axis predictor (instead of k=1)
  aux50       : aux_top_n=50 (deeper axis/proc/ica_map pools)
  all         : all changes combined

Same 70/15/15 split and seed as scripts/eval_reranker.py. The MLP and
candidate sources are fit once and shared; only the reranker feature
set / source params vary.

Usage:
    pixi run python scripts/exp_feature_ablation.py
"""

import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from src.antonym_classifier import AntonymClassifier
from src.antonym_loader import extract_antonym_pairs
from src.ica_transformer import load_ica_space
from src.reranker import CandidateSources, Reranker
from src.word2vec_loader import Word2VecLoader


class FlexReranker(Reranker):
    """Reranker with optional extra features."""

    def __init__(self, space, model, extra_feats=()):
        super().__init__(space, model)
        self.extra_feats = extra_feats
        self.N_FEATURES = 14 + len(extra_feats)

    def _features(
        self,
        query,
        candidate,
        mlp_rank,
        mlp_score,
        in_mlp=1,
        in_axis=0,
        in_proc=0,
        in_ica_map=0,
        in_morph=0,
    ):
        base = super()._features(
            query, candidate, mlp_rank, mlp_score, in_mlp, in_axis, in_proc, in_ica_map, in_morph
        )
        if not self.extra_feats:
            return base
        q_idx = self.space.word_to_idx.get(query)
        c_idx = self.space.word_to_idx.get(candidate)
        if q_idx is None or c_idx is None:
            return np.concatenate([base, np.zeros(len(self.extra_feats), dtype=np.float32)])
        s1 = self.space.S[q_idx].astype(np.float64)
        s2 = self.space.S[c_idx].astype(np.float64)
        z1 = s1 / self._axis_std
        z2 = s2 / self._axis_std
        extras = []
        for f in self.extra_feats:
            if f == "dom_axis_flip":
                dom = int(np.argmax(np.abs(z1)))
                # positive when candidate has opposite sign on dominant axis
                extras.append(float(-z1[dom] * z2[dom]))
            elif f == "n_axes_flipped":
                flipped = (np.sign(s1) != np.sign(s2)) & (np.abs(s1 - s2) > 2 * self._axis_std)
                extras.append(float(np.sum(flipped)))
        return np.concatenate([base, np.array(extras, dtype=np.float32)])


class FlexSources(CandidateSources):
    """CandidateSources with configurable k-NN axis predictor."""

    def __init__(self, space, model, knn_k=1):
        super().__init__(space, model)
        self.knn_k = knn_k

    def _predict_axis(self, query):
        q_idx = self.space.word_to_idx[query]
        sims = self._src_vecs @ self._S_norm[q_idx]
        if self.knn_k == 1:
            return self._src_axes[int(np.argmax(sims))]
        # majority vote over top-k
        top = np.argsort(sims)[::-1][: self.knn_k]
        votes = [self._src_axes[i] for i in top]
        return max(set(votes), key=votes.count)


def evaluate(reranker, sources, test_pairs, top_n=10):
    hits = {1: 0, 5: 0, 10: 0}
    n = 0
    for w1, w2 in test_pairs:
        for query, target in [(w1, w2), (w2, w1)]:
            pool = sources.pool(query)
            reranked = reranker.rerank(query, pool, top_n=top_n)
            words = [w for w, _ in reranked]
            for k in hits:
                if target in words[:k]:
                    hits[k] += 1
            n += 1
    return {k: v / n for k, v in hits.items()}


def main():
    t0 = time.time()
    print("Loading model and ICA space...")
    model = Word2VecLoader().load_glove(100)
    space = load_ica_space("results/ica_space.npz")

    antonym_pairs = extract_antonym_pairs()
    valid_pairs = [
        (w1, w2)
        for w1, w2 in antonym_pairs
        if space.score(w1) is not None and space.score(w2) is not None
    ]

    rng = np.random.RandomState(42)
    idx = np.arange(len(valid_pairs))
    rng.shuffle(idx)
    n = len(valid_pairs)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_pairs = [valid_pairs[i] for i in idx[:n_train]]
    val_pairs = [valid_pairs[i] for i in idx[n_train : n_train + n_val]]
    test_pairs = [valid_pairs[i] for i in idx[n_train + n_val :]]
    print(f"Split: {len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}")

    # ConceptNet augmentation (reranker only)
    import csv
    from pathlib import Path

    cn_pairs = []
    cn_path = Path("data/conceptnet_antonyms.tsv")
    if cn_path.exists():
        wn_set = {tuple(sorted(p)) for p in valid_pairs}
        seen = set()
        with open(cn_path) as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) < 2:
                    continue
                a, b = row[0].strip(), row[1].strip()
                key = tuple(sorted((a, b)))
                if (
                    a != b
                    and key not in seen
                    and key not in wn_set
                    and space.score(a) is not None
                    and space.score(b) is not None
                ):
                    seen.add(key)
                    cn_pairs.append((a, b))
    val_aug = val_pairs + cn_pairs[:500]
    print(f"CN augmentation: {len(cn_pairs)} pairs")

    print("\nTraining MLP...")
    mlp = AntonymClassifier(space, "mlp")
    mlp.fit(train_pairs, neg_ratio=3.0, rng=rng)

    variants = {
        "base": dict(extra_feats=(), knn_k=1, aux=20),
        "dom_flip": dict(extra_feats=("dom_axis_flip",), knn_k=1, aux=20),
        "n_flipped": dict(extra_feats=("n_axes_flipped",), knn_k=1, aux=20),
        "both_feat": dict(extra_feats=("dom_axis_flip", "n_axes_flipped"), knn_k=1, aux=20),
        "knn5": dict(extra_feats=(), knn_k=5, aux=20),
        "aux50": dict(extra_feats=(), knn_k=1, aux=50),
        "all": dict(extra_feats=("dom_axis_flip", "n_axes_flipped"), knn_k=5, aux=50),
    }

    results = {}
    for name, cfg in variants.items():
        print(f"\n=== {name}: feats={cfg['extra_feats']} knn={cfg['knn_k']} aux={cfg['aux']} ===")
        sources = FlexSources(space, model, knn_k=cfg["knn_k"])
        sources.fit(train_pairs, mlp, mlp_top_n=100, aux_top_n=cfg["aux"])
        reranker = FlexReranker(space, model, extra_feats=cfg["extra_feats"])
        reranker.fit(val_aug, mlp, top_n=100, sources=sources)
        hits = evaluate(reranker, sources, test_pairs)
        results[name] = hits
        print(f"  @1={hits[1]:.1%} @5={hits[5]:.1%} @10={hits[10]:.1%}")

    out = {
        "config": {"split": "70/15/15", "seed": 42, "cn_aug": len(cn_pairs)},
        "results": results,
        "runtime_s": time.time() - t0,
    }
    with open("results/exp_feature_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/exp_feature_ablation.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
