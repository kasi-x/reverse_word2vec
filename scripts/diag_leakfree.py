"""Diagnostic for leak-free results: bug vs. distribution shift.

Builds ONE fold (seed 42, fold 0) exactly like eval_leakfree.py, then checks:
  1. Fold-MLP retrieval on TRAIN pairs (wiring sanity: must be high).
  2. Mean cosine of train vs test pairs in fold CF space (shift mechanism).
  3. Word overlap between train pairs and test queries.
  4. Sample top-5 outputs for a few queries (mlp_ica vs linear_raw).
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np
from eval_leakfree import LinearAntonymMap, fit_fold_space

from src.antonym_classifier import AntonymClassifier
from src.antonym_loader import extract_antonym_pairs
from src.eval_utils import unit_rows
from src.word2vec_loader import Word2VecLoader

model = Word2VecLoader().load_glove(100)
all_pairs = [(a, b) for a, b in extract_antonym_pairs() if a in model and b in model]

rng = np.random.RandomState(42)
idx = np.arange(len(all_pairs))
rng.shuffle(idx)
folds = np.array_split(idx, 5)
test_idx = set(folds[0].tolist())
train_pairs = [all_pairs[i] for i in range(len(all_pairs)) if i not in test_idx]
test_pairs = [all_pairs[i] for i in folds[0]]

space, cf_model = fit_fold_space(model, train_pairs, 50000, 100, 50)
in_train = [p for p in train_pairs if p[0] in space.word_to_idx and p[1] in space.word_to_idx]
in_test = [p for p in test_pairs if p[0] in space.word_to_idx and p[1] in space.word_to_idx]
print(
    f"train in vocab: {len(in_train)}/{len(train_pairs)}, "
    f"test in vocab: {len(in_test)}/{len(test_pairs)}"
)


# 1. cosine shift in CF space
def meancos(pairs):
    sims = []
    for w1, w2 in pairs:
        v1 = cf_model[w1].astype(np.float64)
        v2 = cf_model[w2].astype(np.float64)
        sims.append(float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    return float(np.mean(sims))


print(f"mean cos in CF space: train={meancos(in_train):+.3f}  test={meancos(in_test):+.3f}")


# raw-space cosines for reference
def meancos_raw(pairs):
    sims = []
    for w1, w2 in pairs:
        v1 = model[w1].astype(np.float64)
        v2 = model[w2].astype(np.float64)
        sims.append(float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    return float(np.mean(sims))


print(
    f"mean cos in RAW space: train={meancos_raw(in_train):+.3f}  test={meancos_raw(in_test):+.3f}"
)

# 2. word overlap
train_words = {w for p in train_pairs for w in p}
overlap = sum(1 for p in in_test for q in p for _ in [0] if q in train_words)
print(
    f"test query-slots whose word appears in train pairs: "
    f"{overlap}/{2 * len(in_test)} = {overlap / (2 * len(in_test)):.1%}"
)

# 3. fold-MLP on train (sanity) and test
mlp = AntonymClassifier(space, "mlp")
mlp.fit(in_train, rng=np.random.RandomState(42))


def hits(pairs, retr):
    h = {1: 0, 5: 0, 10: 0}
    for w1, w2 in pairs:
        best = None
        for src, tgt in [(w1, w2), (w2, w1)]:
            for j, w in enumerate(retr(src)[:10]):
                if w == tgt:
                    r = j + 1
                    if best is None or r < best:
                        best = r
                    break
        if best is not None:
            for k in (1, 5, 10):
                if best <= k:
                    h[k] += 1
    return {k: h[k] / len(pairs) for k in (1, 5, 10)}


def retr(q):
    return [w for w, _ in mlp.retrieve(q, top_n=10)]


print("mlp_ica on TRAIN:", {k: round(v, 3) for k, v in hits(in_train[:300], retr).items()})
print("mlp_ica on TEST :", {k: round(v, 3) for k, v in hits(in_test, retr).items()})

# 4. samples
raw_vec = {w: model[w].astype(np.float64) for w in space.words}
cand_u = unit_rows(np.array([raw_vec[w] for w in space.words]))
lin = LinearAntonymMap()
lin.fit(train_pairs, raw_vec.get)
for q in ["hot", "good", "king", "happy"]:
    if q not in space.word_to_idx:
        continue
    print(f"\n[{q}]")
    print("  mlp_ica   :", [w for w, _ in mlp.retrieve(q, top_n=5)])
    print("  linear_raw:", lin.retrieve(q, raw_vec.get, space.words, cand_u, top_n=5))
