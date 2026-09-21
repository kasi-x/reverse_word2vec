"""
Leak-free antonym retrieval comparison.

Design (all test-pair information strictly isolated per fold):
  For each fold with TRAIN pairs / TEST pairs:
    1. Raw GloVe vectors (frozen, never modified).
    2. Counter-fit on TRAIN pairs only -> new vectors for this fold.
    3. Fit ICA on the fold's counter-fitted vectors (top vocab_limit words).
    4. Train every method using ONLY train pairs / fold-local space.
    5. Evaluate Hits@1/5/10 on TEST pairs (best-rank over both directions,
       same `compute_hits` protocol as the existing scripts).

Methods (all retrieval is over the same candidate pool: fold ICA vocab):
  - linear_raw : least-squares W mapping v(src)->v(tgt) on RAW GloVe + cosine search
  - linear_cf  : least-squares W on fold counter-fitted vectors + cosine search
  - negcos_cf  : nearest neighbour to -v(src) in fold CF space (cosine)
  - oracle_1ax : cheat baseline — flip single best axis (knows target)
  - mlp_ica    : MLP on ICA features, trained on fold train pairs (no reranker;
                 reranker training uses MLP predictions on train pairs, so a
                 leak-free reranker would halve its data — excluded here to keep
                 the comparison clean and focused)

Candidate pool: fold ICA vocab words (valid English, same filter as pipeline),
query word excluded.

Usage:
    pixi run python scripts/eval_leakfree.py [--folds 5] [--vocab-limit 50000]
"""

import argparse
import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import FastICA

from src.antonym_classifier import AntonymClassifier
from src.antonym_loader import extract_antonym_pairs
from src.counter_fitting import CounterFitConfig, CounterFitter
from src.eval_utils import unit_rows
from src.ica_transformer import ICASpace, is_valid_english_word
from src.semantic_operations import SemanticOperator
from src.word2vec_loader import Word2VecLoader

# --------------------------------------------------------------------------- #
# Fold-local space construction
# --------------------------------------------------------------------------- #


def fit_fold_space(
    model: KeyedVectors,
    train_pairs: list[tuple[str, str]],
    vocab_limit: int = 50000,
    n_components: int = 100,
    cf_iters: int = 50,
    verbose: bool = False,
) -> tuple[ICASpace, KeyedVectors]:
    """Counter-fit on TRAIN pairs only, then fit ICA. Returns (space, cf_model)."""
    cfg = CounterFitConfig(n_iter=cf_iters, verbose=verbose)
    cf_model = CounterFitter(cfg).fit(model, train_pairs)

    words, vectors = [], []
    for word in cf_model.index_to_key[:vocab_limit]:
        if is_valid_english_word(word):
            words.append(word)
            vectors.append(cf_model[word])
    X = np.array(vectors, dtype=np.float64)
    mean_vec = X.mean(axis=0)
    ica = FastICA(
        n_components=n_components,
        random_state=42,
        max_iter=1000,
        whiten="unit-variance",
    )
    S = ica.fit_transform(X - mean_vec)
    return (
        ICASpace(
            words=words,
            word_to_idx={w: i for i, w in enumerate(words)},
            S=S,
            mixing_matrix=ica.mixing_,
            unmixing_matrix=ica.components_,
            mean_vector=mean_vec,
            n_components=n_components,
        ),
        cf_model,
    )


# --------------------------------------------------------------------------- #
# Retrieval methods
# --------------------------------------------------------------------------- #


def topn_excluding(scores: np.ndarray, words: list[str], query: str, top_n: int):
    order = np.argsort(scores)[::-1]
    out = []
    for i in order:
        if words[i] == query:
            continue
        out.append(words[i])
        if len(out) >= top_n:
            break
    return out


class LinearAntonymMap:
    """Least-squares W: v(src) -> v(tgt) on train pairs; retrieve by cosine."""

    def fit(self, pairs, get_vec):
        X, Y = [], []
        for w1, w2 in pairs:
            v1, v2 = get_vec(w1), get_vec(w2)
            if v1 is None or v2 is None:
                continue
            X += [v1, v2]
            Y += [v2, v1]
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        self.W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # (d, d)

    def retrieve(self, query, get_vec, cand_words, cand_mat_u, top_n=10):
        v = get_vec(query)
        if v is None:
            return []
        pred = v.astype(np.float64) @ self.W
        pred /= max(np.linalg.norm(pred), 1e-10)
        return topn_excluding(cand_mat_u @ pred, cand_words, query, top_n)


def eval_pairs(pairs, retrieve_fn, top_n=10):
    hits = {1: 0, 5: 0, 10: 0}
    for w1, w2 in pairs:
        best = None
        for src, tgt in [(w1, w2), (w2, w1)]:
            for j, w in enumerate(retrieve_fn(src)[:top_n]):
                if w == tgt:
                    r = j + 1
                    if best is None or r < best:
                        best = r
                    break
        if best is not None:
            for k in (1, 5, 10):
                if best <= k:
                    hits[k] += 1
    n = len(pairs)
    return {k: hits[k] / n for k in (1, 5, 10)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--vocab-limit", type=int, default=50000)
    ap.add_argument("--components", type=int, default=100)
    ap.add_argument("--cf-iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    print("Loading GloVe-100 (raw, frozen)...")
    model: KeyedVectors = Word2VecLoader().load_glove(100)

    all_pairs = [(a, b) for a, b in extract_antonym_pairs() if a in model and b in model]
    print(f"Pairs with both words in GloVe-100: {len(all_pairs)}")

    # Restrict to words that survive the ICA vocab filter at least potentially:
    # valid-english check is vocab-dependent per fold; filter lazily per fold.
    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(all_pairs))
    rng.shuffle(idx)
    folds = np.array_split(idx, args.folds)

    per_fold: dict[str, list[dict]] = {
        "linear_raw": [],
        "linear_cf": [],
        "negcos_cf": [],
        "oracle_1ax": [],
        "mlp_ica": [],
    }
    fold_sizes = []

    for fi in range(args.folds):
        test_idx = set(folds[fi].tolist())
        train_pairs = [all_pairs[i] for i in range(len(all_pairs)) if i not in test_idx]
        test_pairs = [all_pairs[i] for i in folds[fi]]
        print(
            f"\n{'=' * 60}\nFold {fi + 1}/{args.folds}: "
            f"{len(train_pairs)} train / {len(test_pairs)} test"
        )

        # Fold-local vectors + ICA (train pairs only in CF constraints)
        space, cf_model = fit_fold_space(
            model,
            train_pairs,
            args.vocab_limit,
            args.components,
            args.cf_iters,
        )
        in_vocab = [
            p for p in test_pairs if p[0] in space.word_to_idx and p[1] in space.word_to_idx
        ]
        print(
            f"  fold vocab: {len(space.words)} words, "
            f"test pairs in vocab: {len(in_vocab)}/{len(test_pairs)}"
        )
        fold_sizes.append(len(in_vocab))

        raw_vec = {w: model[w].astype(np.float64) for w in space.words}
        cf_vec = {w: cf_model[w].astype(np.float64) for w in space.words}
        cand_u_raw = unit_rows(np.array([raw_vec[w] for w in space.words]))
        cand_u_cf = unit_rows(np.array([cf_vec[w] for w in space.words]))

        # --- linear_raw ---
        lin_raw = LinearAntonymMap()
        lin_raw.fit(train_pairs, raw_vec.get)

        def fn(q, _m=lin_raw, _v=raw_vec.get, _w=space.words, _u=cand_u_raw):
            return _m.retrieve(q, _v, _w, _u)

        per_fold["linear_raw"].append(eval_pairs(in_vocab, fn))

        # --- linear_cf ---
        lin_cf = LinearAntonymMap()
        lin_cf.fit(train_pairs, cf_vec.get)

        def fn(q, _m=lin_cf, _v=cf_vec.get, _w=space.words, _u=cand_u_cf):
            return _m.retrieve(q, _v, _w, _u)

        per_fold["linear_cf"].append(eval_pairs(in_vocab, fn))

        # --- negcos_cf: NN to -v(src) in CF space ---
        def fn_negcos(q, _cf=cf_vec, _U=cand_u_cf, _W=space.words):
            v = _cf.get(q)
            if v is None:
                return []
            n = v / max(np.linalg.norm(v), 1e-10)
            return topn_excluding(_U @ (-n), _W, q, 10)

        per_fold["negcos_cf"].append(eval_pairs(in_vocab, fn_negcos))

        # --- oracle_1ax (cheat): best axis from true pair, reconstruct, NN ---
        operator = SemanticOperator(cf_model, space, None)
        axis_std = np.maximum(np.std(space.S, axis=0), 1e-8)

        hits = {1: 0, 5: 0, 10: 0}
        for w1, w2 in in_vocab:
            s1, s2 = space.score(w1), space.score(w2)
            k = int(np.argmax(np.abs(s1 - s2) / axis_std))
            best = None
            for src, tgt in [(w1, w2), (w2, w1)]:
                for j, nb in enumerate(operator.axis_invert(src, k, top_n=10)):
                    if nb.word == tgt:
                        r = j + 1
                        if best is None or r < best:
                            best = r
                        break
            if best is not None:
                for kk in (1, 5, 10):
                    if best <= kk:
                        hits[kk] += 1
        per_fold["oracle_1ax"].append({k: hits[k] / len(in_vocab) for k in (1, 5, 10)})

        # --- mlp_ica: existing classifier, trained on fold train pairs ---
        mlp = AntonymClassifier(space, "mlp")
        mlp.fit(
            [p for p in train_pairs if p[0] in space.word_to_idx and p[1] in space.word_to_idx],
            rng=np.random.RandomState(args.seed + fi),
        )

        def fn(q, _m=mlp):
            return [w for w, _ in _m.retrieve(q, top_n=10)]

        per_fold["mlp_ica"].append(eval_pairs(in_vocab, fn))

        for name in per_fold:
            m = per_fold[name][-1]
            print(f"  {name:12s} @1={m[1]:.3f} @5={m[5]:.3f} @10={m[10]:.3f}")

    # Aggregate
    print(
        f"\n{'=' * 60}\nLeak-free {args.folds}-fold summary "
        f"(CF+ICA refit per fold on train pairs only):"
    )
    print(f"{'Method':12s}  {'Hits@1':>13s}  {'Hits@5':>13s}  {'Hits@10':>13s}")
    summary = {}
    for name, ms in per_fold.items():
        row = {}
        for k in (1, 5, 10):
            vals = [m[k] for m in ms]
            row[str(k)] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        summary[name] = row
        print(
            f"{name:12s}  "
            f"{row['1']['mean']:.3f}±{row['1']['std']:.3f}  "
            f"{row['5']['mean']:.3f}±{row['5']['std']:.3f}  "
            f"{row['10']['mean']:.3f}±{row['10']['std']:.3f}"
        )

    out = {
        "config": {
            "folds": args.folds,
            "vocab_limit": args.vocab_limit,
            "components": args.components,
            "cf_iters": args.cf_iters,
            "seed": args.seed,
        },
        "fold_test_sizes": fold_sizes,
        "summary": summary,
        "elapsed_s": time.time() - t0,
    }
    with open("results/leakfree_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/leakfree_comparison.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
