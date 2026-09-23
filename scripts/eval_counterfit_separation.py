"""Global counter-fitting separation check (raw GloVe-100, no ICA).

Measures mean antonym cosine before/after a global counter-fit over all
WordNet pairs (the leaky protocol, reported only to quantify the CF effect).
Headline retrieval never uses these vectors; see scripts/eval_leakfree.py
for the leak-free comparison.

Saves results/counterfit_separation.json, consumed by src/generate_paper.py.

Usage:
    pixi run python scripts/eval_counterfit_separation.py [--iters 100]
"""

import argparse
import json
import sys
import time

sys.path.insert(0, ".")

from src.antonym_loader import extract_antonym_pairs
from src.counter_fitting import CounterFitConfig, CounterFitter
from src.word2vec_loader import Word2VecLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--target-sim", type=float, default=-0.3)
    ap.add_argument("--lam", type=float, default=0.1)
    args = ap.parse_args()

    t0 = time.time()
    print("Loading GloVe-100...")
    model = Word2VecLoader().load_glove(100)
    pairs = extract_antonym_pairs()
    print(f"WordNet antonym pairs: {len(pairs)}")

    cfg = CounterFitConfig(
        n_iter=args.iters, target_sim=args.target_sim, lam=args.lam, verbose=True
    )
    fitter = CounterFitter(cfg)
    before = fitter.evaluate_antonym_separation(model, pairs)
    print(f"Before: mean={before['mean']:+.3f} over {before['n_pairs']} pairs")
    fitted = fitter.fit(model, pairs)
    after = fitter.evaluate_antonym_separation(fitted, pairs)
    print(f"After:  mean={after['mean']:+.3f} over {after['n_pairs']} pairs")

    out = {
        "config": {"n_iter": args.iters, "target_sim": args.target_sim, "lam": args.lam},
        "before": before,
        "after": after,
        "elapsed_s": time.time() - t0,
    }
    with open("results/counterfit_separation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/counterfit_separation.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
