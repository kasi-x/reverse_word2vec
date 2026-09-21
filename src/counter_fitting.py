"""
Counter-fitting: inject antonym constraints into word vectors.

Based on Mrkšić et al. (2016) "Counter-fitting Word Vectors to Linguistic Constraints"
https://arxiv.org/abs/1603.00892

Key idea: GloVe encodes distributional similarity, so antonyms end up CLOSE
(hot/cold both appear in "temperature" contexts). Counter-fitting fixes this by
pushing antonym vectors apart while preserving the original embedding structure.

Objective:
  minimize C(V) = C_AR(V) + C_VSP(V)

  C_AR  = antonym repel:  push antonym pairs to cosine < target_sim
  C_VSP = space preserve: penalize drift from original vectors

Only words that appear in constraint pairs are updated.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from gensim.models import KeyedVectors


@dataclass
class CounterFitConfig:
    n_iter: int = 100  # optimisation iterations
    lr: float = 0.05  # learning rate
    lam: float = 0.1  # VSP regularisation weight (pull-back strength)
    target_sim: float = -0.3  # push antonyms below this cosine similarity
    verbose: bool = True


class CounterFitter:
    """
    Pushes antonym word vectors apart while preserving neighbourhood structure.

    Usage:
        fitter = CounterFitter(config)
        new_model = fitter.fit(model, antonym_pairs)
    """

    def __init__(self, config: CounterFitConfig | None = None):
        self.config = config or CounterFitConfig()

    def fit(
        self,
        model: KeyedVectors,
        antonym_pairs: list[tuple[str, str]],
        synonym_pairs: list[tuple[str, str]] | None = None,
    ) -> KeyedVectors:
        """
        Return a new KeyedVectors with counter-fitted vectors.

        Args:
            model: Original GloVe / word2vec model.
            antonym_pairs: Pairs to push apart.
            synonym_pairs: Optional pairs to pull together (not used by default).

        Returns:
            New KeyedVectors with modified vectors.
        """
        cfg = self.config

        # --- Collect words to update -----------------------------------------
        valid_pairs: list[tuple[int, int]] = []
        words_to_update: set[str] = set()
        for w1, w2 in antonym_pairs:
            if w1 in model and w2 in model:
                valid_pairs.append((model.key_to_index[w1], model.key_to_index[w2]))
                words_to_update.add(w1)
                words_to_update.add(w2)

        update_indices = sorted(model.key_to_index[w] for w in words_to_update)

        if cfg.verbose:
            print(
                f"Counter-fitting: {len(valid_pairs)} antonym pairs, "
                f"{len(update_indices)} words to update"
            )

        # --- Working copy of vectors (unit-normalised) -----------------------
        V = model.vectors.astype(np.float64).copy()
        # Normalise
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        V /= norms
        V_orig = V.copy()

        pair_arr = np.array(valid_pairs, dtype=np.int32)  # (n_pairs, 2)

        # --- Optimisation loop -----------------------------------------------
        for it in range(cfg.n_iter):
            # Antonym Repel: for each pair, if cos > target_sim, push apart
            i_idx = pair_arr[:, 0]
            j_idx = pair_arr[:, 1]

            vi = V[i_idx]  # (n_pairs, d)
            vj = V[j_idx]

            sims = np.sum(vi * vj, axis=1)  # (n_pairs,) cosine (vectors are unit)
            active = sims > cfg.target_sim  # pairs that still need pushing

            if active.any():
                s = sims[active, None]  # (k, 1)
                u = vi[active]  # (k, d)
                w = vj[active]  # (k, d)

                # Gradient of cosine w.r.t. u (normalised): ∂cos/∂u = w - cos·u
                grad_u = w - s * u  # (k, d)
                grad_w = u - s * w  # (k, d)

                # Gradient step to DECREASE cosine (subtract positive grad)
                np.add.at(V, i_idx[active], -cfg.lr * grad_u)
                np.add.at(V, j_idx[active], -cfg.lr * grad_w)

            # Vector Space Preservation: pull updated words back to original
            V[update_indices] = (1 - cfg.lam) * V[update_indices] + cfg.lam * V_orig[update_indices]

            # Re-normalise updated words
            sub = V[update_indices]
            sub_norms = np.linalg.norm(sub, axis=1, keepdims=True)
            V[update_indices] = sub / np.maximum(sub_norms, 1e-10)

            if cfg.verbose and (it + 1) % 20 == 0:
                # Report mean cosine of antonym pairs
                vi2 = V[i_idx]
                vj2 = V[j_idx]
                mean_sim = float(np.mean(np.sum(vi2 * vj2, axis=1)))
                n_still_close = int((np.sum(vi2 * vj2, axis=1) > cfg.target_sim).sum())
                print(
                    f"  iter {it + 1:4d}: mean antonym cos = {mean_sim:.3f}, "
                    f"pairs still above target = {n_still_close}/{len(valid_pairs)}"
                )

        # --- Build new KeyedVectors ------------------------------------------
        new_model = copy.deepcopy(model)
        # Scale back to original norm magnitude (preserves downstream scale)
        orig_norms = np.linalg.norm(model.vectors, axis=1, keepdims=True)
        new_model.vectors = (V * orig_norms).astype(np.float32)

        if cfg.verbose:
            # Final stats
            vi_f = V[i_idx]
            vj_f = V[j_idx]
            sims_orig = np.sum(
                (model.vectors[i_idx] / np.linalg.norm(model.vectors[i_idx], axis=1, keepdims=True))
                * (
                    model.vectors[j_idx]
                    / np.linalg.norm(model.vectors[j_idx], axis=1, keepdims=True)
                ),
                axis=1,
            )
            sims_new = np.sum(vi_f * vj_f, axis=1)
            print("\nAntonym cosine similarity:")
            print(
                f"  Before: mean={np.mean(sims_orig):.3f}, "
                f"median={np.median(sims_orig):.3f}, "
                f"% > 0 = {(sims_orig > 0).mean():.1%}"
            )
            print(
                f"  After:  mean={np.mean(sims_new):.3f}, "
                f"median={np.median(sims_new):.3f}, "
                f"% > 0 = {(sims_new > 0).mean():.1%}"
            )

        return new_model

    def evaluate_antonym_separation(
        self,
        model: KeyedVectors,
        antonym_pairs: list[tuple[str, str]],
    ) -> dict:
        """
        Report cosine similarity statistics for antonym pairs.
        """
        sims = []
        for w1, w2 in antonym_pairs:
            if w1 not in model or w2 not in model:
                continue
            v1 = model[w1] / max(np.linalg.norm(model[w1]), 1e-10)
            v2 = model[w2] / max(np.linalg.norm(model[w2]), 1e-10)
            sims.append(float(v1 @ v2))

        sims = np.array(sims)
        return {
            "n_pairs": len(sims),
            "mean": float(np.mean(sims)),
            "median": float(np.median(sims)),
            "pct_positive": float((sims > 0).mean()),
            "pct_below_minus05": float((sims < -0.5).mean()),
        }
