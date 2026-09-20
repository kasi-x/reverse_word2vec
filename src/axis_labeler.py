"""
Automatic axis interpretation and labeling for ICA-decomposed embedding spaces.

Uses kurtosis to identify non-Gaussian (structured) axes, then maps
WordNet antonym pairs to axes for semantic labeling.
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis

from src.ica_transformer import ICASpace


@dataclass
class AxisProfile:
    """Profile of a single ICA axis."""

    axis_idx: int
    positive_pole: list[str]  # top words on positive side
    negative_pole: list[str]  # top words on negative side
    label: str = ""  # e.g. "sentiment", "gender", "size"
    kurtosis: float = 0.0
    antonym_pairs: list[tuple[str, str]] = field(default_factory=list)


class AxisLabeler:
    """Profiles and labels ICA axes using distributional properties and antonym pairs."""

    def __init__(self, space: ICASpace):
        self.space = space

    def profile_all_axes(self, top_n: int = 20) -> list[AxisProfile]:
        """Generate profiles for all axes with top words and kurtosis."""
        profiles = []
        kurt_values = kurtosis(self.space.S, axis=0, fisher=True)

        for k in range(self.space.n_components):
            pos_words = self.space.top_words(k, n=top_n, positive=True)
            neg_words = self.space.top_words(k, n=top_n, positive=False)
            profile = AxisProfile(
                axis_idx=k,
                positive_pole=pos_words,
                negative_pole=neg_words,
                kurtosis=float(kurt_values[k]),
            )
            profiles.append(profile)
        return profiles

    def auto_label(
        self,
        antonym_pairs: list[tuple[str, str]],
        profiles: list[AxisProfile],
        min_pairs: int = 3,
    ) -> list[AxisProfile]:
        """
        Assign semantic labels to axes using WordNet antonym pairs.

        For each antonym pair, finds the axis with the largest score difference.
        Axes that attract many pairs get labeled by their most representative pair cluster.

        Args:
            antonym_pairs: List of (word1, word2) antonym pairs.
            profiles: Pre-computed axis profiles.
            min_pairs: Minimum pairs needed to label an axis.

        Returns:
            Updated profiles with labels and associated antonym pairs.
        """
        # Map each pair to its best axis
        axis_pairs: dict[int, list[tuple[str, str, float]]] = defaultdict(list)

        for w1, w2 in antonym_pairs:
            s1 = self.space.score(w1)
            s2 = self.space.score(w2)
            if s1 is None or s2 is None:
                continue
            diff = np.abs(s1 - s2)
            best_axis = int(np.argmax(diff))
            axis_pairs[best_axis].append((w1, w2, float(diff[best_axis])))

        # Predefined semantic categories with seed words
        CATEGORY_SEEDS = {
            "gender": {"male", "female", "man", "woman", "he", "she", "boy", "girl",
                        "king", "queen", "father", "mother", "husband", "wife"},
            "sentiment": {"good", "bad", "happy", "sad", "love", "hate", "nice", "nasty",
                          "pleasant", "unpleasant", "cheerful", "gloomy", "joy", "sorrow"},
            "size": {"big", "small", "large", "little", "huge", "tiny", "giant", "dwarf",
                     "wide", "narrow", "long", "short", "tall", "deep", "shallow"},
            "temperature": {"hot", "cold", "warm", "cool", "heat", "freeze", "boil", "chill"},
            "speed": {"fast", "slow", "quick", "gradual", "rapid", "sluggish", "swift"},
            "light": {"light", "dark", "bright", "dim", "luminous", "gloomy", "shine"},
            "age": {"young", "old", "new", "ancient", "modern", "youthful", "elderly"},
            "strength": {"strong", "weak", "powerful", "feeble", "mighty", "frail"},
            "wetness": {"wet", "dry", "moist", "arid", "damp", "parched"},
            "formality": {"formal", "informal", "polite", "rude", "casual", "official"},
            "morality": {"right", "wrong", "moral", "immoral", "virtue", "vice",
                         "honest", "dishonest", "true", "false"},
            "complexity": {"simple", "complex", "easy", "difficult", "hard", "plain"},
            "quantity": {"many", "few", "more", "less", "much", "little", "abundant", "scarce"},
            "direction": {"up", "down", "above", "below", "rise", "fall", "ascend", "descend",
                          "top", "bottom", "high", "low"},
            "openness": {"open", "closed", "public", "private", "free", "restricted"},
            "activity": {"active", "passive", "busy", "idle", "alive", "dead",
                         "awake", "asleep", "start", "stop", "begin", "end"},
        }

        for profile in profiles:
            k = profile.axis_idx
            pairs = axis_pairs.get(k, [])
            # Sort by score difference descending
            pairs.sort(key=lambda x: x[2], reverse=True)
            profile.antonym_pairs = [(w1, w2) for w1, w2, _ in pairs]

            if len(pairs) < min_pairs:
                profile.label = ""
                continue

            # Try to match category by seed word overlap
            pair_words = set()
            for w1, w2, _ in pairs:
                pair_words.add(w1)
                pair_words.add(w2)

            best_cat = ""
            best_overlap = 0
            for cat, seeds in CATEGORY_SEEDS.items():
                overlap = len(pair_words & seeds)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cat = cat

            if best_overlap >= 2:
                profile.label = best_cat
            else:
                # Fallback: label by the top antonym pair
                w1, w2, _ = pairs[0]
                profile.label = f"{w1}/{w2}"

        return profiles

    def print_summary(self, profiles: list[AxisProfile], top_n: int = 5) -> None:
        """Print a summary of axes sorted by kurtosis (most structured first)."""
        sorted_profiles = sorted(profiles, key=lambda p: p.kurtosis, reverse=True)

        print(f"\n{'='*80}")
        print(f"ICA Axis Summary ({self.space.n_components} components)")
        print(f"{'='*80}")

        for p in sorted_profiles[:30]:
            label_str = f"[{p.label}]" if p.label else "[unlabeled]"
            print(f"\nAxis {p.axis_idx:3d} {label_str:20s} kurtosis={p.kurtosis:7.2f}  "
                  f"({len(p.antonym_pairs)} antonym pairs)")
            print(f"  (+) {', '.join(p.positive_pole[:top_n])}")
            print(f"  (-) {', '.join(p.negative_pole[:top_n])}")

    def save_profiles(self, profiles: list[AxisProfile], path: str) -> None:
        """Save axis profiles to JSON."""
        data = []
        for p in profiles:
            data.append({
                "axis_idx": p.axis_idx,
                "label": p.label,
                "kurtosis": p.kurtosis,
                "positive_pole": p.positive_pole,
                "negative_pole": p.negative_pole,
                "antonym_pairs": p.antonym_pairs,
            })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(profiles)} axis profiles to {path}")

    def load_profiles(self, path: str) -> list[AxisProfile]:
        """Load axis profiles from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        profiles = []
        for d in data:
            profiles.append(AxisProfile(
                axis_idx=d["axis_idx"],
                label=d["label"],
                kurtosis=d["kurtosis"],
                positive_pole=d["positive_pole"],
                negative_pole=d["negative_pole"],
                antonym_pairs=[tuple(p) for p in d["antonym_pairs"]],
            ))
        return profiles
