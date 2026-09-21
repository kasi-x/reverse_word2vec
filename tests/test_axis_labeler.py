"""Tests for axis profiling and labeling."""

import numpy as np

from src.axis_labeler import AxisLabeler
from src.ica_transformer import ICASpace


def make_synthetic_space() -> ICASpace:
    """Hand-built score matrix: axis 2 separates the temperature pairs,
    axis 0 separates gender pairs; all pairs present in vocab."""
    words = [
        "hot",
        "cold",
        "warm",
        "cool",
        "king",
        "queen",
        "man",
        "woman",
        "up",
        "down",
        "filler1",
        "filler2",
    ]
    rng = np.random.RandomState(0)
    S = rng.normal(0, 0.1, (len(words), 6))

    def set_row(w, vals):
        S[words.index(w), : len(vals)] = vals

    # Temperature on axis 2
    set_row("hot", [0, 0, 3.0])
    set_row("cold", [0, 0, -3.0])
    set_row("warm", [0, 0, 2.5])
    set_row("cool", [0, 0, -2.5])
    # Gender on axis 0
    set_row("king", [3.0])
    set_row("queen", [-3.0])
    set_row("man", [2.8])
    set_row("woman", [-2.8])
    set_row("up", [0, 2.0])
    set_row("down", [0, -2.0])

    d = 4
    S_full = np.hstack([S, rng.normal(0, 0.1, (len(words), d - 1))])
    mixing = np.eye(d)
    unmixing = np.eye(d)
    mean = np.zeros(d)
    word_to_idx = {w: i for i, w in enumerate(words)}
    return ICASpace(
        words=words,
        word_to_idx=word_to_idx,
        S=S_full,
        mixing_matrix=mixing,
        unmixing_matrix=unmixing,
        mean_vector=mean,
        n_components=d,
    )


def test_auto_label_assigns_pairs_to_largest_diff_axis():
    space = make_synthetic_space()
    labeler = AxisLabeler(space)
    profiles = labeler.profile_all_axes()

    pairs = [("hot", "cold"), ("warm", "cool"), ("king", "queen"), ("man", "woman"), ("up", "down")]
    labeled = labeler.auto_label(pairs, profiles, min_pairs=2)

    # Axis 2 must hold the temperature pairs
    axis2 = labeled[2]
    assert ("hot", "cold") in axis2.antonym_pairs
    assert ("warm", "cool") in axis2.antonym_pairs
    # Axis 0 must hold the gender pairs
    assert ("king", "queen") in labeled[0].antonym_pairs


def test_labels_reflect_seed_categories():
    space = make_synthetic_space()
    labeler = AxisLabeler(space)
    profiles = labeler.profile_all_axes()
    pairs = [("hot", "cold"), ("warm", "cool"), ("boil", "freeze")]
    labeled = labeler.auto_label(pairs, profiles, min_pairs=2)
    assert labeled[2].label == "temperature"


def test_unsupported_axes_stay_unlabeled():
    space = make_synthetic_space()
    labeler = AxisLabeler(space)
    profiles = labeler.profile_all_axes()
    labeled = labeler.auto_label([("hot", "cold")], profiles, min_pairs=3)
    assert labeled[2].label == ""
    assert labeled[2].antonym_pairs == [("hot", "cold")]


def test_save_and_load_profiles_roundtrip(tmp_path):
    space = make_synthetic_space()
    labeler = AxisLabeler(space)
    profiles = labeler.profile_all_axes()
    profiles = labeler.auto_label(
        [("hot", "cold"), ("warm", "cool"), ("king", "queen"), ("man", "woman")], profiles
    )
    path = str(tmp_path / "profiles.json")
    labeler.save_profiles(profiles, path)
    loaded = labeler.load_profiles(path)

    assert len(loaded) == len(profiles)
    assert [p.axis_idx for p in loaded] == [p.axis_idx for p in profiles]
    for a, b in zip(loaded, profiles, strict=False):
        assert a.label == b.label
        assert a.antonym_pairs == b.antonym_pairs
