"""Tests de build_features / create_classification_labels."""

from __future__ import annotations

import numpy as np

from src.ml.build_features import build_features, create_classification_labels


def test_build_features_baseline_shapes() -> None:
    series = np.arange(20.0)
    x, y = build_features(series, external_series=None, n_lags=4)
    assert x.shape == (16, 8)  # 4 lags + mean + std + sin + cos
    assert y.shape == (16,)
    # lags first 4 columns of row 0 == series[0..3]
    np.testing.assert_allclose(x[0, :4], series[:4])
    assert y[0] == series[4]


def test_build_features_with_externals_adds_two_cols_per_key() -> None:
    series = np.arange(20.0)
    externals = {"oil": np.linspace(-1.0, 1.0, 20)}
    x, _ = build_features(series, external_series=externals, n_lags=4)
    # baseline = 8 cols, externals add 2 cols per key
    assert x.shape == (16, 10)


def test_classification_labels_three_classes() -> None:
    series = np.array([100, 110, 105, 90, 90, 90], dtype=float)
    labels = create_classification_labels(series, threshold_pct=0.05)
    # t=0 reste 0, t=1: +10% → +1, t=2: -5%≈0, t=3: ~-14% → -1, t=4/5: 0%
    assert labels[0] == 0
    assert labels[1] == 1
    assert labels[3] == -1
    assert labels[4] == 0
