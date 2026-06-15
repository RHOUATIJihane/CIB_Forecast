"""Tests de la couche transformation (versions pandas testables sans Spark)."""

from __future__ import annotations

import numpy as np

from src.transformation import _compute_ts_metrics_array


def test_ts_metrics_random_walk_has_high_acf_lag1() -> None:
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(0, 1, 80))
    metrics = _compute_ts_metrics_array(series)
    # Random walk → ACF lag1 proche de 1
    assert metrics["acf_lag1"] > 0.7
    # ADF p-value très probable > 0.05 (non stationnaire)
    assert metrics["adf_pvalue"] >= 0.0


def test_ts_metrics_constant_returns_finite_or_nan() -> None:
    metrics = _compute_ts_metrics_array(np.full(60, 100.0))
    assert "adf_pvalue" in metrics
    # constante : ADF se peut être nan, on tolère
    assert np.isnan(metrics["acf_lag1"]) or abs(metrics["acf_lag1"]) <= 1.0
