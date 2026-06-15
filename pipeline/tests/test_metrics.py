"""Tests des métriques régression / classification."""

from __future__ import annotations

import numpy as np

from src.ml.metrics import classification_scores, mae, mape, r2_manual, rmse


def test_rmse_mae_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0


def test_mape_skips_small_values() -> None:
    y_true = np.array([100.0, 0.0, 200.0])
    y_pred = np.array([110.0, 5.0, 220.0])
    # 0.0 est masqué (eps=1.0), seuls 100 et 200 comptent
    assert abs(mape(y_true, y_pred) - 10.0) < 1e-6


def test_r2_manual_perfect_and_zero() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert r2_manual(y, y) == 1.0
    # mean prediction → R2 = 0
    mean_pred = np.full_like(y, y.mean())
    assert abs(r2_manual(y, mean_pred)) < 1e-9


def test_classification_scores_keys() -> None:
    y = np.array([0, 1, -1, 1])
    yp = np.array([0, 1, -1, 0])
    scores = classification_scores(y, yp)
    assert set(scores.keys()) == {"accuracy", "f1_macro", "f1_weighted"}
    assert 0.0 <= scores["accuracy"] <= 1.0
