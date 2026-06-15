"""Régression par compte avec walk-forward validation (modèle selon politique)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator

from src.ml.metrics import mae, mape, r2_manual, rmse
from src.ml.regression_models import make_regressor


def walk_forward_eval(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_name: str | None = None,
    n_splits: int = 3,
    test_size: int = 4,
) -> Optional[dict[str, float | object]]:
    """3 fenêtres de test glissantes de 4 semaines (port du notebook)."""
    n = len(y)
    if n < 14:
        return None

    all_rmse, all_mae, all_r2 = [], [], []
    last_model: BaseEstimator | None = None

    for split in range(n_splits):
        test_end = n - split * test_size
        test_start = test_end - test_size
        if test_start < 10:
            break

        x_train, x_test = x[:test_start], x[test_start:test_end]
        y_train, y_test = y[:test_start], y[test_start:test_end]

        model = make_regressor(model_name)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        last_model = model

        all_rmse.append(rmse(y_test, y_pred))
        all_mae.append(mae(y_test, y_pred))
        all_r2.append(r2_manual(y_test, y_pred))

    if not all_rmse or last_model is None:
        return None

    last_pred = last_model.predict(x[-test_size:])
    return {
        "rmse": float(np.mean(all_rmse)),
        "mae": float(np.mean(all_mae)),
        "mape": mape(y[-test_size:], last_pred),
        "r2": float(np.mean(all_r2)),
        "model": last_model,
        "y_test": y[-test_size:],
        "y_pred": last_pred,
    }


def fit_final_regressor(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_name: str | None = None,
) -> BaseEstimator:
    """Modèle final entraîné sur 100% des données (pour scoring)."""
    model = make_regressor(model_name)
    model.fit(x, y)
    return model
