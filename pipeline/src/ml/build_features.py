"""Construction des matrices ``X`` / ``y`` par compte (port exact du notebook).

Utilisée par ``train.py`` / ``inference.py`` côté Python (sklearn) ; séparée
du pipeline Spark pour éviter de dépendre du driver pour chaque compte.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


FEATURE_NAMES_BASE: Sequence[str] = (
    "lag_1", "lag_2", "lag_3", "lag_4",
    "rolling_mean_4", "rolling_std_4",
    "week_sin", "week_cos",
)


def feature_columns(external_keys: Sequence[str] | None = None) -> list[str]:
    """Liste ordonnée des colonnes générées par ``build_features``."""
    cols = list(FEATURE_NAMES_BASE)
    for key in external_keys or []:
        cols.extend([f"{key}_t", f"{key}_t-1"])
    return cols


def build_features(
    series: np.ndarray,
    external_series: Optional[dict[str, np.ndarray]] = None,
    n_lags: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Réimplémentation 1-pour-1 du `build_features` du notebook.

    Baseline    : lags + rolling stats + encodage temporel sin/cos.
    With externals : baseline + (valeur ``t`` et ``t-1``) pour chaque clé externe.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    ext_keys = list(external_series.keys()) if external_series else []
    x_rows: list[list[float]] = []
    y_rows: list[float] = []

    for t in range(n_lags, n):
        row = list(series[t - n_lags : t])
        lag_window = series[t - n_lags : t]
        row += [float(np.mean(lag_window)), float(np.std(lag_window))]
        week_pos = t % 52
        row += [
            float(np.sin(2.0 * np.pi * week_pos / 52.0)),
            float(np.cos(2.0 * np.pi * week_pos / 52.0)),
        ]
        for key in ext_keys:
            ext = np.asarray(external_series[key], dtype=float)
            row.append(float(ext[t]))
            row.append(float(ext[t - 1] if t > 0 else 0.0))
        x_rows.append(row)
        y_rows.append(float(series[t]))

    if not x_rows:
        return np.zeros((0, 0)), np.zeros((0,))
    return np.array(x_rows, dtype=float), np.array(y_rows, dtype=float)


def create_classification_labels(
    series: np.ndarray,
    threshold_pct: float = 0.05,
) -> np.ndarray:
    """Étiquettes 3 classes (-1 / 0 / +1) sur la variation hebdo."""
    series = np.asarray(series, dtype=float)
    labels = np.zeros(len(series), dtype=int)
    for t in range(1, len(series)):
        denom = abs(series[t - 1]) + 1e-8
        variation = (series[t] - series[t - 1]) / denom
        if variation > threshold_pct:
            labels[t] = 1
        elif variation < -threshold_pct:
            labels[t] = -1
    return labels
