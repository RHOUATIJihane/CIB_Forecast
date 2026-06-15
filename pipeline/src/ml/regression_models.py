"""Fabrique de modèles de régression alignée sur ``get_ml_model`` du notebook v6."""

from __future__ import annotations

from typing import Any, Final

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.utils import get_logger
from src.policy_rules import REGRESSION_MODEL

LOG = get_logger(__name__)

_ALIASES: Final[dict[str, str]] = {
    "random_forest": "rf",
    "rf": "rf",
    "ridge": "ridge",
    "lgbm": "lgbm",
}


def normalize_regression_model(name: str | None) -> str:
    """Normalise ``regression_model`` issu de ``account_policy``."""
    if not name:
        return normalize_regression_model(REGRESSION_MODEL)
    key = str(name).strip().lower()
    return _ALIASES.get(key, key)


def make_regressor(model_name: str | None = None) -> BaseEstimator:
    """Instancie le modèle champion pour la catégorie (ridge / lgbm / rf)."""
    key = normalize_regression_model(model_name)

    if key == "ridge":
        return Pipeline(
            [
                ("sc", StandardScaler()),
                ("m", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=3)),
            ]
        )
    if key == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            verbose=-1,
        )
    if key == "rf":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )

    LOG.warning("Modèle régression inconnu %r — repli sur rf", model_name)
    return make_regressor("rf")


def resolve_regression_model(rules: dict[str, Any]) -> str:
    """Lit ``regression_model`` depuis une ligne ``account_policy``."""
    return normalize_regression_model(rules.get("regression_model"))
