"""Inférence batch : charge derniers modèles, scoring sur le dernier point disponible."""

from __future__ import annotations

from configparser import SectionProxy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.utils import get_logger, load_config
from src.ml.build_features import build_features
from src.ml.model_registry import latest_run_dir, load_model
from src.ml.regression_models import resolve_regression_model
from src.ml.train_runner import _externals_for_policy, _load_features, _load_policy

LOG = get_logger(__name__)


def _resolve_model(run_dir: Path | None, account_id: str, feature_mode: str) -> Path | None:
    if run_dir is None:
        return None
    candidate = run_dir / f"{account_id}__{feature_mode}.joblib"
    return candidate if candidate.exists() else None


def run_inference(env: str | None = None) -> pd.DataFrame:
    """Produit ``predictions_cib_forecast`` (CSV) à partir des derniers modèles entraînés."""
    cfg = load_config(env)
    features = _load_features(cfg)
    policy = _load_policy(cfg)

    if "week" in features.columns:
        features["week"] = pd.to_datetime(features["week"])
    features = features.sort_values(["numero_compte", "week"]).reset_index(drop=True)

    n_lags = int(cfg.get("ml_n_lags", fallback="4"))
    reg_dir = latest_run_dir(cfg["hdfs_models_regression"])
    clf_dir = latest_run_dir(cfg["hdfs_models_classification"])
    if reg_dir is None and clf_dir is None:
        raise FileNotFoundError("No trained model directory found. Run train.py first.")

    policy_map = policy.set_index("numero_compte").to_dict(orient="index")
    rows: list[dict[str, Any]] = []

    for account_id, group in features.groupby("numero_compte"):
        series = group["total_amount"].to_numpy(dtype=float)
        if len(series) < n_lags + 1:
            continue
        last_week = group["week"].iloc[-1]
        rules = policy_map.get(account_id, {})

        # Régression
        use_ext_reg = bool(rules.get("use_externals_regression", False))
        externals_reg = _externals_for_policy(group, rules, use_externals=use_ext_reg)
        feature_mode_reg = "with_externals" if use_ext_reg else "baseline"
        model_path = _resolve_model(reg_dir, account_id, feature_mode_reg)
        if model_path is not None:
            x_reg, _ = build_features(
                series,
                external_series=externals_reg,
                n_lags=n_lags,
            )
            if len(x_reg) > 0:
                model = load_model(str(model_path))
                pred = float(model.predict(x_reg[-1:])[0])
                rows.append({
                    "numero_compte": account_id,
                    "last_week": last_week,
                    "task": "regression",
                    "feature_mode": feature_mode_reg,
                    "model_name": resolve_regression_model(rules),
                    "prediction": pred,
                })

        # Classification
        use_ext_clf = bool(rules.get("use_externals_classification", False))
        feature_mode_clf = "with_externals" if use_ext_clf else "baseline"
        model_path = _resolve_model(clf_dir, account_id, feature_mode_clf)
        if model_path is not None:
            externals_clf = _externals_for_policy(group, rules, use_externals=use_ext_clf)
            x_clf, _ = build_features(
                series,
                external_series=externals_clf,
                n_lags=n_lags,
            )
            if len(x_clf) > 0:
                model = load_model(str(model_path))
                cls = int(model.predict(x_clf[-1:])[0])
                rows.append({
                    "numero_compte": account_id,
                    "last_week": last_week,
                    "task": "classification",
                    "feature_mode": feature_mode_clf,
                    "prediction": cls,
                })

    predictions = pd.DataFrame(rows)
    out_dir = Path(cfg["local_base"]) / "ml" / cfg["table_predictions"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.csv"
    predictions.to_csv(out_file, index=False)
    LOG.info("Predictions saved: %s (%d rows)", out_file, len(predictions))
    return predictions
