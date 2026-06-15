"""Orchestration de l'entraînement : features ORC → modèles HDFS + métriques."""

from __future__ import annotations

from configparser import SectionProxy
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.utils import get_logger, load_config
from src.ml.build_features import build_features, create_classification_labels
from src.ml.model_registry import _utc_stamp, save_model
from src.sector_macro_mapping import CALENDAR_EXTERNAL_KEYS, MACRO_EXTERNAL_KEYS, parse_externals_for_model
from src.ml.train_classification import (
    fit_final_classifier,
    walk_forward_eval as walk_forward_clf,
)
from src.ml.regression_models import resolve_regression_model
from src.ml.train_regression import (
    fit_final_regressor,
    walk_forward_eval as walk_forward_reg,
)

LOG = get_logger(__name__)

EXTERNAL_KEYS = MACRO_EXTERNAL_KEYS + CALENDAR_EXTERNAL_KEYS


def _externals_dict(account_df: pd.DataFrame, keys: list[str] | None = None) -> dict[str, Any]:
    import numpy as np

    selected = keys if keys is not None else [k for k in EXTERNAL_KEYS if k in account_df.columns]
    out: dict[str, Any] = {}
    for k in selected:
        if k not in account_df.columns:
            continue
        arr = account_df[k].to_numpy(dtype=float)
        out[k] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _externals_for_policy(group: pd.DataFrame, rules: dict, *, use_externals: bool) -> dict[str, Any] | None:
    if not use_externals:
        return None
    keys = parse_externals_for_model(rules.get("externals_for_model"))
    if not keys:
        keys = [k for k in EXTERNAL_KEYS if k in group.columns]
    filtered = _externals_dict(group, keys)
    return filtered or None


def _load_features(cfg: SectionProxy) -> pd.DataFrame:
    """Charge features depuis Hive (défaut) ou CSV local (fallback tests)."""
    from src.common.table_loader import read_table_pandas

    if cfg.get("format", "hive").lower() == "hive":
        return read_table_pandas(cfg, layer="ml", table_key="table_features")
    local = Path(cfg["local_base"]) / "ml" / cfg["table_features"]
    if local.exists():
        return _read_dir_or_file(local)
    from src.common.table_loader import _csv_path

    return _read_dir_or_file(Path(_csv_path(cfg, layer="ml", table_key="table_features")))


def _read_dir_or_file(path: Path) -> pd.DataFrame:
    if path.is_dir():
        parts = sorted(path.glob("part-*.csv")) or sorted(path.glob("*.csv"))
        if not parts:
            raise FileNotFoundError(f"No CSV parts under {path}")
        return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    return pd.read_csv(path)


def _load_policy(cfg: SectionProxy) -> pd.DataFrame:
    from src.common.table_loader import read_table_pandas

    if cfg.get("format", "hive").lower() == "hive":
        return read_table_pandas(cfg, layer="silver", table_key="table_policy")
    local = Path(cfg["local_base"]) / "silver" / cfg["table_policy"]
    if local.exists():
        return _read_dir_or_file(local)
    from src.common.table_loader import _csv_path

    return _read_dir_or_file(Path(_csv_path(cfg, layer="silver", table_key="table_policy")))


def run_training(env: str | None = None) -> pd.DataFrame:
    """Entraîne régression (modèle selon ``account_policy``) et logistique par compte.

    Retourne un DataFrame de métriques (qui sera aussi sauvegardé en CSV).
    """
    cfg = load_config(env)
    features = _load_features(cfg)
    policy = _load_policy(cfg)
    run_stamp = _utc_stamp()

    if "week" in features.columns:
        features["week"] = pd.to_datetime(features["week"])
    features = features.sort_values(["numero_compte", "week"]).reset_index(drop=True)

    n_lags = int(cfg.get("ml_n_lags", fallback="4"))
    n_splits = int(cfg.get("ml_n_splits", fallback="3"))
    test_size = int(cfg.get("ml_test_size", fallback="4"))

    policy_map = policy.set_index("numero_compte").to_dict(orient="index")
    metrics_rows: list[dict[str, Any]] = []

    for account_id, group in features.groupby("numero_compte"):
        series = group["total_amount"].to_numpy(dtype=float)
        if len(series) < n_lags + 14:
            LOG.info("Skipping %s (only %d weeks)", account_id, len(series))
            continue
        rules = policy_map.get(account_id, {})
        use_ext_reg = bool(rules.get("use_externals_regression", False))
        use_ext_clf = bool(rules.get("use_externals_classification", False))
        reg_model = resolve_regression_model(rules)
        externals = _externals_for_policy(group, rules, use_externals=use_ext_reg)

        # Régression
        x_reg, y_reg = build_features(
            series,
            external_series=externals,
            n_lags=n_lags,
        )
        reg_eval = walk_forward_reg(
            x_reg,
            y_reg,
            model_name=reg_model,
            n_splits=n_splits,
            test_size=test_size,
        )
        if reg_eval is not None:
            final_reg = fit_final_regressor(x_reg, y_reg, model_name=reg_model)
            artifact = save_model(
                final_reg,
                account_id=account_id,
                task="regression",
                feature_mode="with_externals" if use_ext_reg else "baseline",
                base_uri=cfg["hdfs_models_regression"],
                run_stamp=run_stamp,
            )
            metrics_rows.append({
                "numero_compte": account_id,
                "task": "regression",
                "feature_mode": artifact.feature_mode,
                "model_name": reg_model,
                "rmse": reg_eval["rmse"],
                "mae": reg_eval["mae"],
                "mape": reg_eval["mape"],
                "r2": reg_eval["r2"],
                "n_test_obs": int(len(reg_eval["y_test"])),
                "model_path": artifact.path,
                "run_stamp": artifact.created_utc,
            })

        # Classification
        labels = create_classification_labels(series)[n_lags:]
        externals_clf = _externals_for_policy(group, rules, use_externals=use_ext_clf)
        x_clf, _ = build_features(
            series,
            external_series=externals_clf,
            n_lags=n_lags,
        )
        clf_eval = walk_forward_clf(x_clf, labels, n_splits=n_splits, test_size=test_size)
        if clf_eval is not None:
            final_clf = fit_final_classifier(x_clf, labels)
            artifact = save_model(
                final_clf,
                account_id=account_id,
                task="classification",
                feature_mode="with_externals" if use_ext_clf else "baseline",
                base_uri=cfg["hdfs_models_classification"],
                run_stamp=run_stamp,
            )
            metrics_rows.append({
                "numero_compte": account_id,
                "task": "classification",
                "feature_mode": artifact.feature_mode,
                "model_name": "logistic",
                "accuracy": clf_eval["accuracy"],
                "f1_macro": clf_eval["f1_macro"],
                "f1_weighted": clf_eval["f1_weighted"],
                "n_test_obs": int(len(clf_eval["y_test"])),
                "model_path": artifact.path,
                "run_stamp": artifact.created_utc,
            })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_out = Path(cfg["local_base"]) / "ml" / f"training_metrics_{run_stamp}.csv"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_out, index=False)
    LOG.info("Training metrics saved: %s (%d rows)", metrics_out, len(metrics_df))
    return metrics_df
