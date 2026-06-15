"""
CIB forecasting — experiment suite v6.
Loaded by final-cib.ipynb after synthetic data generation (Bloc A).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, LGBMRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf as sm_acf

# ── Constants ───────────────────────────────────────────────────

CALIBRATION_CONDITION_PREFIXES = (
    "A_clean", "B_noisy", "C_short",
    "A_no_break", "B_amplified_w70", "C_level_shift_w70",
    "regular", "irregular",
    "A_perfect", "B_lag1w", "C_lag3w", "D_noisy30", "E_missing20",
    "sparse", "volatile", "flat", "irregular_normal",
)

VALIDATION_CONDITIONS = frozenset({
    "correct_external", "wrong_external", "random_external",
    "strict", "moderate", "loose",
})

RUN_CLASSIFICATION = False  # False = régression seule (plus rapide)

SELECTION_METHOD = {
    "sarima": "correlation",
    "holt_winters": "correlation",
    "ridge": "union",
    "lgbm": "importance",
    "rf": "importance",
}

SARIMA_TRAIN_MIN_WEEKS = 60

ALL_EXTERNAL_KEYS = (
    "oil_price_z", "commodity_index_z", "masi_index_z", "realestate_index_z",
    "is_ramadan", "is_eid_alfitr", "is_payroll_week", "is_month_end_week", "is_quarter_end",
)


# ── Dataclasses ─────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    name: str
    condition: str
    model_type: str
    model_family: str
    model_name: str
    rmse: float
    mae: float
    mape: float
    r2: float
    n_test_obs: int
    sector: str = ""
    account_type: str = ""
    acf1: float = 0.0
    cv: float = 0.0
    trend_strength: float = 0.0
    seasonality_strength: float = 0.0
    n_weeks: int = 0
    account_category: str = ""
    account_category_basis: str = ""
    experiment_role: str = ""
    n_externals_universe: int = 0
    n_externals_selected: int = 0
    externals_selected: str = ""
    externals_removed: str = ""
    selection_method: str = ""
    selection_corr_keys: str = ""
    selection_imp_keys: str = ""
    selection_both_keys: str = ""
    sarima_seasonal_active: int = 0
    notes: str = ""


@dataclass
class ClassificationResult:
    name: str
    condition: str
    model_type: str
    model_name: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    n_test_obs: int
    sector: str = ""
    account_category: str = ""
    notes: str = ""


# ── External universe ───────────────────────────────────────────

def build_external_universe(cal: pd.DataFrame, ind: pd.DataFrame, n_weeks: int) -> dict:
    ci = cal.set_index("week")
    return {
        "oil_price_z": ind["oil_price_z"].values[:n_weeks],
        "commodity_index_z": ind["commodity_index_z"].values[:n_weeks],
        "masi_index_z": ind["masi_index_z"].values[:n_weeks],
        "realestate_index_z": ind["realestate_index_z"].values[:n_weeks],
        "is_ramadan": ci["is_ramadan"].values[:n_weeks],
        "is_eid_alfitr": ci["is_eid_alfitr"].values[:n_weeks],
        "is_payroll_week": ci["is_payroll_week"].values[:n_weeks],
        "is_month_end_week": ci["is_month_end_week"].values[:n_weeks],
        "is_quarter_end": ci["is_quarter_end"].values[:n_weeks],
    }


def experiment_role_for_condition(condition: str) -> str:
    if condition in VALIDATION_CONDITIONS:
        return "validation"
    if condition in ("correct_external", "wrong_external", "random_external"):
        return "validation"
    return "calibration"


# ── Categorization (v7 encapsulé — aligné sur cib_forecast/src/account_categorization.py) ──

def regularity_score(acf1: float, cv: float) -> float:
    return float(max(0.0, min(1.0, 0.5 * acf1 + 0.5 * max(0.0, 1.0 - min(cv, 2.0) / 2.0))))


def categorize_account(
    acf1: float,
    cv: float,
    trend_strength: float,
    seasonality_strength: float,
    n_weeks: int,
) -> str:
    if n_weeks < 52:
        return "E_short_history"
    if cv > 1.0:
        return "D_sparse"
    if trend_strength < 0.02 and cv < 0.40:
        return "F_flat"
    if seasonality_strength > 0.60 and acf1 >= 0.30 and cv < 1.0 and trend_strength < 0.50:
        return "B_seasonal"
    if trend_strength >= 0.12:
        return "C_trending"
    reg = regularity_score(acf1, cv)
    if cv < 0.45 and reg >= 0.55:
        return "A_regular_stable"
    if cv >= 0.50 and trend_strength < 0.12:
        return "D_noisy"
    return "D_irregular"


def format_category_basis(
    acf1: float,
    cv: float,
    trend_strength: float,
    seasonality_strength: float,
    n_weeks: int,
) -> str:
    return (
        f"acf1={acf1:.2f},cv={cv:.2f},seas={seasonality_strength:.2f},"
        f"trend={trend_strength:.2f},n_weeks={n_weeks}"
    )


# ── Selection §2.4.5 ────────────────────────────────────────────

def select_externals_by_correlation(
    series: np.ndarray,
    external_universe: dict,
    alpha: float = 0.05,
    max_lag: int = 4,
) -> dict:
    selected = {}
    n = len(series)
    for name, ext_array in external_universe.items():
        ext = np.asarray(ext_array[:n], dtype=float)
        significant = False
        for lag in range(0, max_lag + 1):
            if lag == 0:
                x, y = ext, series
            else:
                x, y = ext[:-lag], series[lag:]
            if len(x) < 10:
                continue
            try:
                r, p = pearsonr(x, y)
                if p < alpha and abs(r) > 0.05:
                    significant = True
                    break
            except Exception:
                continue
        if significant:
            selected[name] = ext_array
    return selected


def select_externals_by_importance(
    series: np.ndarray,
    external_universe: dict,
    importance_threshold: float = 0.01,
    n_lags: int = 4,
) -> dict:
    n = len(series)
    if n < 20:
        return dict(external_universe)
    ext_keys = list(external_universe.keys())
    x_rows, y_rows = [], []
    for t in range(n_lags, n):
        lw = series[t - n_lags : t]
        row = list(lw) + [float(np.mean(lw)), float(np.std(lw))]
        row += [np.sin(2 * np.pi * (t % 52) / 52), np.cos(2 * np.pi * (t % 52) / 52)]
        for key in ext_keys:
            ext = external_universe[key]
            row.append(float(ext[t]))
            row.append(float(ext[t - 1]) if t > 0 else 0.0)
        x_rows.append(row)
        y_rows.append(series[t])
    x = np.array(x_rows)
    y = np.array(y_rows)
    n_train = int(len(y) * 0.80)
    if n_train < 10:
        return dict(external_universe)
    try:
        lgbm = LGBMRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, verbose=-1)
        lgbm.fit(x[:n_train], y[:n_train])
        importances = lgbm.feature_importances_
        total = importances.sum()
        if total == 0:
            return dict(external_universe)
        importances_norm = importances / total
        n_base = n_lags + 2 + 2
        selected = {}
        for k, key in enumerate(ext_keys):
            imp = importances_norm[n_base + k * 2] + importances_norm[n_base + k * 2 + 1]
            if imp >= importance_threshold:
                selected[key] = external_universe[key]
    except Exception:
        return dict(external_universe)
    if not selected:
        selected = {k: v for k, v in external_universe.items() if k.startswith("is_")}
    return selected


def select_externals_combined(
    series: np.ndarray,
    external_universe: dict,
    alpha: float = 0.05,
    importance_threshold: float = 0.01,
    method: str = "union",
) -> tuple[dict, dict]:
    corr_selected = select_externals_by_correlation(series, external_universe, alpha=alpha)
    imp_selected = select_externals_by_importance(
        series, external_universe, importance_threshold=importance_threshold
    )
    all_keys = set(external_universe.keys())
    corr_keys = set(corr_selected.keys())
    imp_keys = set(imp_selected.keys())
    if method == "union":
        final_keys = corr_keys | imp_keys
    elif method == "intersection":
        final_keys = corr_keys & imp_keys
    elif method == "importance":
        final_keys = imp_keys
    elif method == "correlation":
        final_keys = corr_keys
    else:
        final_keys = all_keys
    calendar_keys = {k for k in all_keys if k.startswith("is_")}
    final_keys = final_keys | calendar_keys
    selected = {k: external_universe[k] for k in final_keys if k in external_universe}
    report = {
        "n_universe": len(all_keys),
        "n_correlation": len(corr_keys),
        "n_importance": len(imp_keys),
        "n_selected": len(selected),
        "selected": sorted(selected.keys()),
        "removed": sorted(all_keys - final_keys),
        "corr_only": sorted(corr_keys - imp_keys),
        "imp_only": sorted(imp_keys - corr_keys),
        "both": sorted(corr_keys & imp_keys),
        "method": method,
    }
    return selected, report


# ── Metrics & features ──────────────────────────────────────────

def compute_account_metrics(series: np.ndarray) -> dict:
    from scipy import stats as scipy_stats

    m = {"acf1": 0.0, "cv": 0.0, "trend_strength": 0.0, "seasonality_strength": 0.0}
    try:
        m["acf1"] = float(sm_acf(series, nlags=1, fft=True)[1])
    except Exception:
        pass
    m["cv"] = float(np.std(series) / (np.mean(np.abs(series)) + 1e-8))
    try:
        _, _, r, _, _ = scipy_stats.linregress(np.arange(len(series)), series)
        m["trend_strength"] = float(r**2)
    except Exception:
        pass
    try:
        lag = 52 if len(series) > 52 else max(len(series) // 2, 1)
        corr = np.corrcoef(series[lag:], series[:-lag])[0, 1]
        m["seasonality_strength"] = float(abs(corr)) if np.isfinite(corr) else 0.0
    except Exception:
        pass
    return m


def mape_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def build_features(
    series: np.ndarray,
    external_series: Optional[dict] = None,
    n_lags: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(series)
    x_rows, y_rows = [], []
    ext_keys = list(external_series.keys()) if external_series else []
    for t in range(n_lags, n):
        lw = series[t - n_lags : t]
        row = list(lw) + [float(np.mean(lw)), float(np.std(lw))]
        wp = t % 52
        row += [np.sin(2 * np.pi * wp / 52), np.cos(2 * np.pi * wp / 52)]
        active_window = (np.abs(series[t - n_lags : t]) > 0).astype(float)
        row += [active_window.mean()]
        row += [float(np.max(np.abs(np.diff(series[t - n_lags : t])))) if n_lags > 1 else 0.0]
        for key in ext_keys:
            ext = external_series[key]
            row.append(float(ext[t]))
            row.append(float(ext[t - 1]) if t > 0 else 0.0)
        x_rows.append(row)
        y_rows.append(series[t])
    return np.array(x_rows), np.array(y_rows)


ML_MODELS = ["ridge", "lgbm", "rf"]
STAT_MODELS = ["sarima", "holt_winters"]
BENCHMARK_MODELS = STAT_MODELS + ML_MODELS
MODEL_FAMILY = {
    "sarima": "statistical",
    "holt_winters": "statistical",
    "ridge": "ml",
    "lgbm": "ml",
    "rf": "ml",
}

BENCHMARK_CLF_MODELS = ["logistic", "lgbm_clf"]

CLASSIFICATION_MODELS_CATALOG = {
    "logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegressionCV(cv=3, max_iter=500)),
    ]),
    "lgbm_clf": LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, verbose=-1),
}


def get_ml_model(name: str):
    if name == "ridge":
        return Pipeline([
            ("sc", StandardScaler()),
            ("m", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=3)),
        ])
    if name == "lgbm":
        return LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, verbose=-1)
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    raise ValueError(f"Modèle inconnu : {name}")


def train_eval_sarima(
    series: np.ndarray,
    exog: Optional[np.ndarray] = None,
    n_splits: int = 2,
    test_size: int = 4,
) -> tuple[Optional[dict], int]:
    n = len(series)
    all_rmse, all_mae, all_r2 = [], [], []
    last_pred = None
    seasonal_used = 0
    for split in range(n_splits):
        te = n - split * test_size
        ts = te - test_size
        if ts < 15:
            break
        yt = series[:ts]
        yv = series[ts:te]
        xt = exog[:ts] if exog is not None else None
        xv = exog[ts:te] if exog is not None else None
        s_order = (1, 0, 1, 52) if len(yt) >= SARIMA_TRAIN_MIN_WEEKS else (0, 0, 0, 0)
        if s_order[3] > 0:
            seasonal_used = 1
        try:
            fit = SARIMAX(
                yt, exog=xt, order=(1, 0, 1), seasonal_order=s_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            yp = fit.forecast(steps=test_size, exog=xv)
        except Exception:
            yp = np.full(test_size, np.mean(yt))
        all_rmse.append(np.sqrt(mean_squared_error(yv, yp)))
        all_mae.append(mean_absolute_error(yv, yp))
        all_r2.append(r2_manual(yv, yp))
        last_pred = yp
    if not all_rmse:
        return None, seasonal_used
    return {
        "rmse": np.mean(all_rmse),
        "mae": np.mean(all_mae),
        "mape": mape_score(series[-test_size:], last_pred) if last_pred is not None else np.nan,
        "r2": np.mean(all_r2),
        "y_test": series[-test_size:],
        "y_pred": last_pred if last_pred is not None else np.full(test_size, np.nan),
    }, seasonal_used


def train_eval_holt_winters(
    series: np.ndarray,
    exog: Optional[np.ndarray] = None,
    n_splits: int = 2,
    test_size: int = 4,
) -> Optional[dict]:
    n = len(series)
    all_rmse, all_mae, all_r2 = [], [], []
    last_pred = None
    for split in range(n_splits):
        te = n - split * test_size
        ts = te - test_size
        if ts < 15:
            break
        yt = series[:ts]
        yv = series[ts:te]
        try:
            hw = ExponentialSmoothing(
                yt, trend="add", seasonal=None, initialization_method="estimated"
            ).fit(optimized=True)
            hwp = hw.forecast(test_size)
            if exog is not None and len(exog) >= te:
                res = yt - hw.fittedvalues
                corrector = LGBMRegressor(
                    n_estimators=50, learning_rate=0.1, max_depth=3, verbose=-1
                )
                corrector.fit(exog[:ts], res)
                yp = hwp + corrector.predict(exog[ts:te])
            else:
                yp = hwp
        except Exception:
            yp = np.full(test_size, np.mean(yt))
        all_rmse.append(np.sqrt(mean_squared_error(yv, yp)))
        all_mae.append(mean_absolute_error(yv, yp))
        all_r2.append(r2_manual(yv, yp))
        last_pred = yp
    if not all_rmse:
        return None
    return {
        "rmse": np.mean(all_rmse),
        "mae": np.mean(all_mae),
        "mape": mape_score(series[-test_size:], last_pred) if last_pred is not None else np.nan,
        "r2": np.mean(all_r2),
        "y_test": series[-test_size:],
        "y_pred": last_pred if last_pred is not None else np.full(test_size, np.nan),
    }


def train_eval_ml(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_splits: int = 3,
    test_size: int = 4,
) -> Optional[dict]:
    n = len(y)
    all_rmse, all_mae, all_r2 = [], [], []
    last_model = None
    for split in range(n_splits):
        te = n - split * test_size
        ts = te - test_size
        if ts < 10:
            break
        xtr, xte = x[:ts], x[ts:te]
        ytr, yte = y[:ts], y[ts:te]
        m = get_ml_model(model_name)
        m.fit(xtr, ytr)
        yp = m.predict(xte)
        all_rmse.append(np.sqrt(mean_squared_error(yte, yp)))
        all_mae.append(mean_absolute_error(yte, yp))
        all_r2.append(r2_manual(yte, yp))
        last_model = m
    if not all_rmse or last_model is None:
        return None
    return {
        "rmse": np.mean(all_rmse),
        "mae": np.mean(all_mae),
        "mape": np.nan,
        "r2": np.mean(all_r2),
        "y_test": y[-test_size:],
        "y_pred": last_model.predict(x[-test_size:]),
    }


def train_eval_any(
    series: np.ndarray,
    model_name: str,
    external_series: Optional[dict] = None,
    n_splits: int = 3,
    test_size: int = 4,
    apply_selection: bool = True,
) -> tuple[Optional[dict], dict, int]:
    selection_report: dict = {
        "n_universe": 0,
        "n_selected": 0,
        "selected": [],
        "removed": [],
        "method": "none",
        "corr_only": [],
        "imp_only": [],
        "both": [],
    }
    sarima_seasonal = 0
    ext_used = external_series
    if external_series is not None and apply_selection:
        method = SELECTION_METHOD.get(model_name, "union")
        filtered, sel_report = select_externals_combined(
            series, external_series, method=method
        )
        selection_report = sel_report
        ext_used = filtered if filtered else None
    elif external_series:
        selection_report = {
            "n_universe": len(external_series),
            "n_selected": len(external_series),
            "selected": sorted(external_series.keys()),
            "removed": [],
            "method": "none (no selection)",
            "corr_only": [],
            "imp_only": [],
            "both": [],
        }

    if model_name == "sarima":
        exog = (
            np.column_stack([v[: len(series)] for v in ext_used.values()])
            if ext_used
            else None
        )
        res, sarima_seasonal = train_eval_sarima(
            series, exog=exog, n_splits=min(n_splits, 2), test_size=test_size
        )
        return res, selection_report, sarima_seasonal
    if model_name == "holt_winters":
        exog = (
            np.column_stack([v[: len(series)] for v in ext_used.values()])
            if ext_used
            else None
        )
        res = train_eval_holt_winters(
            series, exog=exog, n_splits=min(n_splits, 2), test_size=test_size
        )
        return res, selection_report, sarima_seasonal
    if model_name in ML_MODELS:
        x, y = build_features(series, external_series=ext_used)
        res = train_eval_ml(x, y, model_name, n_splits=n_splits, test_size=test_size)
        return res, selection_report, sarima_seasonal
    raise ValueError(f"Modèle inconnu : {model_name}")


def _result_row(
    acc: dict,
    condition_name: str,
    model_name: str,
    feature_mode: str,
    res: dict,
    metrics: dict,
    category: str,
    basis: str,
    role: str,
    sel_report: dict,
    sarima_seasonal: int,
) -> ExperimentResult:
    return ExperimentResult(
        name=acc["account_id"],
        condition=condition_name,
        model_type=feature_mode,
        model_family=MODEL_FAMILY.get(model_name, "unknown"),
        model_name=model_name,
        rmse=float(res["rmse"]),
        mae=float(res["mae"]),
        mape=float(res["mape"]) if not np.isnan(res.get("mape", np.nan)) else np.nan,
        r2=float(res["r2"]),
        n_test_obs=len(res["y_test"]),
        sector=acc.get("sector", ""),
        account_type=acc.get("account_type", ""),
        acf1=metrics["acf1"],
        cv=metrics["cv"],
        trend_strength=metrics["trend_strength"],
        seasonality_strength=metrics["seasonality_strength"],
        n_weeks=int(metrics.get("n_weeks", len(acc["series"]))),
        account_category=category,
        account_category_basis=basis,
        experiment_role=role,
        n_externals_universe=sel_report.get("n_universe", len(acc.get("externals") or {})),
        n_externals_selected=sel_report.get("n_selected", 0) if feature_mode == "with_externals" else 0,
        externals_selected=",".join(sel_report.get("selected", [])) if feature_mode == "with_externals" else "",
        externals_removed=",".join(sel_report.get("removed", [])) if feature_mode == "with_externals" else "",
        selection_method=str(sel_report.get("method", "")) if feature_mode == "with_externals" else "",
        selection_corr_keys=",".join(sel_report.get("corr_only", [])),
        selection_imp_keys=",".join(sel_report.get("imp_only", [])),
        selection_both_keys=",".join(sel_report.get("both", [])),
        sarima_seasonal_active=sarima_seasonal if model_name == "sarima" else 0,
        notes=acc.get("notes", ""),
    )


def run_comparison(
    accounts_data: list[dict],
    condition_name: str,
    models: list[str] | None = None,
) -> list[ExperimentResult]:
    models = models or BENCHMARK_MODELS
    results = []
    role = experiment_role_for_condition(condition_name)
    for acc in accounts_data:
        series = acc["series"]
        ext_univ = acc.get("externals") or {}
        metrics = compute_account_metrics(series)
        metrics["n_weeks"] = len(series)
        category = categorize_account(
            metrics["acf1"], metrics["cv"],
            metrics["trend_strength"], metrics["seasonality_strength"],
            metrics["n_weeks"],
        )
        basis = format_category_basis(
            metrics["acf1"], metrics["cv"],
            metrics["trend_strength"], metrics["seasonality_strength"],
            metrics["n_weeks"],
        )
        for model_name in models:
            rb, _, _ = train_eval_any(series, model_name, external_series=None, apply_selection=False)
            re, sel_report, sarima_seasonal = train_eval_any(
                series, model_name, external_series=ext_univ, apply_selection=True
            )
            if rb is None or re is None:
                continue
            for fmode, res, sr, seas in [
                ("baseline", rb, {}, 0),
                ("with_externals", re, sel_report, sarima_seasonal),
            ]:
                results.append(_result_row(
                    acc, condition_name, model_name, fmode, res,
                    metrics, category, basis, role, sr, seas,
                ))
    return results


def create_classification_labels(series: np.ndarray, threshold_pct: float = 0.05) -> np.ndarray:
    labels = np.zeros(len(series), dtype=int)
    for t in range(1, len(series)):
        denom = abs(series[t - 1]) + 1e-8
        variation = (series[t] - series[t - 1]) / denom
        if variation > threshold_pct:
            labels[t] = 1
        elif variation < -threshold_pct:
            labels[t] = -1
    return labels


def train_eval_classifier(
    x: np.ndarray,
    y_labels: np.ndarray,
    model_name: str,
    n_splits: int = 3,
    test_size: int = 4,
) -> Optional[dict]:
    n = len(y_labels)
    all_acc, all_f1_macro, all_f1_weighted = [], [], []
    last_model = None
    for split in range(n_splits):
        te = n - split * test_size
        ts = te - test_size
        if ts < 10:
            break
        xtr, xte = x[:ts], x[ts:te]
        ytr, yte = y_labels[:ts], y_labels[ts:te]
        if len(np.unique(ytr)) < 2:
            continue
        try:
            model = clone(CLASSIFICATION_MODELS_CATALOG[model_name])
            model.fit(xtr, ytr)
            y_pred = model.predict(xte)
        except (ValueError, IndexError):
            continue
        last_model = model
        all_acc.append(accuracy_score(yte, y_pred))
        all_f1_macro.append(f1_score(yte, y_pred, average="macro", zero_division=0))
        all_f1_weighted.append(f1_score(yte, y_pred, average="weighted", zero_division=0))
    if not all_acc or last_model is None:
        return None
    return {
        "accuracy": np.mean(all_acc),
        "f1_macro": np.mean(all_f1_macro),
        "f1_weighted": np.mean(all_f1_weighted),
        "y_test": y_labels[-test_size:],
        "y_pred": last_model.predict(x[-test_size:]),
    }


def run_comparison_classification(
    accounts_data: list[dict],
    condition_name: str,
    threshold_pct: float = 0.05,
    models: list[str] | None = None,
) -> list[ClassificationResult]:
    models = models or BENCHMARK_CLF_MODELS
    results = []
    n_lags = 4
    for acc in accounts_data:
        series = acc["series"]
        externals = acc.get("externals") or {}
        metrics = compute_account_metrics(series)
        category = categorize_account(
            metrics["acf1"], metrics["cv"],
            metrics["trend_strength"], metrics["seasonality_strength"],
            len(series),
        )
        y_labels = create_classification_labels(series, threshold_pct)
        y_clf = y_labels[n_lags:]
        x_base, _ = build_features(series, external_series=None)
        x_ext, _ = build_features(series, external_series=externals)
        for model_name in models:
            res_base = train_eval_classifier(x_base, y_clf, model_name)
            res_ext = train_eval_classifier(x_ext, y_clf, model_name)
            if res_base is None or res_ext is None:
                continue
            for feature_mode, res in [("baseline", res_base), ("with_externals", res_ext)]:
                results.append(ClassificationResult(
                    name=acc["account_id"],
                    condition=condition_name,
                    model_type=feature_mode,
                    model_name=model_name,
                    accuracy=res["accuracy"],
                    f1_macro=res["f1_macro"],
                    f1_weighted=res["f1_weighted"],
                    n_test_obs=len(res["y_test"]),
                    sector=acc.get("sector", ""),
                    account_category=category,
                    notes=acc.get("notes", ""),
                ))
    return results


def run_experiment_full(
    accounts_data: list[dict],
    condition_name: str,
    models_reg: list[str] | None = None,
    models_clf: list[str] | None = None,
) -> tuple[list[ExperimentResult], list[ClassificationResult]]:
    reg = run_comparison(accounts_data, condition_name, models_reg)
    if RUN_CLASSIFICATION:
        clf = run_comparison_classification(accounts_data, condition_name, models=models_clf)
    else:
        clf = []
    return reg, clf


# ── Experiment helpers ──────────────────────────────────────────

def _agg_series(df: pd.DataFrame, n_weeks: int) -> np.ndarray:
    return (
        df.groupby(pd.Grouper(key="date_operation", freq="W-MON"))["montant"]
        .sum()
        .values[:n_weeks]
    )


def _make_sector_data( acc_id, sector, acc_type, cal, ind, n_weeks, noise_frac, extra_notes=""):
    series_df = g["generate_account_series_v2"](
        acc_id, sector, acc_type, cal, ind, n_weeks=n_weeks, noise_fraction=noise_frac
    )
    series = _agg_series(series_df, n_weeks)
    ext = build_external_universe(cal, ind, n_weeks)
    notes = f"{sector}|{acc_type}|{extra_notes}" if extra_notes else f"{sector}|{acc_type}"
    return {
        "account_id": acc_id,
        "series": series,
        "sector": sector,
        "account_type": acc_type,
        "externals": {k: v[: len(series)] for k, v in ext.items()},
        "notes": notes,
    }


def experiment_1_signal_degradation(g, n_accounts_per_sector: int = 14):
    print("\n[Exp 1] Dégradation du signal — tous secteurs...")
    all_reg, all_clf = [], []
    for cond, n_weeks, noise_frac in [
        ("A_clean", 104, 0.25),
        ("B_noisy", 104, 0.60),
        ("C_short", 28, 0.40),
    ]:
        dates = pd.date_range("2022-01-03", periods=n_weeks * 7, freq="D")
        cal = g["generate_calendar_flags"](dates)
        ind = g["generate_sector_indicators"](dates, n_weeks)
        for sector in g["ALL_SECTORS"]:
            for i in range(n_accounts_per_sector):
                acc_id = f"EXP1_{cond}_{sector[:3].upper()}_{i:03d}"
                data = _make_sector_data(
                    g, acc_id, sector, "normal", cal, ind, n_weeks, noise_frac, extra_notes=cond
                )
                reg, clf = run_experiment_full([data], condition_name=cond)
                all_reg.extend(reg)
                all_clf.extend(clf)
    print(f"  Terminé — {len(all_reg)} reg, {len(all_clf)} clf")
    return all_reg, all_clf


def experiment_2_structural_breaks(g, n_accounts_per_sector: int = 14):
    print("\n[Exp 2] Ruptures structurelles...")
    n_weeks = 104
    dates = pd.date_range("2022-01-03", periods=n_weeks * 7, freq="D")
    cal = g["generate_calendar_flags"](dates)
    ind = g["generate_sector_indicators"](dates, n_weeks)
    cal_idx = cal.set_index("week")
    payroll = cal_idx["is_payroll_week"].values[:n_weeks]
    month_end = cal_idx["is_month_end_week"].values[:n_weeks]
    all_reg, all_clf = [], []
    for sector in g["ALL_SECTORS"]:
        beta6 = g["SECTOR_BETAS"][sector][6]
        beta7 = g["SECTOR_BETAS"][sector][7]
        cal_fx = beta6 * month_end + beta7 * payroll
        for i in range(n_accounts_per_sector):
            acc_id = f"EXP2_{sector[:3].upper()}_{i:03d}"
            df = g["generate_account_series_v2"](
                acc_id, sector, "normal", cal, ind, n_weeks=n_weeks, noise_fraction=0.35
            )
            s_base = _agg_series(df, n_weeks)
            lmean = np.mean(np.abs(s_base))
            s_amp = s_base.copy()
            s_amp[70:] += cal_fx[70:] * 2.0
            s_lvl = s_base.copy()
            s_lvl[70:] += lmean * 0.30
            ext = build_external_universe(cal, ind, n_weeks)
            for cond, series in [
                ("A_no_break", s_base),
                ("B_amplified_w70", s_amp),
                ("C_level_shift_w70", s_lvl),
            ]:
                data = {
                    "account_id": acc_id,
                    "series": series,
                    "sector": sector,
                    "account_type": "normal",
                    "externals": {k: v[: len(series)] for k, v in ext.items()},
                    "notes": cond,
                }
                reg, clf = run_experiment_full([data], condition_name=cond)
                all_reg.extend(reg)
                all_clf.extend(clf)
    print(f"  Terminé — {len(all_reg)} reg, {len(all_clf)} clf")
    return all_reg, all_clf


def experiment_3_regularity_segmentation(g, n_accounts: int = 120):
    print("\n[Exp 3] Segmentation régularité...")
    n_weeks = 104
    dates = pd.date_range("2022-01-03", periods=n_weeks * 7, freq="D")
    cal = g["generate_calendar_flags"](dates)
    ind = g["generate_sector_indicators"](dates, n_weeks)
    account_types = ["normal", "seasonal", "volatile", "trend_up"]
    noise_map = {"normal": 0.35, "seasonal": 0.25, "volatile": 0.60, "trend_up": 0.30}
    all_reg, all_clf = [], []
    np.random.seed(42)
    for i in range(n_accounts):
        acc_type = np.random.choice(account_types, p=[0.40, 0.25, 0.10, 0.25])
        sector = np.random.choice(g["ALL_SECTORS"])
        acc_id = f"EXP3_{acc_type[:3].upper()}_{sector[:3].upper()}_{i:03d}"
        data = _make_sector_data( acc_id, sector, acc_type, cal, ind, n_weeks, noise_map[acc_type])
        m = compute_account_metrics(data["series"])
        reg_score = float(np.clip(0.5 * m["acf1"] + 0.5 * max(0, 1 - min(m["cv"], 2) / 2), 0, 1))
        reg_label = "regular" if reg_score >= 0.55 else "irregular"
        data["notes"] = f"{acc_type}|{sector}|reg={reg_score:.3f}|{reg_label}"
        reg, clf = run_experiment_full([data], condition_name=reg_label)
        all_reg.extend(reg)
        all_clf.extend(clf)
    print(f"  Terminé — {len(all_reg)} reg, {len(all_clf)} clf")
    return all_reg, all_clf


def degrade_external(
    series: np.ndarray,
    lag_weeks: int = 0,
    noise_sigma_frac: float = 0.0,
    missing_frac: float = 0.0,
) -> np.ndarray:
    out = series.copy().astype(float)
    if lag_weeks > 0:
        out = np.concatenate([np.full(lag_weeks, np.nan), out[:-lag_weeks]])
    if noise_sigma_frac > 0:
        out += np.random.normal(0, np.nanstd(out) * noise_sigma_frac, len(out))
    if missing_frac > 0:
        idx = np.random.choice(len(out), size=int(len(out) * missing_frac), replace=False)
        out[idx] = np.nan
    return pd.Series(out).ffill().bfill().values


def experiment_4_external_quality(g, n_accounts_per_sector: int = 14):
    print("\n[Exp 4] Qualité des externals...")
    n_weeks = 104
    dates = pd.date_range("2022-01-03", periods=n_weeks * 7, freq="D")
    cal = g["generate_calendar_flags"](dates)
    ind = g["generate_sector_indicators"](dates, n_weeks)
    deg_conds = [
        ("A_perfect", dict(lag_weeks=0, noise_sigma_frac=0.00, missing_frac=0.00)),
        ("B_lag1w", dict(lag_weeks=1, noise_sigma_frac=0.00, missing_frac=0.00)),
        ("C_lag3w", dict(lag_weeks=3, noise_sigma_frac=0.00, missing_frac=0.00)),
        ("D_noisy30", dict(lag_weeks=0, noise_sigma_frac=0.30, missing_frac=0.00)),
        ("E_missing20", dict(lag_weeks=0, noise_sigma_frac=0.00, missing_frac=0.20)),
    ]
    all_reg, all_clf = [], []
    ext_univ = build_external_universe(cal, ind, n_weeks)
    for sector in g["ALL_SECTORS"]:
        for i in range(n_accounts_per_sector):
            acc_id = f"EXP4_{sector[:3].upper()}_{i:03d}"
            df = g["generate_account_series_v2"](
                acc_id, sector, "normal", cal, ind, n_weeks=n_weeks, noise_fraction=0.60
            )
            series = _agg_series(df, n_weeks)
            for cond, dparams in deg_conds:
                ext_deg = {}
                for k, v in ext_univ.items():
                    if k.startswith("is_"):
                        ext_deg[k] = v[: len(series)]
                    else:
                        ext_deg[k] = degrade_external(v[: len(series)], **dparams)
                data = {
                    "account_id": acc_id,
                    "series": series,
                    "sector": sector,
                    "account_type": "normal",
                    "externals": ext_deg,
                    "notes": f"{sector}|{cond}",
                }
                reg, clf = run_experiment_full([data], condition_name=cond)
                all_reg.extend(reg)
                all_clf.extend(clf)
    print(f"  Terminé — {len(all_reg)} reg, {len(all_clf)} clf")
    return all_reg, all_clf


def experiment_5_wrong_external(g, n_accounts_per_sector: int = 10):
    print("\n[Exp 5] Validation causalité (wrong external)...")
    n_weeks = 104
    dates = pd.date_range("2022-01-03", periods=n_weeks * 7, freq="D")
    cal = g["generate_calendar_flags"](dates)
    ind = g["generate_sector_indicators"](dates, n_weeks)
    cal_idx = cal.set_index("week")
    all_reg, all_clf = [], []
    builders = g["SECTOR_EXT_BUILDERS"]
    for sector in g["ALL_SECTORS"]:
        correct_ext = builders[sector](ind, cal_idx, n_weeks)
        wrong_sector = [s for s in g["ALL_SECTORS"] if s != sector][0]
        wrong_ext = builders[wrong_sector](ind, cal_idx, n_weeks)
        for i in range(n_accounts_per_sector):
            acc_id = f"EXP5_{sector[:3].upper()}_{i:03d}"
            df = g["generate_account_series_v2"](
                acc_id, sector, "normal", cal, ind, n_weeks=n_weeks, noise_fraction=0.35
            )
            series = _agg_series(df, n_weeks)
            for case_name, ext in [
                ("correct_external", {k: v[: len(series)] for k, v in correct_ext.items()}),
                ("wrong_external", {k: v[: len(series)] for k, v in wrong_ext.items()}),
                ("random_external", {"random": np.random.default_rng(i * 100).normal(0, 1, len(series))}),
            ]:
                data = {
                    "account_id": acc_id,
                    "series": series,
                    "sector": sector,
                    "externals": ext,
                    "notes": f"{sector}|{case_name}",
                }
                reg, clf = run_experiment_full([data], condition_name=case_name)
                all_reg.extend(reg)
                all_clf.extend(clf)
    print(f"  Terminé — {len(all_reg)} reg, {len(all_clf)} clf")
    return all_reg, all_clf


def experiment_6_irregular_accounts(g, n_accounts_per_cell: int = 4):
    print("\n[Exp 6] Comptes irréguliers...")
    n_weeks = 104
    dates = pd.date_range("2022-01-03", periods=n_weeks * 7, freq="D")
    cal = g["generate_calendar_flags"](dates)
    ind = g["generate_sector_indicators"](dates, n_weeks)
    profiles = {
        "sparse": {"noise_fraction": 0.70, "account_type": "sparse"},
        "volatile": {"noise_fraction": 0.60, "account_type": "volatile"},
        "flat": {"noise_fraction": 0.90, "account_type": "flat"},
        "irregular_normal": {"noise_fraction": 0.55, "account_type": "normal"},
    }
    all_reg, all_clf = [], []
    for profile_name, params in profiles.items():
        for sector in g["ALL_SECTORS"]:
            for i in range(n_accounts_per_cell):
                acc_id = f"EXP6_{profile_name[:3].upper()}_{sector[:3].upper()}_{i:03d}"
                data = _make_sector_data(
                    g, acc_id, sector, params["account_type"], cal, ind, n_weeks,
                    params["noise_fraction"], extra_notes=profile_name,
                )
                reg, clf = run_experiment_full([data], condition_name=profile_name)
                all_reg.extend(reg)
                all_clf.extend(clf)
    print(f"  Terminé — {len(all_reg)} reg, {len(all_clf)} clf")
    return all_reg, all_clf


def experiment_7_step7_ablation(g, n_accounts_per_sector: int = 14):
    print("\n[Exp 7] Validation Step 7 (ablation seuils)...")
    n_weeks = 104
    dates = pd.date_range("2022-01-03", periods=n_weeks * 7, freq="D")
    cal = g["generate_calendar_flags"](dates)
    ind = g["generate_sector_indicators"](dates, n_weeks)
    thresholds = {
        "strict": {"acf1_min": 0.35, "cv_min": 0.30, "cv_max": 1.50},
        "moderate": {"acf1_min": 0.20, "cv_min": 0.20, "cv_max": 2.00},
        "loose": {"acf1_min": 0.10, "cv_min": 0.10, "cv_max": 3.00},
    }
    all_reg, all_clf = [], []
    for sector in g["ALL_SECTORS"]:
        for i in range(n_accounts_per_sector):
            acc_type = "volatile" if i % 2 == 0 else "normal"
            acc_id = f"EXP7_{sector[:3].upper()}_{acc_type[:3].upper()}_{i:03d}"
            data = _make_sector_data(
                g, acc_id, sector, acc_type, cal, ind, n_weeks,
                0.60 if acc_type == "volatile" else 0.35,
            )
            m = compute_account_metrics(data["series"])
            for seuil_name, seuil in thresholds.items():
                passe = (
                    m["acf1"] > seuil["acf1_min"]
                    and seuil["cv_min"] < m["cv"] < seuil["cv_max"]
                )
                data_run = dict(data)
                data_run["notes"] = f"{seuil_name}|passe={passe}|type={acc_type}|{sector}"
                reg, clf = run_experiment_full([data_run], condition_name=seuil_name)
                all_reg.extend(reg)
                all_clf.extend(clf)
    print(f"  Terminé — {len(all_reg)} reg, {len(all_clf)} clf")
    return all_reg, all_clf


# ── Exports & aggregates ────────────────────────────────────────

def summarise_lift(results: list[ExperimentResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in results])
    idx = [
        c for c in [
            "name", "condition", "sector", "account_type", "account_category",
            "notes", "model_name", "model_family", "experiment_role",
            "acf1", "cv", "trend_strength", "seasonality_strength", "n_weeks",
            "n_externals_selected",
        ]
        if c in df.columns
    ]
    pivot = df.pivot_table(
        index=idx, columns="model_type", values=["rmse", "mae", "r2"], aggfunc="mean"
    ).reset_index()
    pivot.columns = ["_".join(str(c) for c in col).strip("_") for col in pivot.columns]
    if "rmse_baseline" in pivot.columns and "rmse_with_externals" in pivot.columns:
        pivot["rmse_lift_pct"] = (
            (pivot["rmse_baseline"] - pivot["rmse_with_externals"])
            / pivot["rmse_baseline"].replace(0, np.nan)
            * 100
        )
        pivot["mae_lift_pct"] = (
            (pivot["mae_baseline"] - pivot["mae_with_externals"])
            / pivot["mae_baseline"].replace(0, np.nan)
            * 100
        )
        pivot["r2_gain"] = pivot["r2_with_externals"] - pivot["r2_baseline"]
    return pivot


def summarise_clf_lift(results: list[ClassificationResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in results])
    pivot = df.pivot_table(
        index=["name", "condition", "sector", "account_category", "notes", "model_name"],
        columns="model_type",
        values=["accuracy", "f1_macro", "f1_weighted"],
        aggfunc="mean",
    ).reset_index()
    pivot.columns = ["_".join(str(c) for c in col).strip("_") for col in pivot.columns]
    if "accuracy_baseline" in pivot.columns and "accuracy_with_externals" in pivot.columns:
        pivot["accuracy_lift_pct"] = (
            (pivot["accuracy_with_externals"] - pivot["accuracy_baseline"])
            / (pivot["accuracy_baseline"] + 1e-9)
            * 100
        )
    return pivot


def build_category_benchmark(lift_df: pd.DataFrame, role: str | None = "calibration") -> pd.DataFrame:
    df = lift_df.copy()
    if role and "experiment_role" in df.columns:
        df = df[df["experiment_role"] == role]
    if df.empty or "account_category" not in df.columns:
        return pd.DataFrame()
    rows = []
    for (cat, model), g in df.groupby(["account_category", "model_name"], dropna=False):
        rows.append({
            "account_category": cat,
            "model_name": model,
            "model_family": g["model_family"].iloc[0] if "model_family" in g.columns else "",
            "n_accounts": g["name"].nunique() if "name" in g.columns else len(g),
            "rmse_baseline_mean": g["rmse_baseline"].mean(),
            "rmse_baseline_median": g["rmse_baseline"].median(),
            "rmse_with_externals_mean": g["rmse_with_externals"].mean(),
            "rmse_with_externals_median": g["rmse_with_externals"].median(),
            "rmse_lift_pct_mean": g["rmse_lift_pct"].mean(),
            "rmse_lift_pct_median": g["rmse_lift_pct"].median(),
            "mae_baseline_mean": g.get("mae_baseline", pd.Series(dtype=float)).mean(),
            "mae_with_externals_mean": g.get("mae_with_externals", pd.Series(dtype=float)).mean(),
            "n_externals_selected_mean": g["n_externals_selected"].mean()
            if "n_externals_selected" in g.columns else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["is_champion_category"] = (
        out.groupby("account_category")["rmse_with_externals_mean"].transform("min")
        == out["rmse_with_externals_mean"]
    ).astype(int)
    return out.sort_values(["account_category", "rmse_with_externals_mean"])


def build_category_benchmark_by_condition(lift_df: pd.DataFrame) -> pd.DataFrame:
    if lift_df.empty:
        return pd.DataFrame()
    rows = []
    for (cond, cat, model), g in lift_df.groupby(
        ["condition", "account_category", "model_name"], dropna=False
    ):
        rows.append({
            "condition": cond,
            "experiment_role": g["experiment_role"].iloc[0] if "experiment_role" in g.columns else "",
            "account_category": cat,
            "model_name": model,
            "n_accounts": g["name"].nunique() if "name" in g.columns else len(g),
            "rmse_baseline_mean": g["rmse_baseline"].mean(),
            "rmse_with_externals_mean": g["rmse_with_externals"].mean(),
            "rmse_lift_pct_mean": g["rmse_lift_pct"].mean(),
        })
    return pd.DataFrame(rows)


def build_selection_audit(results: list[ExperimentResult]) -> pd.DataFrame:
    ext_rows = [r for r in results if r.model_type == "with_externals"]
    if not ext_rows:
        return pd.DataFrame()
    records = []
    for r in ext_rows:
        selected = set(r.externals_selected.split(",")) if r.externals_selected else set()
        universe = set(ALL_EXTERNAL_KEYS)
        for ext_name in universe:
            records.append({
                "sector": r.sector,
                "model_name": r.model_name,
                "external_name": ext_name,
                "was_selected": int(ext_name in selected),
                "account_id": r.name,
            })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    agg = (
        df.groupby(["sector", "model_name", "external_name"], dropna=False)
        .agg(
            n_times_selected=("was_selected", "sum"),
            n_accounts=("account_id", "nunique"),
        )
        .reset_index()
    )
    agg["pct_selected"] = agg["n_times_selected"] / agg["n_accounts"].replace(0, np.nan) * 100
    return agg.sort_values(["sector", "model_name", "pct_selected"], ascending=[True, True, False])


def build_policy_v7_encapsulated(
    experiment_df: pd.DataFrame,
    *,
    min_lift_pct: float = 3.0,
) -> pd.DataFrame:
    """1 catégorie encapsulée = 1 EXP source = 1 décision (champion + externes)."""
    try:
        from src.category_policy_v7 import build_policy_v7_encapsulated as _build
    except ImportError:
        _build = None

    if _build is not None:
        return _build(experiment_df, min_lift_pct=min_lift_pct)

    # Repli autonome (notebook sans cib_forecast sur le path).
    ml_models = ("ridge", "rf", "lgbm")
    specs = [
        ("E_short_history", "E_short", "EXP1_C_short", "condition", "C_short"),
        ("D_noisy", "D_noisy", "EXP1_B_noisy", "condition", "B_noisy"),
        ("B_seasonal", "B_seasonal", "EXP3_seasonal", "account_type", "seasonal"),
        ("D_sparse", "D_sparse", "EXP6_sparse", "condition", "sparse"),
        ("F_flat", "F_flat", "EXP6_flat", "condition", "flat"),
        ("C_trending", "C_trending", "EXP2_C_level_shift", "condition", "C_level_shift_w70"),
        ("A_regular_stable", "A_regular", "EXP1_A_clean", "condition", "A_clean"),
        ("D_irregular", "D_irregular", "EXP3_irregular", "condition", "irregular"),
    ]
    df = experiment_df.copy()
    if "experiment_role" in df.columns:
        df = df[df["experiment_role"] == "calibration"]
    df = df[df["model_name"].isin(ml_models)]
    idx = ["name", "condition", "model_name"]
    base = df[df["model_type"] == "baseline"][idx + ["rmse"]].rename(columns={"rmse": "rmse_baseline"})
    ext = df[df["model_type"] == "with_externals"][idx + ["rmse"]].rename(
        columns={"rmse": "rmse_with_externals"}
    )
    merged = base.merge(ext, on=idx, how="inner")
    merged["rmse_lift_pct"] = (
        (merged["rmse_baseline"] - merged["rmse_with_externals"]) / merged["rmse_baseline"] * 100
    )
    rows: list[dict[str, object]] = []
    for cat, rule_id, source_exp, kind, value in specs:
        if kind == "condition":
            sub = merged[merged["condition"] == value]
        else:
            scope = df.loc[df["account_type"] == value, ["name", "condition"]].drop_duplicates()
            sub = merged.merge(scope, on=["name", "condition"], how="inner")
        n_accounts = int(sub["name"].nunique()) if not sub.empty else 0
        if sub.empty:
            continue
        stats = (
            sub.groupby("model_name", dropna=False)
            .agg(rmse_with_externals=("rmse_with_externals", "median"), lift_med=("rmse_lift_pct", "median"))
            .reset_index()
        )
        champ_row = stats.loc[stats["rmse_with_externals"].idxmin()]
        champion = str(champ_row["model_name"]).lower()
        lift_med = float(champ_row["lift_med"])
        rows.append(
            {
                "account_category": cat,
                "activate_externals": bool(lift_med >= min_lift_pct),
                "champion_model": champion,
                "policy_rule_id": rule_id,
                "source_exp": source_exp,
                "lift_med_champion_pct": round(lift_med, 2),
                "n_accounts_v6": n_accounts,
                "policy_source": "v7_encapsulated",
            }
        )
    return pd.DataFrame(rows)


def build_policy_draft(category_benchmark: pd.DataFrame, min_lift_pct: float = 3.0) -> pd.DataFrame:
    if category_benchmark.empty:
        return pd.DataFrame()
    rows = []
    for cat, g in category_benchmark.groupby("account_category"):
        champ = g.loc[g["is_champion_category"] == 1]
        if champ.empty:
            champ = g.loc[g["rmse_with_externals_mean"].idxmin()]
        else:
            champ = champ.iloc[0]
        ref = g[g["model_name"] == "lgbm"]
        lift_val = ref["rmse_lift_pct_mean"].iloc[0] if not ref.empty else champ["rmse_lift_pct_mean"]
        rows.append({
            "account_category": cat,
            "activate_externals_draft": bool(lift_val > min_lift_pct),
            "champion_model_draft": champ["model_name"],
            "evidence_lift_mean_pct": lift_val,
            "evidence_rmse_ext_mean": champ["rmse_with_externals_mean"],
            "min_lift_threshold_used": min_lift_pct,
            "status": "DRAFT — à valider manuellement",
        })
    return pd.DataFrame(rows)


def print_run_summary(all_reg: list[ExperimentResult], out_dir) -> None:
    sep = "═" * 70
    print(f"\n{sep}\nRÉSUMÉ v6 (chiffres bruts, sans verdict automatique)\n{sep}")
    df = pd.DataFrame([r.__dict__ for r in all_reg])
    print(f"  Évaluations régression : {len(df):,}")
    if "experiment_role" in df.columns:
        print(f"  Par rôle : {df['experiment_role'].value_counts().to_dict()}")
    if "account_category" in df.columns:
        print(f"  Par catégorie : {df['account_category'].value_counts().to_dict()}")
    print(f"  CSV → {out_dir}/")
    print(sep)


def save_v6_outputs(g, all_reg, all_clf, lift_wide: pd.DataFrame) -> None:
    out = g["OUT"]
    out.mkdir(parents=True, exist_ok=True)
    kaggle_working = g.get("KAGGLE_WORKING", out.parent)

    df_reg = pd.DataFrame([r.__dict__ for r in all_reg])
    df_reg.to_csv(out / "experiment_results_v6.csv", index=False)
    df_reg.to_csv(kaggle_working / "experiment_results_v6.csv", index=False)
    if all_clf:
        df_clf = pd.DataFrame([r.__dict__ for r in all_clf])
        df_clf.to_csv(out / "experiment_clf_results_v6.csv", index=False)
        df_clf.to_csv(kaggle_working / "experiment_clf_results_v6.csv", index=False)

    lift_wide.to_csv(out / "lift_summary_v6.csv", index=False)
    lift_wide.to_csv(kaggle_working / "lift_summary_v6.csv", index=False)

    cal = lift_wide[lift_wide["experiment_role"] == "calibration"] if "experiment_role" in lift_wide.columns else lift_wide
    cat_bench = build_category_benchmark(cal)
    cat_bench.to_csv(out / "category_benchmark_v6.csv", index=False)
    cat_bench.to_csv(kaggle_working / "category_benchmark_v6.csv", index=False)

    build_category_benchmark_by_condition(lift_wide).to_csv(
        out / "category_benchmark_by_condition_v6.csv", index=False
    )
    build_category_benchmark_by_condition(lift_wide).to_csv(
        kaggle_working / "category_benchmark_by_condition_v6.csv", index=False
    )

    build_selection_audit(all_reg).to_csv(out / "selection_audit_v6.csv", index=False)
    build_selection_audit(all_reg).to_csv(kaggle_working / "selection_audit_v6.csv", index=False)

    build_policy_draft(cat_bench).to_csv(out / "policy_draft_v6.csv", index=False)
    build_policy_draft(cat_bench).to_csv(kaggle_working / "policy_draft_v6.csv", index=False)

    policy_v7 = build_policy_v7_encapsulated(df_reg)
    policy_v7.to_csv(out / "policy_v7_encapsulated.csv", index=False)
    policy_v7.to_csv(kaggle_working / "policy_v7_encapsulated.csv", index=False)

    (
        lift_wide.groupby(["condition", "model_name", "experiment_role"], dropna=False)["rmse_lift_pct"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .to_csv(out / "lift_aggregate_v6.csv", index=False)
    )

    if "acf1" in lift_wide.columns:
        lw = lift_wide.copy()
        lw["acf1_bin"] = pd.cut(lw["acf1"], bins=[-1, 0.3, 0.6, 1.1], labels=["acf1_low", "acf1_mid", "acf1_high"])
        lw["cv_bin"] = pd.cut(lw["cv"], bins=[0, 0.5, 1.0, 5], labels=["cv_stable", "cv_moderate", "cv_volatile"])
        lw["seas_bin"] = pd.cut(
            lw["seasonality_strength"], bins=[-0.1, 0.3, 0.6, 1.1],
            labels=["seas_low", "seas_mid", "seas_high"],
        )
        lw.to_csv(out / "lift_with_bins_v6.csv", index=False)
        lw.to_csv(kaggle_working / "lift_with_bins_v6.csv", index=False)

    df_e7 = df_reg[df_reg["condition"].isin(["strict", "moderate", "loose"])].copy()
    if "notes" in df_e7.columns:
        pat = (
            r"(?P<step7_profile>strict|moderate|loose)\|passe=(?P<passe_step7>True|False)\|"
            r"type=(?P<acc_type>[^|]+)"
        )
        df_e7 = df_e7.join(df_e7["notes"].str.extract(pat))
    df_e7.to_csv(out / "exp7_step7_ablation_parsed.csv", index=False)

    # Legacy aliases for dashboard fallback
    df_reg.to_csv(out / "experiment_reg_results_v3.csv", index=False)
    lift_wide.to_csv(out / "lift_by_account_condition_model.csv", index=False)


def register_experiment_suite(g: dict[str, Any]) -> None:
    """Expose v6 symbols in notebook globals."""
    exports = {
        "ExperimentResult": ExperimentResult,
        "ClassificationResult": ClassificationResult,
        "build_external_universe": build_external_universe,
        "categorize_account": categorize_account,
        "format_category_basis": format_category_basis,
        "compute_account_metrics": compute_account_metrics,
        "select_externals_combined": select_externals_combined,
        "BENCHMARK_MODELS": BENCHMARK_MODELS,
        "BENCHMARK_CLF_MODELS": BENCHMARK_CLF_MODELS,
        "run_comparison": run_comparison,
        "run_comparison_classification": run_comparison_classification,
        "run_experiment_full": run_experiment_full,
        "summarise_lift": summarise_lift,
        "summarise_clf_lift": summarise_clf_lift,
        "build_category_benchmark": build_category_benchmark,
        "build_selection_audit": build_selection_audit,
        "experiment_1_signal_degradation": lambda n=14: experiment_1_signal_degradation(g, n),
        "experiment_2_structural_breaks": lambda n=14: experiment_2_structural_breaks(g, n),
        "experiment_3_regularity_segmentation": lambda n=120: experiment_3_regularity_segmentation(g, n),
        "experiment_4_external_quality": lambda n=14: experiment_4_external_quality(g, n),
        "experiment_5_wrong_external": lambda n=10: experiment_5_wrong_external(g, n),
        "experiment_6_irregular_accounts": lambda n=4: experiment_6_irregular_accounts(g, n),
        "experiment_7_step7_ablation": lambda n=14: experiment_7_step7_ablation(g, n),
        "print_run_summary": lambda ar: print_run_summary(ar, g["OUT"]),
        "save_v6_outputs": lambda ar, ac, lw: save_v6_outputs(g, ar, ac, lw),
    }
    g.update(exports)


def run_main(g: dict[str, Any], n_per_sector: int = 14, n_exp3: int = 120, n_exp6: int = 4) -> None:
    register_experiment_suite(g)
    g["KAGGLE_WORKING"] = g.get("KAGGLE_WORKING", g["OUT"].parent)

    sep = "═" * 70
    print(sep)
    print("CIB Cashflow Forecasting — Experiment Suite v6")
    print("Régression : SARIMA · Holt-Winters · Ridge · LGBM · RF")
    if RUN_CLASSIFICATION:
        print("Classification : Logistic · LGBM")
    else:
        print("Classification : désactivée (RUN_CLASSIFICATION=False)")
    print("Sélection externals §2.4.5 · Catégories A–F · CSV policy_draft (brouillon)")
    print(sep)

    exp1_reg, exp1_clf = experiment_1_signal_degradation(g, n_per_sector)
    exp2_reg, exp2_clf = experiment_2_structural_breaks(g, n_per_sector)
    exp3_reg, exp3_clf = experiment_3_regularity_segmentation(g, n_exp3)
    exp4_reg, exp4_clf = experiment_4_external_quality(g, n_per_sector)
    exp5_reg, exp5_clf = experiment_5_wrong_external(g, max(10, n_per_sector // 2))
    exp6_reg, exp6_clf = experiment_6_irregular_accounts(g, n_exp6)
    exp7_reg, exp7_clf = experiment_7_step7_ablation(g, n_per_sector)

    all_reg = exp1_reg + exp2_reg + exp3_reg + exp4_reg + exp5_reg + exp6_reg + exp7_reg
    all_clf = (
        exp1_clf + exp2_clf + exp3_clf + exp4_clf + exp5_clf + exp6_clf + exp7_clf
        if RUN_CLASSIFICATION
        else []
    )

    lift_wide = summarise_lift(all_reg)
    print_run_summary(all_reg, g["OUT"])
    save_v6_outputs(g, all_reg, all_clf, lift_wide)

    g["_v6_all_reg"] = all_reg
    g["_v6_all_clf"] = all_clf
    g["_v6_lift_wide"] = lift_wide

    if "generate_synthetic_dataset_v2" in g:
        np.random.seed(12345)
        _df_bronze, _cal_bronze, _ind_bronze = g["generate_synthetic_dataset_v2"](n_accounts=200)
        g["OUT"].mkdir(parents=True, exist_ok=True)
        _df_bronze.to_csv(g["OUT"] / "bronze_transactions_synthetic.csv", index=False)
        _cal_bronze.to_csv(g["OUT"] / "bronze_calendar_weekly_flags.csv", index=False)
        _ind_bronze.to_csv(g["OUT"] / "bronze_macro_indicators_weekly.csv", index=False)

    done_msg = f"\n[Done] {len(all_reg):,} régression"
    if RUN_CLASSIFICATION:
        done_msg += f" · {len(all_clf):,} classification"
    print(done_msg)
    print(f"  category_benchmark_v6 → {g['OUT']}/category_benchmark_v6.csv")
    print(f"  policy_draft_v6 (brouillon) → {g['OUT']}/policy_draft_v6.csv")
