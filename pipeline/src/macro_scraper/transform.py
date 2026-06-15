"""Transformations communes : daily → weekly, z-score."""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_series(values: pd.Series, *, min_periods: int = 8) -> pd.Series:
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def _monday_week(date: pd.Series | pd.Timestamp) -> pd.Series | pd.Timestamp:
    """Normalise vers le lundi de la semaine (W-MON pandas = fin de semaine)."""
    if isinstance(date, pd.Timestamp):
        ts = pd.Timestamp(date).normalize()
        return ts - pd.Timedelta(days=ts.weekday())
    ts = pd.to_datetime(date).dt.normalize()
    return ts - pd.to_timedelta(ts.dt.weekday, unit="D")


def resample_to_weekly_monday(daily: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Agrège une série journalière en semaines (lundi), dernière valeur connue."""
    if daily.empty:
        return pd.DataFrame(columns=["week", value_col])
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=[value_col])
    df["week"] = _monday_week(df["date"])
    weekly = df.groupby("week", as_index=False)[value_col].last()
    weekly["week"] = pd.to_datetime(weekly["week"])
    return weekly.rename(columns={value_col: "raw_value"})


def align_weekly_series(
    series_map: dict[str, pd.DataFrame],
    *,
    n_weeks: int | None = None,
    end_week: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aligne plusieurs séries hebdo sur un index commun (forward-fill)."""
    if not series_map:
        raise ValueError("series_map is empty")

    end = _monday_week(pd.Timestamp(end_week) if end_week is not None else max(
        df["week"].max() for df in series_map.values() if not df.empty
    ))

    min_start = _monday_week(min(df["week"].min() for df in series_map.values() if not df.empty))
    weeks = pd.date_range(start=min_start, end=end, freq="W-MON")
    if n_weeks is not None:
        weeks = weeks[-n_weeks:]

    out = pd.DataFrame({"week": weeks})
    for name, df in series_map.items():
        part = df[["week", "raw_value"]].copy().sort_values("week")
        part["week"] = pd.to_datetime(part["week"])
        filled = (
            out[["week"]]
            .merge(part, on="week", how="left")["raw_value"]
            .astype(float)
            .ffill()
            .bfill()
        )
        out[name] = filled.values

    return out


def apply_zscores(weekly_aligned: pd.DataFrame, indicator_cols: list[str]) -> pd.DataFrame:
    result = weekly_aligned[["week"]].copy()
    for col in indicator_cols:
        zcol = col if col.endswith("_z") else f"{col}_z"
        result[zcol] = zscore_series(weekly_aligned[col].astype(float))
    return result
