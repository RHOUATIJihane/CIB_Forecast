"""Tests transform macro scraper (sans réseau)."""

from __future__ import annotations

import pandas as pd

from src.macro_scraper.transform import align_weekly_series, resample_to_weekly_monday, zscore_series


def test_resample_to_weekly_monday_last_value() -> None:
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-08"]),
        "value": [10.0, 12.0, 20.0],
    })
    weekly = resample_to_weekly_monday(daily)
    assert len(weekly) == 2
    assert weekly.iloc[-1]["raw_value"] == 20.0


def test_zscore_zero_mean() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    z = zscore_series(s)
    assert abs(z.mean()) < 1e-9


def test_align_weekly_forward_fill() -> None:
    w1 = pd.DataFrame({"week": pd.to_datetime(["2024-01-01", "2024-01-08"]), "raw_value": [1.0, 3.0]})
    w2 = pd.DataFrame({"week": pd.to_datetime(["2024-01-08"]), "raw_value": [10.0]})
    aligned = align_weekly_series({"a": w1, "b": w2}, n_weeks=2, end_week=pd.Timestamp("2024-01-08"))
    assert list(aligned.columns) == ["week", "a", "b"]
    assert aligned["b"].notna().all()
    assert aligned["b"].iloc[-1] == 10.0
