"""Fetchers pour séries macro (Yahoo chart API + cache local)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from src.common.utils import get_logger

LOG = get_logger(__name__)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CIBForecast/1.0; +https://localhost)",
}


@dataclass(frozen=True)
class FetchResult:
    indicator: str
    symbol: str
    source: str
    daily: pd.DataFrame
    from_cache: bool


def _cache_path(cache_dir: Path, symbol: str) -> Path:
    safe = symbol.replace("^", "idx_").replace("=", "_eq_").replace("/", "_")
    return cache_dir / f"{safe}_daily.csv"


def _read_cache(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.is_file():
        return None
    df = pd.read_csv(cache_path, parse_dates=["date"])
    if df.empty:
        return None
    return df


def _write_cache(cache_path: Path, df: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    meta = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(df),
    }
    cache_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def fetch_yahoo_daily(
    symbol: str,
    *,
    range_: str = "5y",
    cache_dir: Path | None = None,
    cache_max_age_hours: float = 24.0,
    retry_sleep_sec: float = 2.0,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Télécharge une série OHLC journalière via l'API chart Yahoo."""
    cache_path = _cache_path(cache_dir, symbol) if cache_dir else None

    if cache_path is not None:
        cached = _read_cache(cache_path)
        if cached is not None and cache_max_age_hours > 0:
            age_h = (time.time() - cache_path.stat().st_mtime) / 3600.0
            if age_h <= cache_max_age_hours:
                LOG.info("[fetch] cache hit %s (%.1fh)", symbol, age_h)
                return cached

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                YAHOO_CHART_URL.format(symbol=quote(symbol, safe="")),
                params={"interval": "1d", "range": range_},
                headers=DEFAULT_HEADERS,
                timeout=45,
            )
            if resp.status_code == 429:
                raise RuntimeError(f"Yahoo rate limit (429) for {symbol}")
            resp.raise_for_status()
            payload: dict[str, Any] = resp.json()
            result = payload["chart"]["result"][0]
            timestamps = result["timestamp"]
            closes = result["indicators"]["quote"][0]["close"]
            rows = [
                {"date": pd.Timestamp.utcfromtimestamp(ts).tz_localize(None), "value": close}
                for ts, close in zip(timestamps, closes, strict=False)
                if close is not None
            ]
            if not rows:
                raise RuntimeError(f"No close prices for {symbol}")
            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            if cache_path is not None:
                _write_cache(cache_path, df)
            return df
        except Exception as exc:
            last_error = exc
            LOG.warning("[fetch] attempt %d/%d failed for %s: %s", attempt, max_retries, symbol, exc)
            if attempt < max_retries:
                time.sleep(retry_sleep_sec * attempt)

    if cache_path is not None:
        cached = _read_cache(cache_path)
        if cached is not None:
            LOG.warning("[fetch] using stale cache for %s after failures", symbol)
            return cached

    raise RuntimeError(f"Failed to fetch {symbol}: {last_error}") from last_error


def fetch_indicator_daily(
    indicator: str,
    symbol: str,
    *,
    cache_dir: Path,
    range_: str = "5y",
    fallback_symbols: tuple[str, ...] = (),
) -> FetchResult:
    """Fetch avec symboles de repli ; indique si cache utilisé."""
    symbols = (symbol, *fallback_symbols)
    last_exc: Exception | None = None
    for idx, sym in enumerate(symbols):
        try:
            daily = fetch_yahoo_daily(sym, range_=range_, cache_dir=cache_dir)
            source = "yahoo"
            if idx > 0:
                source = f"yahoo_fallback:{sym}"
                LOG.warning("[fetch] %s via fallback symbol %s", indicator, sym)
            return FetchResult(
                indicator=indicator,
                symbol=sym,
                source=source,
                daily=daily,
                from_cache=False,
            )
        except Exception as exc:
            last_exc = exc
            LOG.warning("[fetch] %s symbol %s failed: %s", indicator, sym, exc)

    cache_path = _cache_path(cache_dir, symbol)
    cached = _read_cache(cache_path)
    if cached is not None:
        return FetchResult(
            indicator=indicator,
            symbol=symbol,
            source="cache_stale",
            daily=cached,
            from_cache=True,
        )
    raise RuntimeError(f"All sources failed for {indicator}: {last_exc}") from last_exc
