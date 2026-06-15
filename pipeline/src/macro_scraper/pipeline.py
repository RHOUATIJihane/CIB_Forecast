"""Pipeline scrape → macro_indicators_weekly (pandas)."""

from __future__ import annotations

import json
from configparser import SectionProxy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.utils import get_logger
from src.datagen.synthetic import generate_sector_indicators
from src.macro_scraper.fetchers import FetchResult, fetch_indicator_daily
from src.macro_scraper.transform import align_weekly_series, resample_to_weekly_monday, zscore_series

LOG = get_logger(__name__)

# indicator z-column → clé interne alignée
INDICATORS: dict[str, str] = {
    "oil_price_z": "oil_price",
    "commodity_index_z": "commodity_index",
    "masi_index_z": "masi_index",
    "realestate_index_z": "realestate_index",
}

DEFAULT_SYMBOLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "oil_price_z": ("BZ=F", ()),
    "commodity_index_z": ("DBC", ()),
    "masi_index_z": ("MASI.CS", ("^GSPC",)),
    "realestate_index_z": ("IYR", ()),
}


@dataclass
class ScrapeReport:
    sources: dict[str, str]
    n_weeks: int
    used_synthetic_fallback: bool


def _cache_dir(cfg: SectionProxy) -> Path:
    base = Path(cfg.get("local_base", "cib_forecast_data"))
    return base / cfg.get("macro_scrape_cache_dir", "bronze/raw/macro_cache")


def _parse_symbols(cfg: SectionProxy) -> dict[str, tuple[str, tuple[str, ...]]]:
    raw = cfg.get("macro_yahoo_symbols", "").strip()
    if not raw:
        return DEFAULT_SYMBOLS
    out: dict[str, tuple[str, tuple[str, ...]]] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        indicator, symbol_part = part.split(":", 1)
        if "|" in symbol_part:
            primary, fallbacks = symbol_part.split("|", 1)
            fb = tuple(s.strip() for s in fallbacks.split(";") if s.strip())
        else:
            primary, fb = symbol_part.strip(), ()
        out[indicator.strip()] = (primary, fb)
    return out or DEFAULT_SYMBOLS


def _synthetic_macro(n_weeks: int, end_date: str | None) -> pd.DataFrame:
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(weeks=n_weeks + 2)
    dates = pd.date_range(start=start, end=end, freq="D")
    return generate_sector_indicators(dates, n_weeks=n_weeks)


def scrape_macro_dataframe(
    cfg: SectionProxy,
    *,
    n_weeks: int | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, ScrapeReport]:
    n_weeks = n_weeks or int(cfg.get("macro_n_weeks", "104"))
    cache = _cache_dir(cfg)
    symbols = _parse_symbols(cfg)
    range_ = cfg.get("macro_yahoo_range", "5y")
    end_ts = pd.Timestamp(end_date) if end_date else None

    weekly_parts: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    failed: list[str] = []

    for zcol, raw_key in INDICATORS.items():
        primary, fallbacks = symbols.get(zcol, DEFAULT_SYMBOLS[zcol])
        try:
            result: FetchResult = fetch_indicator_daily(
                zcol,
                primary,
                cache_dir=cache,
                range_=range_,
                fallback_symbols=fallbacks,
            )
            weekly_parts[raw_key] = resample_to_weekly_monday(result.daily)
            tag = "cache_stale" if result.from_cache else result.source
            sources[zcol] = f"{tag}:{result.symbol}"
            LOG.info("[scrape] %s ok (%s)", zcol, sources[zcol])
        except Exception as exc:
            LOG.error("[scrape] %s failed: %s", zcol, exc)
            failed.append(zcol)

    used_fallback = False
    if failed:
        synth = _synthetic_macro(n_weeks, end_date)
        for zcol in failed:
            raw_key = INDICATORS[zcol]
            weekly_parts[raw_key] = synth[["week", zcol]].rename(columns={zcol: "raw_value"})
            sources[zcol] = "synthetic_fallback"
        used_fallback = True

    if not weekly_parts:
        synth = _synthetic_macro(n_weeks, end_date)
        return synth, ScrapeReport(
            sources={z: "synthetic_full" for z in INDICATORS},
            n_weeks=n_weeks,
            used_synthetic_fallback=True,
        )

    aligned = align_weekly_series(weekly_parts, n_weeks=n_weeks, end_week=end_ts)
    macro_df = aligned[["week"]].copy()
    for zcol, raw_key in INDICATORS.items():
        macro_df[zcol] = zscore_series(aligned[raw_key].astype(float))

    macro_df["week"] = pd.to_datetime(macro_df["week"])
    macro_df = macro_df.sort_values("week").reset_index(drop=True)

    report = ScrapeReport(
        sources=sources,
        n_weeks=len(macro_df),
        used_synthetic_fallback=used_fallback,
    )
    return macro_df, report


def run_scrape_macro(
    cfg: SectionProxy,
    *,
    n_weeks: int | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    df, report = scrape_macro_dataframe(cfg, n_weeks=n_weeks, end_date=end_date)
    meta_path = _cache_dir(cfg) / "last_scrape_report.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "sources": report.sources,
                "n_weeks": report.n_weeks,
                "used_synthetic_fallback": report.used_synthetic_fallback,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOG.info(
        "[scrape] done weeks=%d fallback=%s sources=%s",
        report.n_weeks,
        report.used_synthetic_fallback,
        report.sources,
    )
    return df
