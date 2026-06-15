"""Scrape les indicateurs macro et charge ``cib_bronze.macro_indicators_weekly``."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from pyspark.sql import functions as F

from src.common.table_writer import write_table_spark
from src.common.utils import get_logger, get_spark, load_config
from src.macro_scraper.pipeline import run_scrape_macro

LOG = get_logger(__name__)


def _pandas_to_spark(spark, pdf, table_key: str):
    pdf = pdf.copy()
    pdf["week"] = pdf["week"].astype(str)
    fd, path = tempfile.mkstemp(suffix=".csv", prefix=f"cib_scrape_{table_key}_")
    os.close(fd)
    try:
        pdf.to_csv(path, index=False)
        sdf = spark.read.option("header", "true").option("inferSchema", "true").csv(f"file://{path}")
        sdf = sdf.withColumn("week", F.to_timestamp("week"))
        sdf = sdf.cache()
        sdf.count()
        return sdf
    finally:
        os.unlink(path)


def run_scrape_to_bronze(*, n_weeks: int | None = None, end_date: str | None = None) -> int:
    cfg = load_config()
    if cfg.get("format", "hive").lower() != "hive":
        LOG.error("format=hive requis dans config.ini")
        return 1

    macro_pdf = run_scrape_macro(cfg, n_weeks=n_weeks, end_date=end_date)
    LOG.info("[scrape_macro] writing %d rows to bronze", len(macro_pdf))

    spark = get_spark("cib-scrape-macro", enable_hive=True, cfg=cfg)
    try:
        sdf = _pandas_to_spark(spark, macro_pdf, "macro")
        write_table_spark(sdf, cfg, layer="bronze", table_key="table_bronze_macro", mode="overwrite")
    finally:
        spark.stop()

    LOG.info("[scrape_macro] → cib_bronze.%s", cfg["table_bronze_macro"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape macro indicators → bronze Hive table.")
    parser.add_argument("--n-weeks", type=int, default=None)
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--local-csv",
        action="store_true",
        help="Écrit un CSV local sous local_base/bronze/ (utile sans HDFS).",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Avec --local-csv : n'écrit pas sur Hive/HDFS.",
    )
    args = parser.parse_args()

    if args.local_csv:
        cfg = load_config()
        macro_pdf = run_scrape_macro(cfg, n_weeks=args.n_weeks, end_date=args.end_date)
        out = Path(cfg["local_base"]) / "bronze" / cfg["table_bronze_macro"] / "macro_scraped.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        macro_pdf.to_csv(out, index=False)
        LOG.info("[scrape_macro] local CSV → %s (%d rows)", out, len(macro_pdf))
        if args.local_only:
            return 0

    try:
        return run_scrape_to_bronze(n_weeks=args.n_weeks, end_date=args.end_date)
    except Exception as exc:
        if args.local_csv:
            LOG.warning("[scrape_macro] Hive/HDFS write failed (%s) — local CSV kept", exc)
            return 0
        raise


if __name__ == "__main__":
    sys.exit(main())
