"""Charge les tables de référence (mapping secteur → macro) vers la zone bronze."""

from __future__ import annotations

import sys
from pathlib import Path

from src.common.table_writer import write_table_spark
from src.common.utils import get_logger, get_spark, load_config
from src.sector_macro_mapping import default_mapping_csv_path

LOG = get_logger(__name__)


def _pandas_to_spark(spark, pdf, table_key: str):
    import os
    import tempfile

    import pandas as pd
    from pyspark.sql import functions as F

    pdf = pdf.copy()
    fd, path = tempfile.mkstemp(suffix=".csv", prefix=f"cib_ref_{table_key}_")
    os.close(fd)
    try:
        pdf.to_csv(path, index=False)
        sdf = spark.read.option("header", "true").option("inferSchema", "true").csv(f"file://{path}")
        sdf = sdf.cache()
        sdf.count()
        return sdf
    finally:
        os.unlink(path)


def run_load_reference_tables() -> int:
    cfg = load_config()
    csv_path = Path(cfg.get("reference_sector_macro_csv", "") or default_mapping_csv_path())
    if not csv_path.is_file():
        LOG.error("Reference CSV not found: %s", csv_path)
        return 1

    import pandas as pd

    pdf = pd.read_csv(csv_path)
    LOG.info("[load_reference] sector_macro_mapping from %s (%d rows)", csv_path, len(pdf))

    spark = get_spark("cib-load-reference", enable_hive=True, cfg=cfg)
    try:
        sdf = _pandas_to_spark(spark, pdf, "sector_macro_mapping")
        write_table_spark(
            sdf,
            cfg,
            layer="bronze",
            table_key="table_bronze_sector_macro",
            mode="overwrite",
        )
    finally:
        spark.stop()

    LOG.info("[load_reference] done → cib_bronze.%s", cfg["table_bronze_sector_macro"])
    return 0


def main() -> int:
    return run_load_reference_tables()


if __name__ == "__main__":
    sys.exit(main())
