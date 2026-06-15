"""Job Spark : silver weekly + policy + calendrier + macro → features ORC/CSV."""

from __future__ import annotations

from src.common.table_loader import read_table_spark
from src.common.table_writer import write_table_spark
from src.common.utils import get_logger, get_spark, load_config
from src.features_pipeline import build_features


def main() -> None:
    log = get_logger(__name__)
    cfg = load_config()
    spark = get_spark("cib-features", enable_hive=True, cfg=cfg)
    try:
        log.info("Loading silver + bronze inputs for feature engineering...")
        weekly = read_table_spark(spark, cfg, layer="silver", table_key="table_silver_weekly")
        quality = read_table_spark(spark, cfg, layer="silver", table_key="table_silver_metrics")
        calendar = read_table_spark(spark, cfg, layer="bronze", table_key="table_bronze_calendar")
        macro = read_table_spark(spark, cfg, layer="bronze", table_key="table_bronze_macro")

        features_df = build_features(weekly, quality, calendar, macro)

        log.info("Writing features_cib_forecast...")
        write_table_spark(features_df, cfg, layer="ml", table_key="table_features")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
