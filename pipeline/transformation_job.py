"""Job Spark : bronze (transactions) → silver (weekly + qualité)."""

from __future__ import annotations

from src.common.table_loader import read_table_spark
from src.common.table_writer import write_table_spark
from src.common.utils import get_logger, get_spark, load_config
from src.transformation import run_transformation_pipeline


def main() -> None:
    log = get_logger(__name__)
    cfg = load_config()
    quality_kwargs = {
        "acf_min": float(cfg["quality_acf_lag1_min"]),
        "adf_pvalue_max": float(cfg["quality_adf_pvalue_max"]),
        "completeness_min": float(cfg["quality_completeness_min"]),
        "transactions_min": int(cfg["quality_transactions_min"]),
        "cv_min": float(cfg["quality_cv_min"]),
        "cv_max": float(cfg["quality_cv_max"]),
        "predictability_min": float(cfg["quality_predictability_min"]),
        "composite_min": float(cfg["quality_composite_min"]),
    }

    spark = get_spark("cib-transformation", enable_hive=True, cfg=cfg)
    try:
        log.info("Reading bronze transactions...")
        transactions = read_table_spark(spark, cfg, layer="bronze", table_key="table_bronze_transactions")
        weekly, quality = run_transformation_pipeline(
            transactions,
            quality_kwargs=quality_kwargs,
            min_obs=int(cfg["quality_min_obs"]),
        )

        log.info("Writing silver weekly cashflow...")
        write_table_spark(weekly, cfg, layer="silver", table_key="table_silver_weekly")
        log.info("Writing silver quality metrics (post step7)...")
        write_table_spark(quality, cfg, layer="silver", table_key="table_silver_metrics")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
