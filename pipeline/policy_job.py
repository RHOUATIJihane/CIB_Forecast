"""Job Spark : silver_metrics → account_policy (catégorie A--F + politique v6 notebook)."""

from __future__ import annotations

from src.common.table_loader import read_table_spark
from src.common.table_writer import write_table_spark
from src.common.utils import get_logger, get_spark, load_config
from src.policy import build_account_policy


def main() -> None:
    log = get_logger(__name__)
    cfg = load_config()
    spark = get_spark("cib-policy", enable_hive=True, cfg=cfg)
    try:
        log.info("Reading silver quality metrics...")
        quality = read_table_spark(spark, cfg, layer="silver", table_key="table_silver_metrics")
        policy = build_account_policy(quality, spark=spark, cfg=cfg)
        try:
            assignment = read_table_spark(
                spark, cfg, layer="silver", table_key="table_silver_macro_assignment"
            ).select(
                "numero_compte",
                "macro_primary",
                "macro_secondary",
                "externals_for_model",
            )
            policy = policy.join(assignment, on="numero_compte", how="left")
            log.info("Joined account_macro_assignment into account_policy")
        except Exception as exc:
            log.warning("account_macro_assignment not found (%s) — policy without macro columns", exc)
        log.info("Writing account_policy table...")
        write_table_spark(policy, cfg, layer="silver", table_key="table_policy")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
