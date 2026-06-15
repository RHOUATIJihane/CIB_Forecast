"""Job Spark : quality_metrics + sector_macro_mapping → account_macro_assignment."""

from __future__ import annotations

from src.common.table_loader import read_table_spark
from src.common.table_writer import write_table_spark
from src.common.utils import get_logger, get_spark, load_config
from src.sector_macro_mapping import build_account_macro_assignment_pandas, load_sector_macro_mapping_pandas


def _pandas_to_spark(spark, pdf):
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".csv", prefix="cib_assign_macro_")
    os.close(fd)
    try:
        pdf.to_csv(path, index=False)
        sdf = (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .csv(f"file://{path}")
        )
        sdf = sdf.cache()
        sdf.count()
        return sdf
    finally:
        os.unlink(path)


def main() -> None:
    log = get_logger(__name__)
    cfg = load_config()
    spark = get_spark("cib-assign-macro", enable_hive=True, cfg=cfg)
    try:
        log.info("Reading silver quality metrics...")
        quality = read_table_spark(spark, cfg, layer="silver", table_key="table_silver_metrics")
        accounts_pdf = quality.select("numero_compte", "sector").dropDuplicates(["numero_compte"]).toPandas()

        try:
            mapping_pdf = read_table_spark(
                spark, cfg, layer="bronze", table_key="table_bronze_sector_macro"
            ).toPandas()
        except Exception:
            log.warning("Bronze mapping table missing — fallback to reference CSV")
            mapping_pdf = load_sector_macro_mapping_pandas()

        assignment_pdf = build_account_macro_assignment_pandas(accounts_pdf, mapping_pdf)
        log.info("Writing account_macro_assignment (%d accounts)...", len(assignment_pdf))
        sdf = _pandas_to_spark(spark, assignment_pdf)
        write_table_spark(
            sdf,
            cfg,
            layer="silver",
            table_key="table_silver_macro_assignment",
            mode="overwrite",
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
