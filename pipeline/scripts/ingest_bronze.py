"""INGESTION — zone bronze (raw landing).

Workflow data engineer (batch) :
  1. init_hive     → DDL / métadonnées (catalogue Hive)
  2. ingest        → ce script : **landing** des données brutes sur HDFS (ORC)
  3. transform     → silver (nettoyage, agrégation, qualité)
  4. policy        → règles métier par compte
  5. features      → feature store ML
  6. train / inference → modèles

En PFE la **source** est le générateur synthétique (simule un extract SI bancaire).
En production Attijari : remplacer ``generate_synthetic_dataset`` par lecture
fichiers/API (SFTP, Kafka, table source Oracle, etc.) — la **cible** reste
``cib_bronze.*`` sur HDFS.

Prérequis : HDFS démarré, ``./run.sh init-hive`` exécuté une fois.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import SparkSession

from src.common.table_writer import write_table_spark
from src.common.utils import get_logger, get_spark, load_config
from src.datagen.synthetic import generate_synthetic_dataset

LOG = get_logger(__name__)


def _pandas_to_spark(spark: SparkSession, pdf, table_key: str):
    """Convertit pandas → Spark via CSV temporaire.

    Évite ``createDataFrame(pandas)`` : cloudpickle de PySpark 3.3 plante sous Python 3.12
    (même contournement que ``transformation.py`` step4).
    """
    import os
    import tempfile

    import pandas as pd
    from pyspark.sql import functions as F

    pdf = pdf.copy()
    for col in pdf.columns:
        if col in ("date_operation", "week") or "date" in col.lower():
            pdf[col] = pd.to_datetime(pdf[col])

    fd, path = tempfile.mkstemp(suffix=".csv", prefix=f"cib_ingest_{table_key}_")
    os.close(fd)
    try:
        pdf.to_csv(path, index=False)
        sdf = (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .csv(f"file://{path}")
        )
        for col in pdf.columns:
            if col in ("date_operation", "week") or "date" in col.lower():
                if col in sdf.columns:
                    sdf = sdf.withColumn(col, F.to_timestamp(col))
        # Matérialiser avant suppression du CSV (Spark lit en lazy)
        sdf = sdf.cache()
        sdf.count()
        return sdf
    finally:
        os.unlink(path)


def run_ingestion(
    *,
    n_accounts: int = 200,
    n_weeks: int = 104,
    seed: int = 42,
    backup_csv: bool = False,
) -> int:
    """Extrait (ici : synthèse) puis charge les 3 tables bronze dans Hive."""
    cfg = load_config()
    if cfg.get("format", "hive").lower() != "hive":
        LOG.error("format=hive requis dans config.ini")
        return 1

    LOG.info("[INGEST] source=synthetic_datagen accounts=%d weeks=%d", n_accounts, n_weeks)
    payload = generate_synthetic_dataset(n_accounts=n_accounts, n_weeks=n_weeks, seed=seed)

    macro_source = (cfg.get("macro_source") or "synthetic").strip().lower()
    bronze_tables = [
        ("table_bronze_transactions", payload.transactions),
        ("table_bronze_calendar", payload.calendar),
    ]
    if macro_source == "synthetic":
        bronze_tables.append(("table_bronze_macro", payload.macro))
    elif macro_source == "hybrid":
        bronze_tables.append(("table_bronze_macro", payload.macro))
    else:
        LOG.info("[INGEST] macro_source=%s — macro table unchanged (run scrape-macro first)", macro_source)

    spark = get_spark("cib-ingest-bronze", enable_hive=True, cfg=cfg)
    try:
        for table_key, pdf in bronze_tables:
            sdf = _pandas_to_spark(spark, pdf, table_key)
            LOG.info("[INGEST] landing → cib_bronze.%s (%d rows)", cfg[table_key], len(pdf))
            write_table_spark(sdf, cfg, layer="bronze", table_key=table_key, mode="overwrite")
    finally:
        spark.stop()

    if backup_csv:
        base = Path(cfg["local_base"]) / "bronze"
        for name, pdf in [
            (cfg["table_bronze_transactions"], payload.transactions),
            (cfg["table_bronze_calendar"], payload.calendar),
            (cfg["table_bronze_macro"], payload.macro),
        ]:
            out = base / name / f"{name}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            pdf.to_csv(out, index=False)
            LOG.info("[INGEST] backup CSV %s", out)

    LOG.info("[INGEST] terminé — prochaine étape : transform (silver)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingestion batch zone bronze (landing HDFS/Hive).",
    )
    parser.add_argument("--n-accounts", type=int, default=200)
    parser.add_argument("--n-weeks", type=int, default=104)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backup-csv", action="store_true")
    args = parser.parse_args()
    return run_ingestion(
        n_accounts=args.n_accounts,
        n_weeks=args.n_weeks,
        seed=args.seed,
        backup_csv=args.backup_csv,
    )


if __name__ == "__main__":
    sys.exit(main())
