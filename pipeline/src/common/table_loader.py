"""Lecture des tables (Hive) ou de leurs équivalents CSV en HDFS / local.

Le projet supporte deux modes :
- ``format = hive`` (prod) : lecture via `spark.table("schema.table")`.
- ``format = csv`` (dev / pseudo-distribué) : lecture d'un fichier CSV à
  l'emplacement ``hdfs_base/<layer>/<table>/`` (ou ``data/<layer>/<table>.csv``
  en local sans Spark).
"""

from __future__ import annotations

from configparser import SectionProxy
from pathlib import Path
from typing import TYPE_CHECKING

from src.common.utils import full_table_name, get_logger, hdfs_path

if TYPE_CHECKING:
    import pandas as pd
    from pyspark.sql import DataFrame, SparkSession

LOG = get_logger(__name__)


def _storage_format(cfg: SectionProxy) -> str:
    return cfg.get("format", fallback="csv").lower()


def _csv_path(cfg: SectionProxy, layer: str, table_key: str) -> str:
    """Chemin CSV d'une table dans `hdfs_base/<layer>/<table>/` (Spark CSV directory)."""
    table = cfg[table_key]
    return hdfs_path(cfg, f"{layer}/{table}")


def _local_csv_path(cfg: SectionProxy, layer: str, table_key: str) -> Path:
    """Chemin local équivalent (utile pour les tests sans Spark)."""
    table = cfg[table_key]
    base = cfg.get("local_base", fallback=str(Path.cwd() / "data"))
    return Path(base) / layer / f"{table}.csv"


def read_table_spark(
    spark: SparkSession,
    cfg: SectionProxy,
    layer: str,
    table_key: str,
    *,
    schema: object | None = None,
) -> DataFrame:
    """Lit une table (Hive) ou un CSV (HDFS) selon `format` dans la config."""
    fmt = _storage_format(cfg)
    if fmt == "hive":
        name = full_table_name(cfg, layer, table_key)
        LOG.info("Reading Hive table %s", name)
        return spark.table(name)

    path = _csv_path(cfg, layer, table_key)
    LOG.info("Reading CSV from %s", path)
    reader = spark.read.option("header", "true")
    if schema is not None:
        reader = reader.schema(schema)
    else:
        reader = reader.option("inferSchema", "true")
    return reader.csv(path)


def read_table_pandas(cfg: SectionProxy, layer: str, table_key: str) -> pd.DataFrame:
    """Lit une table en pandas : Hive (Spark) si ``format=hive``, sinon CSV local."""
    import pandas as pd

    if _storage_format(cfg) == "hive":
        return read_table_pandas_from_hive(cfg, layer, table_key)

    path = _local_csv_path(cfg, layer, table_key)
    LOG.info("Reading local CSV %s", path)
    return pd.read_csv(path)


def read_table_pandas_from_hive(cfg: SectionProxy, layer: str, table_key: str) -> pd.DataFrame:
    """Lit ``schema.table`` via Spark et retourne un DataFrame pandas."""
    from src.common.utils import get_spark

    name = full_table_name(cfg, layer, table_key)
    spark = get_spark(f"cib-read-{table_key}", enable_hive=True, cfg=cfg)
    LOG.info("Reading Hive table %s → pandas", name)
    return spark.table(name).toPandas()
