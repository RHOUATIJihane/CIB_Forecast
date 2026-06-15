"""Session Spark, configuration INI, logger partagés."""

from __future__ import annotations

import configparser
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.ini"
LOGGER = logging.getLogger("cib_forecast")


def get_logger(name: str = "cib_forecast") -> logging.Logger:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    return logging.getLogger(name)


def load_config(env: str | None = None) -> configparser.SectionProxy:
    """Charge la section unique ``[cib]`` de ``config.ini`` (stack HDFS + Hive)."""
    if not DEFAULT_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"config.ini introuvable: {DEFAULT_CONFIG_PATH}")
    cp = configparser.ConfigParser()
    cp.read(DEFAULT_CONFIG_PATH)
    section = env or os.environ.get("CIB_ENV", "cib")
    if section not in cp:
        section = "cib" if "cib" in cp else "DEFAULT"
    return cp[section]


def full_table_name(cfg: configparser.SectionProxy, layer: str, table_key: str) -> str:
    """Construit `schema_<layer>.<table_key>` depuis la config (ex: cib_silver.weekly_cashflow_account)."""
    schema = cfg[f"schema_{layer}"]
    table = cfg[table_key]
    return f"{schema}.{table}"


def hdfs_path(cfg: configparser.SectionProxy, sub: str) -> str:
    """Préfixe un chemin sous `hdfs_base` de la config."""
    base = cfg["hdfs_base"].rstrip("/")
    return f"{base}/{sub.lstrip('/')}"


def get_spark(
    app_name: str = "cib_forecast",
    *,
    enable_hive: bool = True,
    cfg: configparser.SectionProxy | None = None,
) -> SparkSession:
    """Crée une SparkSession avec support Hive (metastore + warehouse HDFS)."""
    from pyspark.sql import SparkSession  # import paresseux

    builder = SparkSession.builder.appName(app_name).config(
        "spark.sql.session.timeZone", "UTC"
    )
    if enable_hive:
        if cfg is None:
            try:
                cfg = load_config()
            except FileNotFoundError:
                cfg = None
        warehouse = (
            cfg.get("hive_warehouse", "hdfs://127.0.0.1:9000/user/hive/warehouse")
            if cfg
            else "hdfs://127.0.0.1:9000/user/hive/warehouse"
        )
        builder = (
            builder.enableHiveSupport()
            .config("spark.sql.warehouse.dir", warehouse)
            .config("spark.hadoop.hive.metastore.warehouse.dir", warehouse)
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "WARN"))
    return spark


def is_valid_numeric(value: object) -> bool:
    """Filtre simple pour éviter NaN / inf dans la logique métier."""
    import math

    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)
