"""Écriture des tables (Hive / ORC / CSV) avec sélection du format via la config."""

from __future__ import annotations

from configparser import SectionProxy
from pathlib import Path
from typing import TYPE_CHECKING

from src.common.utils import full_table_name, get_logger, hdfs_path

if TYPE_CHECKING:
    import pandas as pd
    from pyspark.sql import DataFrame

LOG = get_logger(__name__)


def _storage_format(cfg: SectionProxy) -> str:
    return cfg.get("format", fallback="csv").lower()


def _csv_dir(cfg: SectionProxy, layer: str, table_key: str) -> str:
    return hdfs_path(cfg, f"{layer}/{cfg[table_key]}")


def write_table_spark(
    df: DataFrame,
    cfg: SectionProxy,
    layer: str,
    table_key: str,
    *,
    file_format: str | None = None,
    partition_col: str | None = None,
    mode: str = "overwrite",
) -> None:
    """Écrit un DataFrame Spark selon ``format`` configuré (hive / csv / orc)."""
    fmt = file_format or _storage_format(cfg)
    writer = df.write.mode(mode)
    if partition_col and partition_col in df.columns:
        writer = writer.partitionBy(partition_col)

    if fmt == "hive":
        target_format = cfg.get("hive_format", fallback="orc")
        name = full_table_name(cfg, layer, table_key)
        LOG.info("Writing Hive table %s (%s)", name, target_format)
        writer.format(target_format).saveAsTable(name)
        return

    if fmt in {"orc", "parquet"}:
        path = _csv_dir(cfg, layer, table_key)
        LOG.info("Writing %s to %s", fmt, path)
        writer.format(fmt).save(path)
        return

    path = _csv_dir(cfg, layer, table_key)
    LOG.info("Writing CSV to %s", path)
    writer.option("header", "true").csv(path)


def write_table_pandas(
    df: pd.DataFrame,
    cfg: SectionProxy,
    layer: str,
    table_key: str,
) -> None:
    """Écrit un DataFrame pandas en CSV unique (mode local / tests)."""
    base = cfg.get("local_base", fallback=str(Path.cwd() / "data"))
    out = Path(base) / layer / f"{cfg[table_key]}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Writing local CSV %s (%d rows)", out, len(df))
    df.to_csv(out, index=False)
