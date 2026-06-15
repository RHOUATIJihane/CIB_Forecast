"""Tests config / helpers."""

from __future__ import annotations

from pathlib import Path

from src.common.utils import full_table_name, hdfs_path, load_config


def test_load_config_cib_has_expected_keys() -> None:
    cfg = load_config("cib")
    assert cfg.get("table_features") == "features_cib_forecast"
    assert cfg.get("format") == "hive"


def test_config_file_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "config.ini").is_file()


def test_full_table_name_and_hdfs_path() -> None:
    cfg = load_config("cib")
    assert full_table_name(cfg, "bronze", "table_bronze_transactions").endswith(".transactions_raw")
    assert hdfs_path(cfg, "bronze/transactions_raw").endswith("/bronze/transactions_raw")
