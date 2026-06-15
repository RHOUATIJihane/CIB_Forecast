"""Tests catégorisation encapsulée v7 et chargement politique (sans Spark)."""

from __future__ import annotations

from src.account_categorization import categorize_account, categorize_account_v6
from src.category_policy_v6 import load_category_policy_table as load_category_policy_table_v6
from src.category_policy_v7 import (
    load_category_policy_table as load_category_policy_table_v7,
    resolve_policy_v7_csv_path,
)
from src.common.utils import load_config


def test_categorize_short_history() -> None:
    assert categorize_account(0.5, 0.3, 0.1, 0.1, 40) == "E_short_history"


def test_categorize_sparse() -> None:
    assert categorize_account(0.2, 1.2, 0.1, 0.2, 60) == "D_sparse"


def test_categorize_flat() -> None:
    assert categorize_account(0.1, 0.3, 0.01, 0.1, 60) == "F_flat"


def test_categorize_trending() -> None:
    assert categorize_account(0.35, 0.8, 0.25, 0.4, 60) == "C_trending"


def test_categorize_regular_stable() -> None:
    assert categorize_account(0.6, 0.3, 0.1, 0.1, 60) == "A_regular_stable"


def test_categorize_noisy() -> None:
    assert categorize_account(0.2, 0.55, 0.08, 0.2, 60) == "D_noisy"


def test_categorize_irregular() -> None:
    assert categorize_account(0.2, 0.48, 0.08, 0.2, 60) == "D_irregular"


def test_categorize_v6_legacy_volatile() -> None:
    assert categorize_account_v6(0.2, 1.2, 0.1, 0.2, 60) == "D_volatile"


def test_policy_v7_distinct_decisions_per_category() -> None:
    """Chaque catégorie encapsulée a sa propre décision issue de son EXP source."""
    cfg = load_config()
    by_cat = load_category_policy_table_v7(cfg).set_index("account_category")

    assert by_cat.loc["E_short_history", "regression_model"] == "lgbm"
    assert bool(by_cat.loc["E_short_history", "activate_externals"]) is False

    assert by_cat.loc["D_noisy", "regression_model"] == "ridge"
    assert bool(by_cat.loc["D_noisy", "activate_externals"]) is False

    assert by_cat.loc["D_sparse", "regression_model"] == "ridge"
    assert bool(by_cat.loc["D_sparse", "activate_externals"]) is False

    assert by_cat.loc["F_flat", "regression_model"] == "rf"
    assert bool(by_cat.loc["F_flat", "activate_externals"]) is False

    assert by_cat.loc["B_seasonal", "regression_model"] == "lgbm"
    assert bool(by_cat.loc["B_seasonal", "activate_externals"]) is False


def test_load_policy_v7_matches_export() -> None:
    cfg = load_config()
    path = resolve_policy_v7_csv_path(cfg)
    assert path.name == "policy_v7_encapsulated.csv"

    table = load_category_policy_table_v7(cfg)
    by_cat = table.set_index("account_category")

    row_c = by_cat.loc["C_trending"]
    assert bool(row_c["activate_externals"]) is False
    assert row_c["regression_model"] == "rf"

    row_d = by_cat.loc["D_irregular"]
    assert bool(row_d["activate_externals"]) is True
    assert row_d["regression_model"] == "ridge"

    row_a = by_cat.loc["A_regular_stable"]
    assert bool(row_a["activate_externals"]) is True
    assert row_a["regression_model"] == "rf"


def test_load_policy_v6_legacy_still_available() -> None:
    cfg = load_config()
    table = load_category_policy_table_v6(cfg)
    assert "C_trending" in table["account_category"].values
