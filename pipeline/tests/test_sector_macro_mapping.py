"""Tests du mapping secteur synthétique → macro_primary."""

from __future__ import annotations

from src.sector_macro_mapping import (
    load_sector_macro_mapping_pandas,
    parse_externals_for_model,
    resolve_macro_keys,
)


def test_agriculture_maps_to_commodity() -> None:
    mapping = load_sector_macro_mapping_pandas()
    out = resolve_macro_keys("agriculture", mapping)
    assert out["macro_primary"] == "commodity_index_z"
    assert "commodity_index_z" in out["externals_for_model"]
    assert "is_ramadan" in out["externals_for_model"]


def test_construction_maps_to_realestate_and_oil() -> None:
    mapping = load_sector_macro_mapping_pandas()
    out = resolve_macro_keys("construction", mapping)
    assert out["macro_primary"] == "realestate_index_z"
    assert out["macro_secondary"] == "oil_price_z"
    keys = parse_externals_for_model(out["externals_for_model"])
    assert "realestate_index_z" in keys
    assert "oil_price_z" in keys


def test_retail_calendar_only() -> None:
    mapping = load_sector_macro_mapping_pandas()
    out = resolve_macro_keys("retail", mapping)
    assert out["macro_primary"] == ""
    keys = parse_externals_for_model(out["externals_for_model"])
    assert "is_payroll_week" in keys
    assert "oil_price_z" not in keys
