"""Tests du module de dérivation policy (sans I/O lourde)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.derive_policy_from_exports import (
    build_sector_cv_lookup,
    build_regression_oracle,
    _evaluate_v2_rules,
)

EXPORTS = Path("/home/jihane/cib_project/cib_project_outputs/cib_experiment_outputs")


def test_build_lookup_from_exports() -> None:
    if not (EXPORTS / "lift_by_account_condition_model.csv").is_file():
        return
    from scripts.derive_policy_from_exports import _load_regression_lifts

    irf = _load_regression_lifts(EXPORTS)
    lookup = build_sector_cv_lookup(irf)
    assert not lookup.empty
    assert "use_externals_regression" in lookup.columns
    assert "sector" in lookup.columns


def test_regression_oracle_flags() -> None:
    irf = pd.DataFrame(
        {
            "name": ["a", "b"],
            "sector": ["transport", "retail"],
            "notes": ["reg=0.4", "reg=0.4"],
            "rmse_lift_pct": [2.0, -1.0],
            "reg": [0.4, 0.4],
            "cv_bucket": ["[0.30,0.50)", "[0.30,0.50)"],
        }
    )
    oracle = build_regression_oracle(irf)
    assert oracle.loc[oracle["experiment_account"] == "a", "use_externals_regression"].iloc[0]
    assert not oracle.loc[oracle["experiment_account"] == "b", "use_externals_regression"].iloc[0]


def test_evaluate_v2_rules_keys() -> None:
    irf = pd.DataFrame(
        {
            "sector": ["transport", "retail"],
            "reg": [0.4, 0.2],
            "rmse_lift_pct": [3.0, 1.0],
        }
    )
    stats = _evaluate_v2_rules(irf)
    assert "regression_v2_on_n" in stats
    assert stats["regression_v2_on_n"] >= 1
