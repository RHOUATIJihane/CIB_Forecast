"""Tests des règles de policy v3.0 (sans Spark)."""

from __future__ import annotations

from src.policy_rules import (
    POLICY_VERSION,
    evaluate_externals_policy,
    use_externals_classification_v3,
    use_externals_regression_oracle,
    use_externals_regression_v3,
)


def test_policy_version_is_v3() -> None:
    assert POLICY_VERSION == "v3.0"


def _metrics(
    acf1: float = 0.40,
    cv: float = 0.50,
    completeness: float = 0.70,
    predictability: float = 40.0,
    composite: float = 45.0,
):
    return dict(
        acf1=acf1,
        cv_cashflow=cv,
        completeness_ratio=completeness,
        predictability_score=predictability,
        composite_score=composite,
    )


def test_r0a_sparse_never_activates() -> None:
    d = evaluate_externals_policy(**_metrics(completeness=0.35))
    assert d.rule_id == "R0-A"
    assert not d.use_externals_regression
    assert not d.use_externals_classification


def test_r0b_flat_never_activates() -> None:
    d = evaluate_externals_policy(**_metrics(cv=0.05))
    assert d.rule_id == "R0-B"
    assert not d.use_externals_regression


def test_r4_noisy_external_blocks() -> None:
    d = evaluate_externals_policy(**_metrics(), external_noise_frac=0.25)
    assert d.rule_id == "R4"
    assert not d.use_externals_regression


def test_r1_regular_activates_both() -> None:
    d = evaluate_externals_policy(**_metrics(acf1=0.40, completeness=0.65, predictability=36.0))
    assert d.rule_id == "R1"
    assert d.use_externals_regression
    assert d.use_externals_classification


def test_r2_irregular_regression_only() -> None:
    d = evaluate_externals_policy(
        **_metrics(acf1=0.25, completeness=0.60, predictability=30.0, composite=38.0)
    )
    assert d.rule_id == "R2"
    assert d.use_externals_regression
    assert not d.use_externals_classification


def test_r3_volatile_regression_only() -> None:
    d = evaluate_externals_policy(
        **_metrics(acf1=0.15, cv=1.60, completeness=0.50, predictability=20.0, composite=32.0)
    )
    assert d.rule_id == "R3"
    assert d.use_externals_regression
    assert not d.use_externals_classification


def test_baseline_when_no_rule_matches() -> None:
    d = evaluate_externals_policy(
        **_metrics(acf1=0.25, cv=0.50, completeness=0.50, predictability=30.0, composite=30.0)
    )
    assert d.rule_id == "baseline"
    assert not d.use_externals_regression


def test_regression_oracle() -> None:
    assert use_externals_regression_oracle(1.5) is True
    assert use_externals_regression_oracle(-0.1) is False


def test_v3_helpers_match_evaluate() -> None:
    m = _metrics(acf1=0.40, completeness=0.65, predictability=36.0)
    assert use_externals_regression_v3(**m) is True
    assert use_externals_classification_v3(**m) is True
