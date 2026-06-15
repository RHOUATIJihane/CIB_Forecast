"""Catégorisation v7 — conditions métriques encapsulant les scénarios EXP1--EXP6."""

from __future__ import annotations


def regularity_score(acf1: float, cv: float) -> float:
    """Score de régularité (identique EXP3 dans ``cib_experiments_v6``)."""
    return float(max(0.0, min(1.0, 0.5 * acf1 + 0.5 * max(0.0, 1.0 - min(cv, 2.0) / 2.0))))


def categorize_account_v6(
    acf1: float,
    cv: float,
    trend_strength: float,
    seasonality_strength: float,
    n_weeks: int,
) -> str:
    """Ancienne catégorisation A--F poolée (v6 notebook)."""
    if n_weeks < 52:
        return "E_short_history"
    if seasonality_strength > 0.60 and acf1 >= 0.30 and cv < 1.0 and trend_strength < 0.50:
        return "B_seasonal"
    if trend_strength > 0.20 and seasonality_strength < 0.60 and cv < 1.0:
        return "C_trending"
    if (acf1 < 0.30 or cv > 1.0) and seasonality_strength < 0.60 and trend_strength < 0.50:
        return "D_volatile"
    if acf1 > 0.60 and cv < 0.50 and seasonality_strength < 0.30 and trend_strength < 0.20:
        return "A_regular_stable"
    return "F_mixed"


def categorize_account(
    acf1: float,
    cv: float,
    trend_strength: float,
    seasonality_strength: float,
    n_weeks: int,
) -> str:
    """Assigne une catégorie encapsulée (v7) — une EXP source par profil.

    Priorité (arbre de décision) :
      E  → EXP1 C_short
      D_sparse → EXP6 sparse
      F_flat   → EXP6 flat
      B_seasonal → EXP3 seasonal
      C_trending → EXP2 C_level_shift
      A_regular_stable → EXP1 A_clean + EXP3 regular
      D_noisy  → EXP1 B_noisy
      D_irregular → EXP3 irregular (défaut résiduel)
    """
    if n_weeks < 52:
        return "E_short_history"
    if cv > 1.0:
        return "D_sparse"
    if trend_strength < 0.02 and cv < 0.40:
        return "F_flat"
    if seasonality_strength > 0.60 and acf1 >= 0.30 and cv < 1.0 and trend_strength < 0.50:
        return "B_seasonal"
    if trend_strength >= 0.12:
        return "C_trending"
    reg = regularity_score(acf1, cv)
    if cv < 0.45 and reg >= 0.55:
        return "A_regular_stable"
    if cv >= 0.50 and trend_strength < 0.12:
        return "D_noisy"
    return "D_irregular"


def categorize_account_column():
    """Expression Spark équivalente à :func:`categorize_account` (v7)."""
    from pyspark.sql import functions as F

    acf = F.col("acf_lag1")
    cv = F.col("cv_cashflow")
    trend = F.col("trend_strength")
    seas = F.col("seasonality_strength")
    n = F.col("n_obs")
    reg = F.greatest(
        F.lit(0.0),
        F.least(
            F.lit(1.0),
            F.lit(0.5) * acf + F.lit(0.5) * F.greatest(F.lit(0.0), F.lit(1.0) - F.least(cv, F.lit(2.0)) / F.lit(2.0)),
        ),
    )

    return (
        F.when(n < F.lit(52), F.lit("E_short_history"))
        .when(cv > F.lit(1.0), F.lit("D_sparse"))
        .when((trend < F.lit(0.02)) & (cv < F.lit(0.40)), F.lit("F_flat"))
        .when(
            (seas > F.lit(0.60)) & (acf >= F.lit(0.30)) & (cv < F.lit(1.0)) & (trend < F.lit(0.50)),
            F.lit("B_seasonal"),
        )
        .when(trend >= F.lit(0.12), F.lit("C_trending"))
        .when((cv < F.lit(0.45)) & (reg >= F.lit(0.55)), F.lit("A_regular_stable"))
        .when((cv >= F.lit(0.50)) & (trend < F.lit(0.12)), F.lit("D_noisy"))
        .otherwise(F.lit("D_irregular"))
    )
