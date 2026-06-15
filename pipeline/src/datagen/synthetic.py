"""Génération de données synthétiques (lab) — port du BLOC A de `final-cib.ipynb`.

Trois sorties bronze :
- ``transactions_raw``         : lignes de transactions individuelles
- ``calendar_weekly_flags``    : flags calendaires hebdomadaires (Maroc)
- ``macro_indicators_weekly``  : indicateurs sectoriels (z-scorés) hebdomadaires
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Catalogue de paramètres (identiques au notebook)
# ---------------------------------------------------------------------------

SECTOR_BETAS: dict[str, list[float]] = {
    # oil, commodity, masi, realestate, ramadan, eid, month_end, payroll, quarter_end
    "transport":    [8000, 500, 200, 100, -3000, 2000, 4000, 3500, -2000],
    "agriculture":  [300, 7000, 200, 200, -1000, 1500, 3500, 3000, -1500],
    "investment":   [200, 300, 6000, 500, 500, 1000, 2000, 2000, 3000],
    "construction": [300, 500, 200, 7500, -1000, 2000, 3000, 3000, -1000],
    "retail":       [200, 300, 200, 200, 4000, 5000, 5000, 5000, 1000],
    "mixed":        [1500, 1500, 1500, 1500, 1000, 1500, 2500, 2500, 500],
}

SECTOR_BASE: dict[str, int] = {
    "transport": 50_000, "agriculture": 40_000, "investment": 70_000,
    "construction": 60_000, "retail": 35_000, "mixed": 45_000,
}

# (phi_AR, multiplicateur sigma, probabilité débit)
ACCOUNT_TYPE_MIX: dict[str, tuple[float, float, float]] = {
    "normal":   (0.70, 1.0, 0.15),
    "seasonal": (0.75, 1.2, 0.12),
    "trend_up": (0.72, 0.8, 0.18),
    "volatile": (0.75, 2.0, 0.30),
    "sparse":   (0.50, 1.5, 0.20),
    "flat":     (0.30, 0.05, 0.05),
}

ACCOUNT_TYPE_PROBS: dict[str, float] = {
    "normal": 0.40, "seasonal": 0.25, "trend_up": 0.15,
    "volatile": 0.10, "sparse": 0.07, "flat": 0.03,
}

SECTOR_PROBS: dict[str, float] = {
    "transport": 0.20, "agriculture": 0.15, "investment": 0.15,
    "construction": 0.15, "retail": 0.25, "mixed": 0.10,
}

NOISE_BY_TYPE: dict[str, float] = {
    "normal": 0.35, "seasonal": 0.25, "trend_up": 0.30,
    "volatile": 0.55, "sparse": 0.70, "flat": 0.90,
}


@dataclass
class SyntheticBronze:
    transactions: pd.DataFrame
    calendar: pd.DataFrame
    macro: pd.DataFrame


# ---------------------------------------------------------------------------
# Calendrier & macro (connus à l'avance — pas de fuite future)
# ---------------------------------------------------------------------------

def generate_calendar_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Flags calendaires Maroc : Ramadan, Aïd, fin de mois, paie, TVA."""
    df = pd.DataFrame({"date": dates})
    df["week"] = df["date"].dt.to_period("W").apply(lambda x: x.start_time)
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["week_num"] = df["date"].dt.isocalendar().week.astype(int)

    ramadan_ranges = [
        ("2022-04-02", "2022-05-01"),
        ("2023-03-22", "2023-04-20"),
        ("2024-03-10", "2024-04-08"),
    ]
    df["is_ramadan"] = 0
    for start, end in ramadan_ranges:
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        df.loc[mask, "is_ramadan"] = 1

    df["is_eid_alfitr"] = 0
    for eid in ["2022-05-02", "2023-04-21", "2024-04-10"]:
        mask = (df["date"] >= pd.Timestamp(eid)) & (
            df["date"] <= pd.Timestamp(eid) + pd.Timedelta(days=7)
        )
        df.loc[mask, "is_eid_alfitr"] = 1

    df["is_new_year_week"] = ((df["month"] == 1) & (df["week_num"] == 1)).astype(int)
    df["is_month_end_week"] = (df["day"] >= 26).astype(int)
    df["is_quarter_end"] = (
        df["month"].isin([3, 6, 9, 12]) & (df["day"] >= 25)
    ).astype(int)
    df["is_tax_deadline_week"] = (
        df["month"].isin([3, 6, 9, 12]) & (df["day"] >= 24)
    ).astype(int)
    df["is_payroll_week"] = ((df["day"] >= 23) & (df["day"] <= 27)).astype(int)

    cal = df.groupby("week").agg(
        is_ramadan=("is_ramadan", "max"),
        is_eid_alfitr=("is_eid_alfitr", "max"),
        is_new_year_week=("is_new_year_week", "max"),
        is_month_end_week=("is_month_end_week", "max"),
        is_quarter_end=("is_quarter_end", "max"),
        is_tax_deadline_week=("is_tax_deadline_week", "max"),
        is_payroll_week=("is_payroll_week", "max"),
    ).reset_index()
    return cal


def _ar1_series(mu: float, sigma: float, phi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    eps = rng.normal(0.0, sigma, n)
    x = np.zeros(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = mu + phi * (x[i - 1] - mu) + eps[i]
    return x


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-10)


def generate_sector_indicators(
    dates: pd.DatetimeIndex,
    n_weeks: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """AR(1) z-scorés : pétrole, commodity, MASI, immobilier."""
    rng = rng if rng is not None else np.random.default_rng(42)
    weeks = pd.date_range(start=dates[0], periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)

    oil_raw = _ar1_series(85, 6, 0.92, n_weeks, rng) + 8 * np.sin(2 * np.pi * t / 78)
    commodity_raw = (
        _ar1_series(100, 5, 0.88, n_weeks, rng) + 12 * np.sin(2 * np.pi * t / 52 + 1.2)
    )
    masi_raw = _ar1_series(12000, 300, 0.95, n_weeks, rng) + 15 * t
    realestate_raw = _ar1_series(1000, 20, 0.97, n_weeks, rng)

    return pd.DataFrame({
        "week": weeks,
        "oil_price_z": _zscore(oil_raw),
        "commodity_index_z": _zscore(commodity_raw),
        "masi_index_z": _zscore(masi_raw),
        "realestate_index_z": _zscore(realestate_raw),
    })


# ---------------------------------------------------------------------------
# Génération d'un compte
# ---------------------------------------------------------------------------

def generate_account_series(
    account_id: str,
    sector: str,
    account_type: str,
    calendar_df: pd.DataFrame,
    indicators_df: pd.DataFrame,
    n_weeks: int = 104,
    noise_fraction: float = 0.35,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """``cashflow_t = base + Σ(beta_k × external_k_t) + dynamics_t + AR1_residual_t``."""
    rng = rng if rng is not None else np.random.default_rng()
    dates = pd.date_range(start="2022-01-03", periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)

    cal = calendar_df.set_index("week")
    ind = indicators_df.set_index("week")
    betas = SECTOR_BETAS[sector]
    base = SECTOR_BASE[sector]

    ext_signal = np.zeros(n_weeks)
    for i, date in enumerate(dates):
        try:
            row_ind = ind.loc[date]
            row_cal = cal.loc[date]
        except KeyError:
            continue
        ext_signal[i] = (
            betas[0] * row_ind["oil_price_z"]
            + betas[1] * row_ind["commodity_index_z"]
            + betas[2] * row_ind["masi_index_z"]
            + betas[3] * row_ind["realestate_index_z"]
            + betas[4] * row_cal["is_ramadan"]
            + betas[5] * row_cal["is_eid_alfitr"]
            + betas[6] * row_cal["is_month_end_week"]
            + betas[7] * row_cal["is_payroll_week"]
            + betas[8] * row_cal["is_quarter_end"]
        )

    phi, sigma_mult, debit_prob = ACCOUNT_TYPE_MIX[account_type]
    sigma_base = base * 0.35 * sigma_mult

    def _residual_ar1(phi_val: float, sigma_val: float) -> np.ndarray:
        eps = rng.normal(0.0, sigma_val, n_weeks)
        out = np.zeros(n_weeks)
        for i in range(1, n_weeks):
            out[i] = phi_val * out[i - 1] + eps[i]
        return out

    if account_type == "seasonal":
        dynamics = (base * 0.30) * np.sin(2 * np.pi * t / 52) + (base * 0.10) * np.sin(
            2 * np.pi * t * 2 / 52 + 0.4
        )
    elif account_type == "trend_up":
        dynamics = (base / n_weeks) * 0.8 * t + (base * 0.08) * np.sin(2 * np.pi * t / 52)
    elif account_type == "volatile":
        dynamics = np.zeros(n_weeks)
        sigma_base *= 1.5
    elif account_type == "sparse":
        mask = np.zeros(n_weeks)
        mask[rng.choice(n_weeks, size=int(n_weeks * 0.45), replace=False)] = 1
        dynamics = np.zeros(n_weeks)
        ext_signal *= mask
        sigma_base = 0
    elif account_type == "flat":
        dynamics = np.zeros(n_weeks)
        ext_signal *= 0.05
        sigma_base = base * 0.001
    else:  # normal
        dynamics = (base * 0.15) * np.sin(2 * np.pi * t / 52) + (base / n_weeks) * 0.3 * t

    total_signal = ext_signal + dynamics
    signal_std = total_signal.std() if total_signal.std() > 1 else sigma_base
    noise_sigma = max(
        (noise_fraction / (1 - noise_fraction + 1e-9)) * signal_std,
        sigma_base * 0.1,
    )
    residual = _residual_ar1(phi, noise_sigma)
    amounts = base + total_signal + residual

    if account_type == "sparse":
        mask = np.zeros(n_weeks)
        mask[rng.choice(n_weeks, size=int(n_weeks * 0.45), replace=False)] = 1
        amounts *= mask

    records: list[dict[str, object]] = []
    for date, weekly_net in zip(dates, amounts, strict=True):
        n_credit = int(rng.poisson(lam=4) + 1)
        credit_split = rng.dirichlet(np.ones(n_credit))
        n_debit = int(rng.binomial(3, debit_prob) + 1)
        debit_total = abs(weekly_net) * float(rng.uniform(0.05, 0.20))
        for frac in credit_split:
            records.append({
                "numero_compte": account_id,
                "date_operation": date + pd.Timedelta(days=int(rng.integers(0, 7))),
                "montant": round(float(weekly_net * frac / (1 - debit_prob * 0.15)), 2),
                "type_operation": rng.choice(
                    ["VIREMENT", "CHEQUE", "PRELEVEMENT", "ESPECES"],
                    p=[0.5, 0.2, 0.2, 0.1],
                ),
            })
        debit_split = rng.dirichlet(np.ones(n_debit))
        for frac in debit_split:
            records.append({
                "numero_compte": account_id,
                "date_operation": date + pd.Timedelta(days=int(rng.integers(0, 7))),
                "montant": round(float(-debit_total * frac), 2),
                "type_operation": rng.choice(["PRELEVEMENT", "CHEQUE"], p=[0.7, 0.3]),
            })

    df = pd.DataFrame(records)
    df["sector"] = sector
    df["account_type_true"] = account_type
    return df


def generate_synthetic_dataset(
    n_accounts: int = 200,
    n_weeks: int = 104,
    seed: int = 42,
) -> SyntheticBronze:
    """Génère le triplet bronze complet (transactions, calendrier, indicateurs)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2022-01-03", periods=n_weeks * 7, freq="D")

    calendar_df = generate_calendar_flags(dates)
    indicators_df = generate_sector_indicators(dates, n_weeks, rng=rng)

    dfs: list[pd.DataFrame] = []
    for i in range(n_accounts):
        acc_type = rng.choice(list(ACCOUNT_TYPE_PROBS.keys()), p=list(ACCOUNT_TYPE_PROBS.values()))
        sector = rng.choice(list(SECTOR_PROBS.keys()), p=list(SECTOR_PROBS.values()))
        dfs.append(
            generate_account_series(
                f"ACC_{i:05d}", str(sector), str(acc_type),
                calendar_df, indicators_df,
                n_weeks=n_weeks,
                noise_fraction=NOISE_BY_TYPE[str(acc_type)],
                rng=rng,
            )
        )
    full = pd.concat(dfs, ignore_index=True)
    full["date_operation"] = pd.to_datetime(full["date_operation"])
    full = full.sort_values(["numero_compte", "date_operation"]).reset_index(drop=True)
    return SyntheticBronze(transactions=full, calendar=calendar_df, macro=indicators_df)
