"""Politique v7 — catégories encapsulées (EXP → seuils métriques → règle)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from src.policy_rules import CLASSIFICATION_MODEL

if TYPE_CHECKING:
    from configparser import SectionProxy

POLICY_VERSION_V7: Final[str] = "v7-encapsulated-3pct"

# Chaque catégorie → une EXP source → une décision propre (champion + externes).
CATEGORY_V7_SOURCES: Final[list[dict[str, str]]] = [
    {
        "account_category": "E_short_history",
        "policy_rule_id": "E_short",
        "source_exp": "EXP1_C_short",
        "source_kind": "condition",
        "source_value": "C_short",
    },
    {
        "account_category": "D_noisy",
        "policy_rule_id": "D_noisy",
        "source_exp": "EXP1_B_noisy",
        "source_kind": "condition",
        "source_value": "B_noisy",
    },
    {
        "account_category": "B_seasonal",
        "policy_rule_id": "B_seasonal",
        "source_exp": "EXP3_seasonal",
        "source_kind": "account_type",
        "source_value": "seasonal",
    },
    {
        "account_category": "D_sparse",
        "policy_rule_id": "D_sparse",
        "source_exp": "EXP6_sparse",
        "source_kind": "condition",
        "source_value": "sparse",
    },
    {
        "account_category": "F_flat",
        "policy_rule_id": "F_flat",
        "source_exp": "EXP6_flat",
        "source_kind": "condition",
        "source_value": "flat",
    },
    {
        "account_category": "C_trending",
        "policy_rule_id": "C_trending",
        "source_exp": "EXP2_C_level_shift",
        "source_kind": "condition",
        "source_value": "C_level_shift_w70",
    },
    {
        "account_category": "A_regular_stable",
        "policy_rule_id": "A_regular",
        "source_exp": "EXP1_A_clean",
        "source_kind": "condition",
        "source_value": "A_clean",
    },
    {
        "account_category": "D_irregular",
        "policy_rule_id": "D_irregular",
        "source_exp": "EXP3_irregular",
        "source_kind": "condition",
        "source_value": "irregular",
    },
]

# Table intégrée (surchargeable par CSV export notebook).
_BUILTIN_POLICY: Final[list[dict[str, object]]] = [
    {
        "account_category": "E_short_history",
        "activate_externals": False,
        "regression_model": "lgbm",
        "policy_rule_id": "E_short",
        "source_exp": "EXP1_C_short",
        "lift_med_champion_pct": 0.0,
    },
    {
        "account_category": "C_trending",
        "activate_externals": False,
        "regression_model": "rf",
        "policy_rule_id": "C_trending",
        "source_exp": "EXP2_C_level_shift",
        "lift_med_champion_pct": 2.13,
    },
    {
        "account_category": "B_seasonal",
        "activate_externals": False,
        "regression_model": "lgbm",
        "policy_rule_id": "B_seasonal",
        "source_exp": "EXP3_seasonal",
        "lift_med_champion_pct": -0.54,
    },
    {
        "account_category": "A_regular_stable",
        "activate_externals": True,
        "regression_model": "rf",
        "policy_rule_id": "A_regular",
        "source_exp": "EXP1_A_clean",
        "lift_med_champion_pct": 5.13,
    },
    {
        "account_category": "D_noisy",
        "activate_externals": False,
        "regression_model": "ridge",
        "policy_rule_id": "D_noisy",
        "source_exp": "EXP1_B_noisy",
        "lift_med_champion_pct": -0.37,
    },
    {
        "account_category": "D_irregular",
        "activate_externals": True,
        "regression_model": "ridge",
        "policy_rule_id": "D_irregular",
        "source_exp": "EXP3_irregular",
        "lift_med_champion_pct": 5.25,
    },
    {
        "account_category": "D_sparse",
        "activate_externals": False,
        "regression_model": "ridge",
        "policy_rule_id": "D_sparse",
        "source_exp": "EXP6_sparse",
        "lift_med_champion_pct": -2.11,
    },
    {
        "account_category": "F_flat",
        "activate_externals": False,
        "regression_model": "rf",
        "policy_rule_id": "F_flat",
        "source_exp": "EXP6_flat",
        "lift_med_champion_pct": -1.53,
    },
]


def build_policy_v7_encapsulated(
    experiment_df: pd.DataFrame,
    *,
    min_lift_pct: float = 3.0,
) -> pd.DataFrame:
    """Construit la politique v7 : 1 catégorie = 1 EXP source = 1 décision."""
    ml_models = ("ridge", "rf", "lgbm")
    df = experiment_df.copy()
    if "experiment_role" in df.columns:
        df = df[df["experiment_role"] == "calibration"]
    df = df[df["model_name"].isin(ml_models)]

    idx = ["name", "condition", "model_name"]
    base = df[df["model_type"] == "baseline"][idx + ["rmse"]].rename(columns={"rmse": "rmse_baseline"})
    ext = df[df["model_type"] == "with_externals"][idx + ["rmse"]].rename(
        columns={"rmse": "rmse_with_externals"}
    )
    merged = base.merge(ext, on=idx, how="inner")
    if merged.empty:
        return pd.DataFrame(_BUILTIN_POLICY)

    merged["rmse_lift_pct"] = (
        (merged["rmse_baseline"] - merged["rmse_with_externals"]) / merged["rmse_baseline"] * 100
    )
    rows: list[dict[str, object]] = []
    for spec in CATEGORY_V7_SOURCES:
        if spec["source_kind"] == "condition":
            sub = merged[merged["condition"] == spec["source_value"]]
        else:
            # Même ``name`` peut exister sous plusieurs conditions (EXP2) :
            # on joint name + condition pour l'EXP source.
            scope = df.loc[df["account_type"] == spec["source_value"], ["name", "condition"]].drop_duplicates()
            sub = merged.merge(scope, on=["name", "condition"], how="inner")
        n_accounts = int(sub["name"].nunique()) if not sub.empty else 0

        if sub.empty:
            fallback = next(
                (r for r in _BUILTIN_POLICY if r["account_category"] == spec["account_category"]),
                None,
            )
            if fallback is None:
                continue
            rows.append(
                {
                    "account_category": spec["account_category"],
                    "activate_externals": fallback["activate_externals"],
                    "champion_model": fallback["regression_model"],
                    "policy_rule_id": spec["policy_rule_id"],
                    "source_exp": spec["source_exp"],
                    "lift_med_champion_pct": fallback.get("lift_med_champion_pct"),
                    "n_accounts_v6": n_accounts,
                    "policy_source": "v7_encapsulated",
                }
            )
            continue

        stats = (
            sub.groupby("model_name", dropna=False)
            .agg(
                rmse_with_externals=("rmse_with_externals", "median"),
                lift_med=("rmse_lift_pct", "median"),
            )
            .reset_index()
        )
        champ_row = stats.loc[stats["rmse_with_externals"].idxmin()]
        champion = str(champ_row["model_name"]).lower()
        lift_med = float(champ_row["lift_med"])
        rows.append(
            {
                "account_category": spec["account_category"],
                "activate_externals": bool(lift_med >= min_lift_pct),
                "champion_model": champion,
                "policy_rule_id": spec["policy_rule_id"],
                "source_exp": spec["source_exp"],
                "lift_med_champion_pct": round(lift_med, 2),
                "n_accounts_v6": n_accounts,
                "policy_source": "v7_encapsulated",
            }
        )
    return pd.DataFrame(rows)


def resolve_policy_v7_csv_path(cfg: SectionProxy) -> Path:
    explicit = (cfg.get("policy_v7_csv") or "").strip()
    if explicit:
        return Path(explicit)
    exports = Path(
        cfg.get(
            "policy_exports_dir",
            "/home/jihane/cib_project/cib_project_outputs/cib_experiment_outputs",
        )
    )
    candidate = exports / "policy_v7_encapsulated.csv"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Policy v7 CSV introuvable ({candidate}). Utilisation de la table intégrée."
    )


def load_category_policy_table(cfg: SectionProxy) -> pd.DataFrame:
    """Charge la politique v7 (CSV si présent, sinon table intégrée)."""
    try:
        path = resolve_policy_v7_csv_path(cfg)
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(_BUILTIN_POLICY)

    if "regression_model" not in df.columns and "champion_model" in df.columns:
        df = df.rename(columns={"champion_model": "regression_model"})
    required = {"account_category", "activate_externals", "regression_model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes politique v7: {missing}")

    if "policy_rule_id" not in df.columns:
        df["policy_rule_id"] = df["account_category"]
    if "policy_source" not in df.columns:
        df["policy_source"] = "v7_encapsulated"

    out = df.copy()
    out["activate_externals"] = out["activate_externals"].map(
        lambda x: str(x).lower() in {"true", "1", "yes"}
    )
    out["regression_model"] = out["regression_model"].astype(str).str.strip().str.lower()
    return out.drop_duplicates(subset=["account_category"], keep="last")
