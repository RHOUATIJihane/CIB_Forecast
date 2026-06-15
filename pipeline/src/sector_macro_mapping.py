"""Mapping secteur (PFE) ou prefix NAF (prod) → variables macro externes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import pandas as pd
    from pyspark.sql import DataFrame

CALENDAR_EXTERNAL_KEYS: Final[tuple[str, ...]] = (
    "is_ramadan",
    "is_eid_alfitr",
    "is_month_end_week",
    "is_payroll_week",
    "is_quarter_end",
)

MACRO_EXTERNAL_KEYS: Final[tuple[str, ...]] = (
    "oil_price_z",
    "commodity_index_z",
    "masi_index_z",
    "realestate_index_z",
)


def default_mapping_csv_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    return root / "data" / "reference" / "sector_macro_mapping.csv"


def load_sector_macro_mapping_pandas(csv_path: Path | None = None) -> pd.DataFrame:
    """Charge le CSV de référence (tests / fallback local)."""
    import pandas as pd

    path = csv_path or default_mapping_csv_path()
    df = pd.read_csv(path)
    required = {"sector", "macro_primary", "macro_secondary", "include_calendar", "priority"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sector_macro_mapping.csv missing columns: {missing}")
    df["sector"] = df["sector"].astype(str).str.strip().str.lower()
    for col in ("macro_primary", "macro_secondary"):
        df[col] = df[col].fillna("").astype(str).str.strip()
        df.loc[df[col] == "", col] = ""
    df["include_calendar"] = df["include_calendar"].astype(bool)
    df["priority"] = df["priority"].astype(int)
    return df.sort_values("priority")


def resolve_macro_keys(
    sector: str,
    mapping_df: pd.DataFrame,
) -> dict[str, object]:
    """Retourne macro_primary, macro_secondary et la liste externals_for_model."""
    sector_norm = str(sector).strip().lower()
    rows = mapping_df[mapping_df["sector"] == sector_norm]
    if rows.empty:
        rows = mapping_df[mapping_df["sector"] == "*"]
    if rows.empty:
        return {
            "macro_primary": "",
            "macro_secondary": "",
            "include_calendar": True,
            "externals_for_model": ",".join(CALENDAR_EXTERNAL_KEYS),
        }
    row = rows.sort_values("priority").iloc[0]
    keys: list[str] = []
    for col in ("macro_primary", "macro_secondary"):
        val = str(row[col]).strip()
        if val and val.lower() != "nan":
            keys.append(val)
    if bool(row["include_calendar"]):
        keys.extend(CALENDAR_EXTERNAL_KEYS)
    # dédupliquer en conservant l'ordre
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return {
        "macro_primary": str(row["macro_primary"]).strip() if str(row["macro_primary"]).strip().lower() != "nan" else "",
        "macro_secondary": str(row["macro_secondary"]).strip() if str(row["macro_secondary"]).strip().lower() != "nan" else "",
        "include_calendar": bool(row["include_calendar"]),
        "externals_for_model": ",".join(ordered),
    }


def parse_externals_for_model(value: str | None) -> list[str]:
    if not value or str(value).strip().lower() in {"nan", "none", ""}:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def build_account_macro_assignment_pandas(
    accounts_df: "pd.DataFrame",
    mapping_df: "pd.DataFrame",
    *,
    account_col: str = "numero_compte",
    sector_col: str = "sector",
) -> "pd.DataFrame":
    """Assigne les macros par compte à partir du secteur synthétique."""
    import pandas as pd

    rows: list[dict[str, object]] = []
    for _, acc in accounts_df.drop_duplicates(subset=[account_col]).iterrows():
        resolved = resolve_macro_keys(str(acc[sector_col]), mapping_df)
        rows.append(
            {
                account_col: acc[account_col],
                sector_col: acc[sector_col],
                **resolved,
            }
        )
    return pd.DataFrame(rows)


