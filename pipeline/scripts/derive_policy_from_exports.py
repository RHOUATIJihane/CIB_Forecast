#!/usr/bin/env python3
"""Dérive la policy v2.0 à partir des exports notebook (``cib_project_outputs``).

Produit sous ``{local_base}/silver/`` :
- ``policy_sector_cv_lookup.csv`` — recommandation par secteur × bande CV (données RF/irregular)
- ``policy_experiment_regression_oracle.csv`` — oracle lift>0 par compte expérience (régression)
- ``policy_experiment_classification_oracle.csv`` — oracle par compte expérience (classification)
- ``policy_account_overrides.csv`` — optionnel, si ``--quality-metrics`` fourni (jointure secteur×CV)
- ``policy_derivation_summary.json`` — statistiques et validation des règles v2.0

Usage :
    python scripts/derive_policy_from_exports.py
    python scripts/derive_policy_from_exports.py --write-account-policy \\
        --quality-metrics /path/to/account_quality_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.utils import load_config  # noqa: E402
from src.policy_rules import (  # noqa: E402
    CLASSIFICATION_MODEL,
    CV_BAND_HIGH,
    CV_BAND_LOW,
    LOOKUP_MIN_MEAN_LIFT_PCT,
    LOOKUP_MIN_PCT_POSITIVE,
    POLICY_VERSION,
    REGRESSION_MODEL,
    REGRESSION_PRIORITY_SECTORS,
    cv_to_bucket,
    parse_reg_from_notes,
    use_externals_classification_oracle,
    use_externals_classification_v2,
    use_externals_regression_oracle,
    use_externals_regression_v2,
)

DEFAULT_EXPORTS = Path("/home/jihane/cib_project/cib_project_outputs/cib_experiment_outputs")


def _load_regression_lifts(exports_dir: Path) -> pd.DataFrame:
    path = exports_dir / "lift_by_account_condition_model.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing regression lifts: {path}")
    lift = pd.read_csv(path)
    mask = (lift["condition"] == "irregular") & (lift["model_name"] == "rf")
    irf = lift.loc[mask].copy()
    irf["reg"] = irf["notes"].map(parse_reg_from_notes)
    irf["cv_bucket"] = irf["reg"].map(lambda r: cv_to_bucket(r) if r is not None else None)
    return irf


def build_sector_cv_lookup(irf: pd.DataFrame) -> pd.DataFrame:
    """Table lookup : activer si lift moyen > 0 et part des comptes gagnants >= seuil."""
    rows: list[dict] = []
    grouped = irf.dropna(subset=["cv_bucket"]).groupby(["sector", "cv_bucket"], observed=True)
    for (sector, bucket), grp in grouped:
        n = len(grp)
        mean_lift = float(grp["rmse_lift_pct"].mean())
        pct_pos = float((grp["rmse_lift_pct"] > 0).mean())
        use_reg = mean_lift > LOOKUP_MIN_MEAN_LIFT_PCT and pct_pos >= LOOKUP_MIN_PCT_POSITIVE
        rows.append(
            {
                "sector": sector,
                "cv_bucket": bucket,
                "cv_low": _bucket_low(bucket),
                "cv_high": _bucket_high(bucket),
                "n_accounts": n,
                "mean_rmse_lift_pct": round(mean_lift, 4),
                "pct_positive": round(pct_pos, 4),
                "use_externals_regression": use_reg,
                "use_externals_classification": False,
                "rationale": (
                    f"irregular+rf n={n} mean_lift={mean_lift:.2f}% pct_pos={pct_pos:.1%}"
                ),
            }
        )
    return pd.DataFrame(rows)


def _bucket_low(bucket: str) -> float:
    if bucket == "<0.30":
        return 0.0
    if bucket == "[0.30,0.50)":
        return 0.30
    if bucket == "[0.50,0.80)":
        return 0.50
    return 0.80


def _bucket_high(bucket: str) -> float:
    if bucket == "<0.30":
        return 0.30
    if bucket == "[0.30,0.50)":
        return 0.50
    if bucket == "[0.50,0.80)":
        return 0.80
    return 9.99


def build_regression_oracle(irf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment_account": irf["name"],
            "sector": irf["sector"],
            "reg": irf["reg"],
            "cv_bucket": irf["cv_bucket"],
            "rmse_lift_pct": irf["rmse_lift_pct"],
            "use_externals_regression": irf["rmse_lift_pct"].map(use_externals_regression_oracle),
            "policy_source": "oracle_lift_gt_0",
        }
    )


def build_classification_oracle(exports_dir: Path) -> pd.DataFrame:
    path = exports_dir / "classification_raw_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing classification results: {path}")
    clf = pd.read_csv(path)
    clf = clf[clf["model_name"] == "logistic"]
    base = clf[clf["model_type"] == "baseline"].set_index("name")["accuracy"]
    ext = clf[clf["model_type"] == "with_externals"].set_index("name")["accuracy"]
    common = base.index.intersection(ext.index)
    sectors = clf[clf["model_type"] == "baseline"].set_index("name")["sector"]
    return pd.DataFrame(
        {
            "experiment_account": common,
            "sector": sectors.reindex(common).values,
            "accuracy_baseline": base[common].values,
            "accuracy_with_externals": ext[common].values,
            "accuracy_lift": (ext[common] - base[common]).values,
            "use_externals_classification": [
                use_externals_classification_oracle(float(ext[n] - base[n])) for n in common
            ],
            "policy_source": "oracle_accuracy_lift_gt_0",
        }
    )


def _evaluate_v2_rules(irf: pd.DataFrame) -> dict:
    flags = irf.apply(
        lambda r: use_externals_regression_v2(str(r["sector"]), float(r["reg"] or 0)),
        axis=1,
    )
    on = irf[flags]
    off = irf[~flags]
    return {
        "regression_v2_on_n": int(len(on)),
        "regression_v2_off_n": int(len(off)),
        "regression_v2_on_mean_lift_pct": float(on["rmse_lift_pct"].mean()) if len(on) else None,
        "regression_v2_off_mean_lift_pct": float(off["rmse_lift_pct"].mean()) if len(off) else None,
        "regression_v2_on_pct_positive": float((on["rmse_lift_pct"] > 0).mean()) if len(on) else None,
        "regression_v2_off_pct_positive": float((off["rmse_lift_pct"] > 0).mean()) if len(off) else None,
        "regression_oracle_pct_positive": float((irf["rmse_lift_pct"] > 0).mean()),
    }


def build_account_policy_from_quality(
    quality: pd.DataFrame,
    lookup: pd.DataFrame,
    *,
    mode: str,
) -> pd.DataFrame:
    """Construit ``account_policy`` pour les comptes prod (``numero_compte``)."""
    quality = quality.copy()
    quality["cv_bucket"] = quality["cv_cashflow"].map(cv_to_bucket)

    rows = []
    for _, row in quality.iterrows():
        sector = str(row["sector"])
        cv = float(row["cv_cashflow"])
        bucket = cv_to_bucket(cv)
        reg_v2 = use_externals_regression_v2(sector, cv)
        clf_v2 = use_externals_classification_v2(sector)

        use_reg, use_clf, source = reg_v2, clf_v2, "rules_v2"
        if mode in {"lookup", "hybrid"}:
            match = lookup[(lookup["sector"] == sector) & (lookup["cv_bucket"] == bucket)]
            if len(match):
                use_reg = bool(match.iloc[0]["use_externals_regression"])
                use_clf = bool(match.iloc[0]["use_externals_classification"])
                source = "lookup_sector_cv"

        rows.append(
            {
                "numero_compte": row["numero_compte"],
                "sector": sector,
                "cv_cashflow": cv,
                "predictability_score": row.get("predictability_score"),
                "composite_score": row.get("composite_score"),
                "use_externals_regression": use_reg,
                "use_externals_classification": use_clf,
                "regression_model": REGRESSION_MODEL,
                "classification_model": CLASSIFICATION_MODEL,
                "policy_version": POLICY_VERSION,
                "policy_source": source,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dérive policy v2.0 depuis cib_project_outputs")
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=DEFAULT_EXPORTS,
        help="Répertoire cib_experiment_outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Répertoire de sortie (défaut: {local_base}/silver)",
    )
    parser.add_argument(
        "--quality-metrics",
        type=Path,
        default=None,
        help="CSV account_quality_metrics pour générer account_policy",
    )
    parser.add_argument(
        "--write-account-policy",
        action="store_true",
        help="Écrit account_policy.csv si --quality-metrics est fourni",
    )
    parser.add_argument(
        "--mode",
        choices=["rules", "lookup", "hybrid"],
        default="hybrid",
        help="Mode pour account_policy (--write-account-policy)",
    )
    args = parser.parse_args()

    cfg = load_config()
    out_dir = args.output_dir or (Path(cfg["local_base"]) / "silver")
    out_dir.mkdir(parents=True, exist_ok=True)

    irf = _load_regression_lifts(args.exports_dir)
    lookup = build_sector_cv_lookup(irf)
    reg_oracle = build_regression_oracle(irf)
    clf_oracle = build_classification_oracle(args.exports_dir)
    summary = {
        "policy_version": POLICY_VERSION,
        "exports_dir": str(args.exports_dir),
        "output_dir": str(out_dir),
        "regression_reference": "lift_by_account_condition_model.csv irregular+rf",
        "classification_reference": "classification_raw_results.csv logistic",
        "v2_rules": {
            "regression": f"sector in {sorted(REGRESSION_PRIORITY_SECTORS)} OR cv in [{CV_BAND_LOW}, {CV_BAND_HIGH})",
            "classification": "default baseline (no externals)",
        },
        "evaluation": _evaluate_v2_rules(irf),
        "sector_cv_lookup_rows": len(lookup),
        "regression_oracle_accounts": len(reg_oracle),
        "classification_oracle_accounts": len(clf_oracle),
    }

    lookup_path = out_dir / "policy_sector_cv_lookup.csv"
    lookup.to_csv(lookup_path, index=False)
    reg_oracle.to_csv(out_dir / "policy_experiment_regression_oracle.csv", index=False)
    clf_oracle.to_csv(out_dir / "policy_experiment_classification_oracle.csv", index=False)

    overrides_note = (
        "Pour overrides par numero_compte (métriques prod), fournir un CSV avec colonnes "
        "numero_compte, use_externals_regression, use_externals_classification."
    )
    summary["overrides_note"] = overrides_note

    summary_path = out_dir / "policy_derivation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {lookup_path} ({len(lookup)} rows)")
    print(f"Wrote {out_dir / 'policy_experiment_regression_oracle.csv'} ({len(reg_oracle)} rows)")
    print(f"Wrote {out_dir / 'policy_experiment_classification_oracle.csv'} ({len(clf_oracle)} rows)")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary["evaluation"], indent=2))

    if args.write_account_policy:
        if args.quality_metrics is None or not args.quality_metrics.is_file():
            raise SystemExit("--write-account-policy requires --quality-metrics <csv>")
        quality = pd.read_csv(args.quality_metrics)
        policy = build_account_policy_from_quality(quality, lookup, mode=args.mode)
        policy_path = out_dir / "account_policy.csv"
        policy.to_csv(policy_path, index=False)
        print(f"Wrote {policy_path} ({len(policy)} accounts, mode={args.mode})")

    print("\nRun policy job with policy_mode=hybrid to apply lookup + rules in Spark.")


if __name__ == "__main__":
    main()
