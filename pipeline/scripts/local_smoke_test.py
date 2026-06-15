"""Smoke test bout-en-bout sans Spark (échantillon réduit).

Étapes :
1. Générer un petit dataset synthétique (10 comptes × 60 semaines).
2. Sauvegarder les CSV bronze dans ``local_base``.
3. Vérifier que ``build_features`` et l'entraînement par compte fonctionnent.

À utiliser AVANT de brancher HDFS + Spark pour valider la logique métier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.common.utils import get_logger, load_config
from src.datagen.synthetic import generate_synthetic_dataset
from src.ml.build_features import build_features, create_classification_labels
from src.ml.train_classification import walk_forward_eval as walk_clf
from src.ml.train_regression import walk_forward_eval as walk_reg

LOG = get_logger(__name__)


def main() -> int:
    cfg = load_config()
    base = Path(cfg["local_base"]) / "bronze"
    base.mkdir(parents=True, exist_ok=True)

    LOG.info("Generating tiny synthetic dataset (10 accounts, 60 weeks)...")
    bronze = generate_synthetic_dataset(n_accounts=10, n_weeks=60, seed=42)
    bronze.transactions.to_csv(base / "transactions_raw_smoke.csv", index=False)
    bronze.calendar.to_csv(base / "calendar_smoke.csv", index=False)
    bronze.macro.to_csv(base / "macro_smoke.csv", index=False)

    weekly = (
        bronze.transactions.assign(
            week=lambda d: pd.to_datetime(d["date_operation"]).dt.to_period("W").dt.start_time
        )
        .groupby(["numero_compte", "week"])["montant"]
        .sum()
        .reset_index(name="total_amount")
        .sort_values(["numero_compte", "week"])
    )

    n_trained_reg = 0
    n_trained_clf = 0
    for _, g in weekly.groupby("numero_compte"):
        series = g["total_amount"].to_numpy(dtype=float)
        x, y = build_features(series)
        if len(x) < 14:
            continue
        if walk_reg(x, y) is not None:
            n_trained_reg += 1
        labels = create_classification_labels(series)[4:]
        if walk_clf(x, labels) is not None:
            n_trained_clf += 1

    LOG.info("Smoke test OK — regression: %d / classification: %d", n_trained_reg, n_trained_clf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
