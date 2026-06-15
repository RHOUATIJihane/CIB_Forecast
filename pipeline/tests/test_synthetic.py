"""Tests datagen (jeu de données réduit)."""

from __future__ import annotations

from src.datagen.synthetic import generate_synthetic_dataset


def test_generate_synthetic_dataset_shapes() -> None:
    bronze = generate_synthetic_dataset(n_accounts=5, n_weeks=30, seed=1)
    assert {"numero_compte", "date_operation", "montant", "sector"} <= set(bronze.transactions.columns)
    assert bronze.transactions["numero_compte"].nunique() == 5

    assert {"week", "is_ramadan", "is_payroll_week"} <= set(bronze.calendar.columns)
    assert len(bronze.calendar) >= 30

    assert {"week", "oil_price_z", "masi_index_z"} <= set(bronze.macro.columns)
    assert len(bronze.macro) == 30
