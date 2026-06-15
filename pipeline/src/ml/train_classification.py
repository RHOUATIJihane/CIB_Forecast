"""Classification (3 classes) par compte — logistique baseline."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.metrics import classification_scores


def make_classifier_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )


def walk_forward_eval(
    x: np.ndarray,
    y_labels: np.ndarray,
    *,
    n_splits: int = 3,
    test_size: int = 4,
) -> Optional[dict[str, float | object]]:
    """Validation walk-forward identique à la régression — métriques classif."""
    n = len(y_labels)
    if n < 14:
        return None

    acc, f1_macro, f1_weighted = [], [], []
    last_model: Pipeline | None = None
    template = make_classifier_pipeline()

    for split in range(n_splits):
        test_end = n - split * test_size
        test_start = test_end - test_size
        if test_start < 10:
            break

        x_train, x_test = x[:test_start], x[test_start:test_end]
        y_train, y_test = y_labels[:test_start], y_labels[test_start:test_end]

        if len(np.unique(y_train)) < 2:
            continue

        model = clone(template)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        last_model = model

        scores = classification_scores(y_test, y_pred)
        acc.append(scores["accuracy"])
        f1_macro.append(scores["f1_macro"])
        f1_weighted.append(scores["f1_weighted"])

    if not acc or last_model is None:
        return None

    return {
        "accuracy": float(np.mean(acc)),
        "f1_macro": float(np.mean(f1_macro)),
        "f1_weighted": float(np.mean(f1_weighted)),
        "model": last_model,
        "y_test": y_labels[-test_size:],
        "y_pred": last_model.predict(x[-test_size:]),
    }


def fit_final_classifier(x: np.ndarray, y_labels: np.ndarray) -> Pipeline:
    model = make_classifier_pipeline()
    model.fit(x, y_labels)
    return model
