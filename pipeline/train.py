"""Entrée d'entraînement (RF régression + logistique classification, par compte)."""

from __future__ import annotations

from src.common.utils import get_logger
from src.ml.train_runner import run_training


def main() -> None:
    log = get_logger(__name__)
    metrics = run_training()
    log.info("Training done. %d rows of metrics.", len(metrics))


if __name__ == "__main__":
    main()
