"""Entrée d'inférence batch (scoring sur la dernière semaine disponible)."""

from __future__ import annotations

from src.common.utils import get_logger
from src.ml.inference_runner import run_inference


def main() -> None:
    log = get_logger(__name__)
    predictions = run_inference()
    log.info("Inference done. %d rows of predictions.", len(predictions))


if __name__ == "__main__":
    main()
