"""Alias déprécié — utiliser ``ingest_bronze.py`` (ingestion zone raw)."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path

warnings.warn(
    "bootstrap_bronze → ingest_bronze (workflow ingestion data engineer).",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "ingest_bronze.py"), run_name="__main__")
