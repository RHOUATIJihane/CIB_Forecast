"""Persistance des modèles (locale + HDFS) avec versionnement par horodatage."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.common.utils import get_logger

LOG = get_logger(__name__)


@dataclass
class ModelArtifact:
    account_id: str
    task: str  # "regression" | "classification"
    feature_mode: str  # "baseline" | "with_externals"
    path: str
    created_utc: str


def _utc_stamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")


def _local_root_for(base_uri: str) -> Path:
    """Convertit un URI HDFS en chemin local équivalent sous /tmp pour mode dev.

    Heuristique : si l'URI commence par ``hdfs://`` mais que `hdfs` n'est pas
    disponible, on retombe sur un dossier local mirroring le chemin.
    """
    if base_uri.startswith("hdfs://"):
        host_path = base_uri.split("hdfs://", 1)[1]
        _, _, path = host_path.partition("/")
        return Path("/tmp/cib_forecast_models") / path
    return Path(base_uri)


def save_model(
    model: Any,
    account_id: str,
    task: str,
    feature_mode: str,
    base_uri: str,
    *,
    run_stamp: str | None = None,
) -> ModelArtifact:
    """Sauvegarde joblib local (et chemin HDFS si la commande `hdfs` est dispo).

    Le projet privilégie une exécution locale (pas de driver pyarrow HDFS) ;
    `train.py` se chargera d'un éventuel `hdfs dfs -put` ultérieur.
    """
    run_stamp = run_stamp or _utc_stamp()
    root = _local_root_for(base_uri) / f"run_{run_stamp}"
    root.mkdir(parents=True, exist_ok=True)
    fname = f"{account_id}__{feature_mode}.joblib"
    local_path = root / fname
    joblib.dump(model, local_path)
    LOG.info("Saved %s/%s model for %s -> %s", task, feature_mode, account_id, local_path)
    return ModelArtifact(
        account_id=account_id,
        task=task,
        feature_mode=feature_mode,
        path=str(local_path),
        created_utc=run_stamp,
    )


def load_model(path: str) -> Any:
    return joblib.load(path)


def serialize_model_bytes(model: Any) -> bytes:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


def latest_run_dir(base_uri: str) -> Path | None:
    root = _local_root_for(base_uri)
    if not root.exists():
        return None
    runs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_"))
    return runs[-1] if runs else None
