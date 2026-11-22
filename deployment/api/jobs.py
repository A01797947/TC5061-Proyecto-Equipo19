"""Trabajo asíncrono de P2 y acceso compartido a los modelos.

El worker de RQ ejecuta cada trabajo en un proceso hijo (fork). Por eso los modelos
se cargan una sola vez en el proceso padre y los hijos los heredan.
"""
from __future__ import annotations

import os
from pathlib import Path

from deployment.api.inference import ModelStore

QUEUE_NAME = "predicciones"
RESULT_TTL = 300    # segundos que se conserva un resultado correcto
FAILURE_TTL = 300   # segundos que se conserva un trabajo fallido
JOB_TIMEOUT = 30    # segundos máximos por predicción

_STORE: ModelStore | None = None


def get_store() -> ModelStore:
    """Carga los snapshots la primera vez y reutiliza el mismo objeto después."""
    global _STORE
    if _STORE is None:
        _STORE = ModelStore.load_snapshot(Path(os.getenv("MIBICI_MODEL_DIR", "/app/model_store")))
    return _STORE


def predict_p2_job(values: dict) -> dict:
    """Misma lógica que POST /v1/predict/p2, pero ejecutada por el worker."""
    store = get_store()
    model = store.info("p2")
    value, frame = store.predict("p2", values)
    label = int(value)
    if label not in (0, 1) or value != label:
        raise RuntimeError("El clasificador devolvió una clase inválida")
    return {
        "modelo": model["modelo"],
        "version": model["version"],
        "riesgo": label,
        "probabilidad_riesgo": store.risk_probability(frame),
    }