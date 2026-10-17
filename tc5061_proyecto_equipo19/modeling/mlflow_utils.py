"""Utilidades comunes para registrar runs en MLflow Tracking.

Centralizan lo que todo run del experimento debe llevar para ser reproducible:
rango de fechas y particiones, semilla, versión de datos del manifiesto y las
etiquetas de equipo, pregunta y commit de Git.

Ejemplo:

    from tc5061_proyecto_equipo19.modeling.mlflow_utils import configurar_experimento, iniciar_run

    configurar_experimento()
    with iniciar_run("P2-baseline-regla-historica", pregunta="P2", tipo="baseline",
                     params={"modelo": "regla"}):
        mlflow.log_metric("val_average_precision", 0.42)
"""

from __future__ import annotations

from contextlib import contextmanager
import subprocess

from loguru import logger
import mlflow

from tc5061_proyecto_equipo19.config import (
    EQUIPO,
    FECHA_FIN,
    FECHA_INICIO,
    FIN_ENTRENAMIENTO,
    FIN_VALIDACION,
    INICIO_PRUEBA,
    INICIO_VALIDACION,
    MANIFEST_PATH,
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
    RANDOM_SEED,
)
from tc5061_proyecto_equipo19.dataset import leer_manifiesto


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJ_ROOT, text=True).strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "desconocido"


def configurar_experimento() -> None:
    """Apunta MLflow al tracking URI de config y activa el experimento del equipo."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_LOCATION
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def contexto_comun() -> tuple[dict, dict]:
    """Parámetros y etiquetas que comparten todos los runs del experimento."""
    manifest = leer_manifiesto()
    params = {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "fin_entrenamiento": FIN_ENTRENAMIENTO,
        "inicio_validacion": INICIO_VALIDACION,
        "fin_validacion": FIN_VALIDACION,
        "inicio_prueba": INICIO_PRUEBA,
        "data_version": manifest["data_version"],
        "data_schema_version": manifest["schema_version"],
        "meses": len(manifest["trips"]),
        "viajes_manifiesto": sum(t["records"] for t in manifest["trips"]),
        "random_seed": RANDOM_SEED,
    }
    tags = {
        "equipo": EQUIPO,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no"))),
        "data_version": manifest["data_version"],
    }
    return params, tags


@contextmanager
def iniciar_run(nombre: str, pregunta: str, tipo: str, params: dict | None = None):
    """Abre un run con los parámetros y etiquetas comunes más los propios del modelo.

    También adjunta el manifiesto de datos como artefacto, para que cada resultado
    quede ligado a los archivos exactos con que se obtuvo.
    """
    comunes, tags = contexto_comun()
    with mlflow.start_run(run_name=nombre) as run:
        mlflow.set_tags({**tags, "pregunta": pregunta, "tipo_modelo": tipo})
        mlflow.log_params({**comunes, **(params or {})})
        mlflow.log_artifact(str(MANIFEST_PATH), artifact_path="datos")
        logger.info(f"Run {nombre} ({run.info.run_id})")
        yield run
