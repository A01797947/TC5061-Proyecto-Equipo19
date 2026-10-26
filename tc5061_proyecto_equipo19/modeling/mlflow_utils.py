"""Contexto reproducible para todas las corridas de MiBici.

Las corridas de modelos se crean en el experimento del equipo. Cuando se inicia
con MLflow Projects se conserva el identificador del run del Project como tag;
la corrida de entrenamiento se registra separadamente para garantizar que quede
en el experimento del equipo incluso sin --experiment-name.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import subprocess

from loguru import logger
import mlflow
import mlflow.sklearn

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.dataset import leer_manifiesto


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cfg.PROJ_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido"


def configurar_experimento() -> None:
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    existing = mlflow.get_experiment_by_name(cfg.MLFLOW_EXPERIMENT_NAME)
    if existing is None:
        mlflow.create_experiment(
            cfg.MLFLOW_EXPERIMENT_NAME,
            artifact_location=cfg.MLFLOW_ARTIFACT_LOCATION,
        )
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)


def contexto_comun() -> tuple[dict, dict]:
    if not cfg.MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Falta {cfg.MANIFEST_PATH}. Ejecuta primero: mlflow run . -e ingest --env-manager local"
        )
    manifest = leer_manifiesto()
    params = {
        "fecha_inicio": cfg.FECHA_INICIO,
        "fecha_fin": cfg.FECHA_FIN,
        "fin_entrenamiento": cfg.FIN_ENTRENAMIENTO,
        "inicio_validacion": cfg.INICIO_VALIDACION,
        "fin_validacion": cfg.FIN_VALIDACION,
        "inicio_prueba": cfg.INICIO_PRUEBA,
        "data_version": manifest["data_version"],
        "data_schema_version": manifest["schema_version"],
        "meses": len(manifest["trips"]),
        "viajes_manifiesto": sum(t["records"] for t in manifest["trips"]),
        "random_seed": cfg.RANDOM_SEED,
    }
    tags = {
        "equipo": cfg.EQUIPO,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no"))),
        "data_version": manifest["data_version"],
    }
    return params, tags


@contextmanager
def _new_mlflow_run(name: str, *, nested: bool = False):
    """Evita que MLFLOW_RUN_ID del Project reabra el run equivocado.

    Cuando nested=True debe existir una corrida padre activa. En ese caso no
    eliminamos el contexto del padre, únicamente la variable heredada del
    proceso de MLflow Project durante la apertura de la corrida hija.
    """
    if nested and mlflow.active_run() is None:
        raise RuntimeError("nested=True requiere una corrida padre activa")
    project_id = os.environ.pop("MLFLOW_RUN_ID", None)
    try:
        with mlflow.start_run(run_name=name, nested=nested) as run:
            if project_id:
                mlflow.set_tag("mlflow_project_run_id", project_id)
            yield run
    finally:
        if project_id is not None:
            os.environ["MLFLOW_RUN_ID"] = project_id


@contextmanager
def iniciar_run(
    nombre: str, pregunta: str, tipo: str, params: dict | None = None,
    nested: bool = False,
):
    comunes, tags = contexto_comun()
    configurar_experimento()
    with _new_mlflow_run(nombre, nested=nested) as run:
        mlflow.set_tags({
            **tags,
            "pregunta": pregunta,
            "tipo_modelo": tipo,
            "question": pregunta.lower(),
            "stage_of_work": tipo,
        })
        mlflow.log_params({**comunes, **(params or {})})
        mlflow.log_artifact(str(cfg.MANIFEST_PATH), artifact_path="data")
        logger.info("Corrida {} ({})", nombre, run.info.run_id)
        yield run


def log_modelo_sklearn(modelo, ejemplo):
    """Guarda modelo con firma y ejemplo y persiste el URI de MLflow 3."""
    from mlflow.models import infer_signature
    prediction = modelo.predict(ejemplo)
    info = mlflow.sklearn.log_model(
        sk_model=modelo,
        name="modelo",
        signature=infer_signature(ejemplo, prediction),
        input_example=ejemplo,
        # El modelo propio contiene el tipo NumPy reportado por skops.
        # Se autoriza específicamente este tipo, sin desactivar skops.
        serialization_format="cloudpickle",
        #skops_trusted_types=["numpy.dtype","sklearn.tree._tree.Tree"],
    )

    mlflow.set_tag("logged_model_uri", info.model_uri)
    return info


def registrar_run(nombre: str, pregunta: str, tipo: str, params=None,
                  metricas=None, artefactos=None, modelo=None, input_example=None):
    with iniciar_run(nombre, pregunta, tipo, params=params) as run:
        for key, val in (metricas or {}).items():
            if isinstance(val, (int, float)) and math.isfinite(val):
                mlflow.log_metric(key, val)
        for file in artefactos or []:
            file = Path(file)
            if file.is_file():
                mlflow.log_artifact(str(file))
            elif file.is_dir():
                mlflow.log_artifacts(str(file))
        if modelo is not None:
            if input_example is None:
                raise ValueError("Se requiere input_example para registrar firma del modelo")
            log_modelo_sklearn(modelo, input_example)
        return run.info.run_id
