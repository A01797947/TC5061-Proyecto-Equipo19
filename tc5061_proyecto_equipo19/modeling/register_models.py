"""Registro explícito de ganadores en MLflow 3 con linaje y alias champion.

No elige automáticamente usando test: primero comparar en validación y justificar.
"""
from __future__ import annotations

import argparse

import mlflow
from mlflow import MlflowClient

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.modeling.mlflow_utils import configurar_experimento

MODEL_NAMES = {
    "p1": "mibici-equipo19-p1-demanda",
    "p2": "mibici-equipo19-p2-riesgo",
    "p3": "mibici-equipo19-p3-grupos",
}
DESCRIPTIONS = {
    "p1": "Predicción de demanda diaria por estación para planificación operativa.",
    "p2": "Predicción de riesgo de quedar sin bicicletas o anclajes dentro de tres horas.",
    "p3": "Agrupamiento de perfiles de uso de estaciones a partir de meses de entrenamiento.",
}


def register(question: str, run_id: str, *, champion: bool = False,
             justification: str = ""):
    configurar_experimento()
    client = MlflowClient(tracking_uri=cfg.MLFLOW_TRACKING_URI)
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"La corrida {run_id} no está FINISHED: {run.info.status}")
    tags = run.data.tags
    if tags.get("question", "").lower() != question:
        raise ValueError(f"La corrida {run_id} no corresponde a {question.upper()}")
    required = ("git_commit", "data_version", "equipo")
    missing = [key for key in required if not tags.get(key)]
    if missing:
        raise ValueError(f"Faltan etiquetas de linaje: {missing}")
    if tags.get("git_dirty") == "True":
        print("ADVERTENCIA: el entrenamiento se realizó con cambios sin commit.")
    model_uri = tags.get("logged_model_uri")
    if not model_uri:
        raise ValueError(
            "Esta corrida no guarda logged_model_uri (MLflow 3). "
            "Reejecuta el modelo con el código actualizado antes de registrarlo."
        )    
    # Verificamos el artefacto incluso antes de crear una versión del Registry.
    loaded = mlflow.pyfunc.load_model(model_uri)
    if loaded.metadata.signature is None:
        raise ValueError("El modelo no tiene firma de entrada/salida")
    model_name = MODEL_NAMES[question]
    version = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.update_registered_model(model_name, description=DESCRIPTIONS[question])
    description = (
        f"Pregunta={question.upper()}, run={run_id}, commit={tags['git_commit']}, "
        f"data_version={tags['data_version']}. "
        f"Criterio de selección: {justification or 'documentado en el reporte de comparación'}."
    )
    client.update_model_version(model_name, version.version, description=description)
    lineage = {
        "question": question,
        "source_run_id": run_id,
        "source_experiment_id": str(run.info.experiment_id),
        "git_commit": tags["git_commit"],
        "data_version": tags["data_version"],
        "equipo": tags["equipo"],
        "source_logged_model_uri": model_uri,
    }
    for key, value in lineage.items():
        client.set_model_version_tag(model_name, version.version, key, str(value))
    if champion:
        # Publicar solo por decisión explícita y después de comprobar carga/firma.
        client.set_registered_model_alias(model_name, "champion", version.version)
    print(f"Registrado: {model_name} | versión {version.version}")
    print(f"Run: {run_id} | commit: {tags['git_commit']} | datos: {tags['data_version']}")
    print("Alias champion:", "asignado" if champion else "sin asignar")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", choices=MODEL_NAMES, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--champion", action="store_true", help="Publica esta versión como champion")
    parser.add_argument("--justification", default="", help="Por qué se escogió esta corrida")
    args = parser.parse_args()
    register(args.question, args.run_id, champion=args.champion, justification=args.justification)