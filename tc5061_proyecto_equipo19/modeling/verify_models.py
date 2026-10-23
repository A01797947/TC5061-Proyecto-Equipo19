"""Comprueba carga de los modelos de producción por URI models:/...@champion."""
import argparse

import mlflow
from mlflow import MlflowClient

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.modeling.register_models import MODEL_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", choices=["all", *MODEL_NAMES], default="all")
    args = parser.parse_args()
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=cfg.MLFLOW_TRACKING_URI)
    questions = MODEL_NAMES if args.question == "all" else {args.question: MODEL_NAMES[args.question]}
    for question, name in questions.items():
        uri = f"models:/{name}@champion"
        version = client.get_model_version_by_alias(name, "champion")
        model = mlflow.pyfunc.load_model(uri)
        signature = model.metadata.signature
        if signature is None or signature.inputs is None:
            raise AssertionError(f"Modelo sin firma: {uri}")
        print(f"[OK] {question.upper()}: {uri}")
        print(f"Versión: {version.version}, run origen: {version.run_id}")
        print(f"Firma: {signature}")
    print("Carga de modelos y firmas verificada.")


if __name__ == "__main__":
    main()