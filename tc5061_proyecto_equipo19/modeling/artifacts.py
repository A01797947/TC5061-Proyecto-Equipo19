"""Funciones de artefactos con organización consistente y archivos temporales seguros."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import mlflow


def log_file(path, artifact_path=None):
    path = Path(path)
    if path.is_file():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    else:
        raise FileNotFoundError(path)


def log_directory(path, artifact_path=None):
    path = Path(path)
    if path.is_dir():
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)
    else:
        raise FileNotFoundError(path)


def log_run_config(config, filename="run_config.json"):
    with tempfile.TemporaryDirectory(prefix="mibici_config_") as directory:
        file = Path(directory) / filename
        file.write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        mlflow.log_artifact(str(file), artifact_path="config")


def log_manifest(manifest_path):
    log_file(manifest_path, artifact_path="data")


def log_plot(path):
    log_file(path, artifact_path="plots")