"""Carga y validación de snapshots registrados en MLflow, sin servidor en producción."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

NAMES = {
    "p1": "mibici-equipo19-p1-demanda",
    "p2": "mibici-equipo19-p2-riesgo",
    "p3": "mibici-equipo19-p3-grupos",
}


def folder_digest(folder: Path) -> str:
    """Hash estable de nombres y contenidos; rechaza enlaces simbólicos."""
    digest = hashlib.sha256()
    # La ordenación de objetos Path depende del sistema operativo:
    # Windows compara nombres sin distinguir mayúsculas/minúsculas, Linux sí.
    # El manifest original se calculó en Windows: replicar ese orden
    # explícitamente para obtener exactamente el mismo SHA256 en Linux.
    paths = sorted(
        folder.rglob("*"),
        key=lambda path: path.relative_to(folder).as_posix().casefold(),
    )
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"No se aceptan enlaces simbólicos en {folder}")
        if not path.is_file():
            continue
        rel = path.relative_to(folder).as_posix()
        digest.update(rel.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(data)
    return digest.hexdigest()


def columns_from_metadata(model: Any) -> list[dict[str, str]]:
    """Obtiene los nombres y tipos de columnas definidos en la firma MLflow."""
    signature = model.metadata.signature
    if not signature or not signature.inputs:
        raise ValueError("El modelo MLflow no contiene firma de entrada")
    result = []
    for column in signature.inputs.inputs:
        if not getattr(column, "name", None):
            raise ValueError("Se requiere firma MLflow con nombres de columnas")
        dtype = getattr(column.type, "name", str(column.type).split(".")[-1]).lower()
        result.append({"name": column.name, "type": dtype})
    if len(result) != len({c["name"] for c in result}):
        raise ValueError("Firma con columnas duplicadas")
    return result


def dataframe_for_model(features: dict, signature: list[dict[str, str]]) -> pd.DataFrame:
    """Valida nombres y ordena/castea según la firma real del modelo exportado."""
    expected = [col["name"] for col in signature]
    missing = sorted(set(expected) - set(features))
    extra = sorted(set(features) - set(expected))
    if missing or extra:
        raise ValueError(f"Variables incorrectas. Faltantes: {missing}; adicionales: {extra}")
    result = pd.DataFrame([{name: features[name] for name in expected}])
    for column in signature:
        name, kind = column["name"], column["type"]
        if kind in {"string"}:
            if result[name].isna().any():
                raise ValueError(f"La variable {name} no admite nulos")
            result[name] = result[name].astype(str)
        elif kind in {"integer", "int", "long"}:
            value = pd.to_numeric(result[name], errors="raise")
            if value.isna().any() or (value % 1 != 0).any():
                raise ValueError(f"{name} necesita un número entero no nulo")
            result[name] = value.astype("int32" if kind in {"integer", "int"} else "int64")
        elif kind in {"float", "double"}:
            result[name] = pd.to_numeric(result[name], errors="raise").astype(
                "float32" if kind == "float" else "float64"
            )
        elif kind == "boolean":
            if any(val not in (0, 1, False, True) for val in result[name]):
                raise ValueError(f"{name} debe ser booleano")
            result[name] = result[name].astype(bool)
        else:
            raise ValueError(f"Tipo MLflow no soportado en esta API: {name} ({kind})")
    return result


class ModelStore:
    def __init__(self, models: dict[str, Any], manifest: dict, classifier: Any = None):
        self.models = models
        self.manifest = manifest
        self.classifier = classifier

    @classmethod
    def load_snapshot(cls, root: Path) -> "ModelStore":
        """Carga los tres modelos una vez al arrancar; falla si falta uno."""
        import mlflow.pyfunc
        import mlflow.sklearn

        manifest_file = root / "manifest.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1:
            raise RuntimeError("Versión de manifiesto de modelos no admitida")
        models = {}
        native_classifier = None
        for key, name in NAMES.items():
            entry = manifest["models"][key]
            if entry.get("name") != name:
                raise RuntimeError(f"Nombre de modelo inconsistente en {key}")
            folder = root / key
            if not (folder / "MLmodel").is_file():
                raise RuntimeError(f"No existe el artefacto MLflow de {key}: {folder}")
            if folder_digest(folder) != entry["sha256"]:
                raise RuntimeError(f"El snapshot {key} cambió desde su exportación")
            model = mlflow.pyfunc.load_model(str(folder))
            if columns_from_metadata(model) != entry["signature"]:
                raise RuntimeError(f"Firma MLflow inconsistente para {key}")
            models[key] = model
            if key == "p2":
                native_classifier = mlflow.sklearn.load_model(str(folder))
        LOGGER.info("Cargados los tres modelos MiBici desde snapshots verificados")
        return cls(models, manifest, native_classifier)

    def info(self, key: str) -> dict:
        entry = self.manifest["models"][key]
        return {
            "modelo": entry["name"],
            "version": str(entry["version"]),
            "alias": entry["alias"],
            "variables": [field["name"] for field in entry["signature"]],
        }

    def predict(self, key: str, features: dict) -> tuple[Any, pd.DataFrame]:
        frame = dataframe_for_model(features, self.manifest["models"][key]["signature"])
        values = self.models[key].predict(frame)
        if len(values) != 1:
            raise RuntimeError("La predicción debe devolver exactamente un registro")
        value = values[0]
        if not np.isfinite(float(value)):
            raise RuntimeError("El modelo devolvió una predicción no finita")
        return value, frame

    def risk_probability(self, frame: pd.DataFrame) -> float | None:
        if self.classifier is None or not hasattr(self.classifier, "predict_proba"):
            return None
        probabilities = self.classifier.predict_proba(frame)[0]
        classes = list(getattr(self.classifier, "classes_", []))
        if len(classes) != len(probabilities) or 1 not in classes:
            return None
        result = float(probabilities[classes.index(1)])
        if not math.isfinite(result) or not 0 <= result <= 1:
            raise RuntimeError("La probabilidad es inválida")
        return result
