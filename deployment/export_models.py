"""Exporta los tres aliases champion desde MLflow Registry a un snapshot OCI.

Ejecutar en el mismo equipo que conserva los artefactos originales. En esta
versión del proyecto, el Registry guarda rutas file:///D:/... de Windows; la
exportación resuelve esas rutas ANTES de construir una imagen Linux.

Uso: python -m deployment.export_models
     python -m deployment.export_models --tracking-uri http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from deployment.api.schemas import P1Request, P2Request

from deployment.api.inference import NAMES, columns_from_metadata, folder_digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "deployment" / "model_store"


def export(tracking_uri: str, destination: Path) -> dict:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mibici-registry-", dir=destination.parent) as tmp:
        output = Path(tmp)
        models = {}
        for key, name in NAMES.items():
            version = client.get_model_version_by_alias(name, "champion")
            uri = f"models:/{name}/{version.version}"  # Fija versión resuelta del alias
            print(f"Exportando {key.upper()} {name} v{version.version}...")
            # En MLflow 3, el artefacto del modelo registrado puede apuntar a file:///...
            source_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
            target_path = output / key
            shutil.copytree(source_path, target_path)
            if not (target_path / "MLmodel").is_file():
                raise RuntimeError(f"MLmodel no encontrado en {uri}")
            model = mlflow.pyfunc.load_model(str(target_path))
            signature = columns_from_metadata(model)
            registered_example = mlflow.models.Model.load(str(target_path)).load_input_example(
                str(target_path)
            )
            if registered_example is None or len(registered_example) == 0:
                raise RuntimeError(f"No existe input_example válido en {uri}")
            # Una predicción real de cada campeón antes de congelarlo en la imagen.
            model.predict(registered_example.head(1))
            required = set(P1Request.model_fields) if key == "p1" else (
                set(P2Request.model_fields) if key == "p2" else None
            )
            if required is not None and {field["name"] for field in signature} != required:
                raise RuntimeError(
                    f"La firma registrada en {key} no coincide con el esquema de la API. "
                    "Actualiza el esquema antes de exportar."
                )
            models[key] = {
                "name": name,
                "version": str(version.version),
                "alias": "champion",
                "signature": signature,
                "sha256": folder_digest(target_path),
            }
            print(f"OK: {len(signature)} variables, sha256={models[key]['sha256'][:12]}...")
        # P3 puede tener 3, 31, 49 o más variables, según el ganador.
        # Generamos un request real desde el input_example del modelo registrado.
        p3_model = mlflow.models.Model.load(str(output / "p3"))
        p3_example = p3_model.load_input_example(str(output / "p3"))
        if p3_example is None or len(p3_example) == 0:
            raise RuntimeError("El modelo P3 no tiene un input_example para generar el ejemplo")
        p3_columns = [f["name"] for f in models["p3"]["signature"]]
        example_row = p3_example.iloc[0].to_dict()
        p3_request = {"features": {key: float(example_row[key]) for key in p3_columns}}
        if not all(math.isfinite(value) for value in p3_request["features"].values()):
            raise RuntimeError("El ejemplo registrado de P3 contiene NaN o infinito")
        (output / "p3_ejemplo_peticion.json").write_text(
            json.dumps(p3_request, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format_version": 1,
            "strategy": "bundled-registry-snapshot",
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "models": models,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # Reemplaza solo después de haber exportado y validado los tres modelos.
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(output), str(destination))
    print(f"Snapshot completo en: {destination}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"),
        help="URI del backend de Model Registry (por defecto mlflow.db de este proyecto)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    export(args.tracking_uri, args.output)


if __name__ == "__main__":
    main()
