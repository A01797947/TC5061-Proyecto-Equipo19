"""Pruebas aisladas de FastAPI (no requieren CSV, Registry ni contenedor)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from deployment.api.inference import ModelStore, NAMES, dataframe_for_model, folder_digest
from deployment.api.main import app

ROOT = Path(__file__).resolve().parents[1]
P1 = [
    {"name": "Año", "type": "long"},
    {"name": "Mes", "type": "long"},
    {"name": "Dia", "type": "long"},
    {"name": "Dia_semana", "type": "long"},
    {"name": "origin_station_id", "type": "string"},
]
P2 = [
    {"name": k, "type": "double"} for k in json.loads(
        (ROOT / "deployment/examples/p2.json").read_text(encoding="utf-8")
    )
]
P3 = [{"name": "viajes_iniciados", "type": "double"},
      {"name": "viajes_finalizados", "type": "double"},
      {"name": "total_viajes", "type": "double"}]


class FakeModel:
    def __init__(self, answer):
        self.answer = answer

    def predict(self, frame):
        assert len(frame) == 1
        return np.array([self.answer])


class FakeClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        return np.array([[0.12, 0.88]])


@pytest.fixture
def client(monkeypatch):
    manifest = {"models": {
        key: {"name": NAMES[key], "version": "1", "alias": "champion", "signature": cols}
        for key, cols in [("p1", P1), ("p2", P2), ("p3", P3)]
    }}
    fake_store = ModelStore({"p1": FakeModel(14.5), "p2": FakeModel(1),
                             "p3": FakeModel(2)}, manifest, FakeClassifier())
    monkeypatch.setattr(ModelStore, "load_snapshot", classmethod(lambda cls, folder: fake_store))
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_models(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models": {"p1": "1", "p2": "1", "p3": "1"}}
    models = client.get("/v1/models").json()
    assert len(models) == 3
    assert models[0]["variables"] == [x["name"] for x in P1]
    assert client.get("/v1/models/p4").status_code == 404


def test_p1_prediction(client):
    payload = json.loads((ROOT / "deployment/examples/p1.json").read_text(encoding="utf-8"))
    response = client.post("/v1/predict/p1", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["prediccion_viajes"] == 14.5
    assert response.json()["version"] == "1"


def test_p1_invalid_date_and_weekday(client):
    payload = json.loads((ROOT / "deployment/examples/p1.json").read_text(encoding="utf-8"))
    payload["Dia_semana"] = 3
    assert client.post("/v1/predict/p1", json=payload).status_code == 422
    payload["Dia"] = 31
    assert client.post("/v1/predict/p1", json=payload).status_code == 422


def test_p2_prediction_with_probability(client):
    payload = json.loads((ROOT / "deployment/examples/p2.json").read_text(encoding="utf-8"))
    response = client.post("/v1/predict/p2", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["riesgo"] == 1
    assert response.json()["probabilidad_riesgo"] == 0.88


def test_p2_missing_field_and_invalid_range(client):
    payload = json.loads((ROOT / "deployment/examples/p2.json").read_text(encoding="utf-8"))
    del payload["capacidad"]
    assert client.post("/v1/predict/p2", json=payload).status_code == 422
    payload["capacidad"] = 10
    payload["ocupacion_est_h"] = 1.2
    assert client.post("/v1/predict/p2", json=payload).status_code == 422


def test_p3_dynamic_schema_and_unknown_field(client):
    correct = {"features": {"viajes_iniciados": 10, "viajes_finalizados": 12,
                            "total_viajes": 22}}
    response = client.post("/v1/predict/p3", json=correct)
    assert response.status_code == 200, response.text
    assert response.json()["cluster"] == 2
    wrong = {"features": {**correct["features"], "extra": 1}}
    assert client.post("/v1/predict/p3", json=wrong).status_code == 422
    incomplete = {"features": {"viajes_iniciados": 10}}
    assert client.post("/v1/predict/p3", json=incomplete).status_code == 422


def test_column_dtype_casting():
    data = dataframe_for_model({"Año": 2025, "origin_station_id": 10}, [
        {"name": "Año", "type": "long"},
        {"name": "origin_station_id", "type": "string"},
    ])
    assert data["Año"].dtype.name == "int64"
    assert data["origin_station_id"].iloc[0] == "10"
    with pytest.raises(ValueError, match="entero"):
        dataframe_for_model({"Año": 2025.2}, [{"name": "Año", "type": "long"}])


def test_digest_is_reproducible(tmp_path):
    file = tmp_path / "MLmodel"
    file.write_text("prueba", encoding="utf-8")
    first = folder_digest(tmp_path)
    assert first == folder_digest(tmp_path)
    file.write_text("cambiado", encoding="utf-8")
    assert first != folder_digest(tmp_path)


def test_export_registry_snapshots_with_fake_mlflow(monkeypatch, tmp_path):
    """Valida el flujo Registry -> versiones -> artefactos -> manifiesto sin datos reales."""
    import sys
    import types
    import pandas as pd
    from deployment.export_models import export

    source_dirs = {}
    signature_map = {"p1": P1, "p2": P2, "p3": P3}
    for key in NAMES:
        folder = tmp_path / f"source-{key}"
        folder.mkdir()
        (folder / "MLmodel").write_text(f"mock-{key}", encoding="utf-8")
        source_dirs[key] = folder

    class StubPyfunc:
        def __init__(self, key):
            specs = [types.SimpleNamespace(
                name=c["name"], type=types.SimpleNamespace(name=c["type"])
            ) for c in signature_map[key]]
            self.metadata = types.SimpleNamespace(
                signature=types.SimpleNamespace(inputs=types.SimpleNamespace(inputs=specs))
            )

        def predict(self, frame):
            return [1] * len(frame)

    class StubModel:
        def __init__(self, key):
            self.key = key

        def load_input_example(self, path):
            return pd.DataFrame([{c["name"]: ("1" if c["type"] == "string" else 1.0)
                                  for c in signature_map[self.key]}])

    def key_from_path(path):
        name = Path(path).name
        return name if name in NAMES else name.split("-")[-1]

    stub_mlflow = types.ModuleType("mlflow")
    stub_mlflow.set_tracking_uri = lambda uri: None
    stub_mlflow.MlflowClient = lambda tracking_uri: types.SimpleNamespace(
        get_model_version_by_alias=lambda name, alias: types.SimpleNamespace(version=1)
    )
    stub_mlflow.artifacts = types.SimpleNamespace(download_artifacts=lambda artifact_uri:
        str(source_dirs[next(k for k, v in NAMES.items() if v in artifact_uri)]))
    stub_mlflow.pyfunc = types.SimpleNamespace(load_model=lambda path: StubPyfunc(key_from_path(path)))
    stub_mlflow.models = types.SimpleNamespace(Model=types.SimpleNamespace(
        load=lambda path: StubModel(key_from_path(path))))
    monkeypatch.setitem(sys.modules, "mlflow", stub_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.pyfunc", stub_mlflow.pyfunc)

    output = tmp_path / "exported"
    result = export("sqlite:///mock.db", output)
    assert result["models"]["p3"]["version"] == "1"
    assert (output / "p1/MLmodel").is_file()
    assert (output / "p2/MLmodel").is_file()
    assert (output / "p3/MLmodel").is_file()
    assert set(json.loads((output / "p3_ejemplo_peticion.json").read_text())["features"]) == {
        c["name"] for c in P3
    }
    assert result["models"]["p1"]["sha256"] == folder_digest(output / "p1")
