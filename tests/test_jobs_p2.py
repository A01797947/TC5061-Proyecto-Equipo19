"""Pruebas de la predicción P2 asíncrona sin Redis ni contenedores (cola simulada)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from deployment.api import jobs
from deployment.api.inference import NAMES, ModelStore
from deployment.api.main import app

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = json.loads((ROOT / "deployment/examples/p2.json").read_text(encoding="utf-8"))
JOB_ID = "f1fd6315-080b-4b19-b3b7-7c5de6aab73a"
RESULTADO = {"modelo": NAMES["p2"], "version": "1", "riesgo": 1, "probabilidad_riesgo": 0.88}


class FakeModel:
    def predict(self, frame):
        return np.array([1])


class FakeClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        return np.array([[0.12, 0.88]])


class FakeJob:
    def __init__(self, estado, valor=None):
        self.id = JOB_ID
        self._estado = estado
        self._valor = valor

    def get_status(self):
        return self._estado

    def return_value(self):
        return self._valor


class FakeQueue:
    def __init__(self, job, falla=None):
        self.job = job
        self.falla = falla
        self.enqueued = []

    def enqueue(self, func, *args, **kwargs):
        if self.falla:
            raise self.falla
        self.enqueued.append((func, args, kwargs))
        return self.job

    def fetch_job(self, job_id):
        return self.job if job_id == self.job.id else None


def _store():
    columnas = [{"name": nombre, "type": "double"} for nombre in PAYLOAD]
    manifest = {"models": {"p2": {"name": NAMES["p2"], "version": "1",
                                  "alias": "champion", "signature": columnas}}}
    return ModelStore({"p2": FakeModel()}, manifest, FakeClassifier())


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(ModelStore, "load_snapshot", classmethod(lambda cls, folder: _store()))
    with TestClient(app) as test_client:
        yield test_client


def test_trabajo_p2_da_el_mismo_resultado_que_la_ruta_sincrona(monkeypatch):
    monkeypatch.setattr(jobs, "_STORE", _store())
    assert jobs.predict_p2_job(PAYLOAD) == RESULTADO


def test_encolar_p2_devuelve_202_y_el_job_id(cliente):
    cola = FakeQueue(FakeJob("queued"))
    app.state.queue = cola
    respuesta = cliente.post("/v1/jobs/predict/p2", json=PAYLOAD)
    assert respuesta.status_code == 202, respuesta.text
    assert respuesta.json() == {"job_id": JOB_ID, "estado": "queued"}
    funcion, _, opciones = cola.enqueued[0]
    assert funcion is jobs.predict_p2_job
    assert opciones["job_timeout"] == jobs.JOB_TIMEOUT
    assert opciones["result_ttl"] == jobs.RESULT_TTL


def test_encolar_p2_con_datos_invalidos_no_toca_la_cola(cliente):
    cola = FakeQueue(FakeJob("queued"))
    app.state.queue = cola
    datos = dict(PAYLOAD)
    del datos["capacidad"]
    assert cliente.post("/v1/jobs/predict/p2", json=datos).status_code == 422
    assert cola.enqueued == []


def test_consultar_trabajo_terminado(cliente):
    app.state.queue = FakeQueue(FakeJob("finished", RESULTADO))
    respuesta = cliente.get(f"/v1/jobs/{JOB_ID}")
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "finished"
    assert respuesta.json()["resultado"]["riesgo"] == 1
    assert respuesta.json()["error"] is None


def test_consultar_trabajo_fallido_no_expone_detalles(cliente):
    app.state.queue = FakeQueue(FakeJob("failed"))
    cuerpo = cliente.get(f"/v1/jobs/{JOB_ID}").json()
    assert cuerpo["estado"] == "failed"
    assert cuerpo["resultado"] is None
    assert cuerpo["error"] == "La predicción falló"


def test_consultar_trabajo_inexistente_o_con_id_invalido(cliente):
    app.state.queue = FakeQueue(FakeJob("queued"))
    assert cliente.get("/v1/jobs/00000000-0000-0000-0000-000000000000").status_code == 404
    assert cliente.get("/v1/jobs/abc").status_code == 422


def test_sin_cola_configurada_responde_503(cliente):
    assert app.state.queue is None
    assert cliente.post("/v1/jobs/predict/p2", json=PAYLOAD).status_code == 503


def test_redis_caido_responde_503(cliente):
    app.state.queue = FakeQueue(FakeJob("queued"), falla=RedisConnectionError("sin conexión"))
    assert cliente.post("/v1/jobs/predict/p2", json=PAYLOAD).status_code == 503