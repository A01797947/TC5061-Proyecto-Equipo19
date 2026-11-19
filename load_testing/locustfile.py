"""Carga real sobre las tres predicciones: P1 50%, P2 30%, P3 20%.

No usa datos de entrenamiento; carga ejemplos válidos incluidos en el proyecto.
Los tiempos incluyen red HTTP, validación de datos e inferencia.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from locust import HttpUser, between, task


def example_path(name: str) -> Path:
    local_root = Path(__file__).resolve().parents[1]
    local = (local_root / name)
    if local.is_file():
        return local
    # El contenedor monta los archivos de ejemplos en rutas conocidas.
    return Path({
        "deployment/examples/p1.json": "/mnt/examples/p1.json",
        "deployment/examples/p2.json": "/mnt/examples/p2.json",
        "deployment/model_store/p3_ejemplo_peticion.json": "/mnt/p3_ejemplo_peticion.json",
    }[name])


def payload(name: str) -> dict:
    with example_path(name).open(encoding="utf-8") as fh:
        return json.load(fh)


P1 = payload("deployment/examples/p1.json")
P2 = payload("deployment/examples/p2.json")
P3 = payload("deployment/model_store/p3_ejemplo_peticion.json")


class UsuarioMiBici(HttpUser):
    # Espera realista; configurar el número de usuarios desde Compose.
    wait_time = between(1.0, 2.0)

    def consultar(self, name: str, data: dict, field: str) -> None:
        with self.client.post(
            f"/v1/predict/{name}",
            json=data,
            name=f"POST /v1/predict/{name}",
            timeout=30,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:180]}")
                return
            try:
                body = response.json()
            except ValueError:
                response.failure("El servidor no devolvió JSON")
                return
            if field not in body or "version" not in body:
                response.failure(f"Respuesta sin {field}/version")
            else:
                response.success()

    @task(5)
    def demanda_p1(self):
        self.consultar("p1", P1, "prediccion_viajes")

    @task(3)
    def riesgo_p2(self):
        self.consultar("p2", P2, "riesgo")

    @task(2)
    def grupos_p3(self):
        self.consultar("p3", P3, "cluster")
