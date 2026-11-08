"""Servicio REST MiBici; los artefactos se exportan previamente del Registry MLflow."""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from deployment.api.inference import ModelStore
from deployment.api.schemas import (
    ModelInfo, P1Request, P1Response, P2Request, P2Response, P3Request, P3Response
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)
ROOT = Path(os.getenv("MIBICI_MODEL_DIR", "/app/model_store"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_store = ModelStore.load_snapshot(Path(os.getenv("MIBICI_MODEL_DIR", str(ROOT))))
    yield
    app.state.model_store = None


app = FastAPI(
    title="MiBici Equipo 19 - API de inferencia",
    description="P1 demanda, P2 riesgo de estación y P3 perfiles de uso.",
    version="1.0.0",
    lifespan=lifespan,
)


def store() -> ModelStore:
    obj = getattr(app.state, "model_store", None)
    if obj is None:
        raise HTTPException(status_code=503, detail="Los modelos no están disponibles")
    return obj


@app.get("/health")
def health():
    s = store()
    return {"status": "ok", "models": {key: s.info(key)["version"] for key in ("p1", "p2", "p3")}}


@app.get("/v1/models", response_model=list[ModelInfo])
def models():
    s = store()
    return [s.info(key) for key in ("p1", "p2", "p3")]


@app.get("/v1/models/{key}", response_model=ModelInfo)
def model_by_key(key: str):
    if key not in {"p1", "p2", "p3"}:
        raise HTTPException(status_code=404, detail="Modelo no reconocido")
    return store().info(key)


def inference(key: str, values: dict):
    try:
        return store().predict(key, values)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Fallo de inferencia para %s", key)
        raise HTTPException(status_code=500, detail="Error interno de inferencia") from exc


@app.post("/v1/predict/p1", response_model=P1Response)
def predict_p1(data: P1Request):
    s = store()
    model = s.info("p1")
    if s.known_stations is not None and str(data.origin_station_id) not in s.known_stations:
        raise HTTPException(status_code=422, detail=f"ID de estación no válido: {data.origin_station_id}")
    value, _ = inference("p1", data.model_dump())
    return P1Response(modelo=model["modelo"], version=model["version"], prediccion_viajes=float(value))


@app.post("/v1/predict/p2", response_model=P2Response)
def predict_p2(data: P2Request):
    s = store()
    model = s.info("p2")
    value, frame = inference("p2", data.model_dump())
    label = int(value)
    if label not in (0, 1) or value != label:
        raise HTTPException(status_code=500, detail="El clasificador devolvió una clase inválida")
    try:
        prob = s.risk_probability(frame)
    except Exception as exc:
        LOGGER.exception("No se pudo obtener la probabilidad P2")
        raise HTTPException(status_code=500, detail="Error interno de inferencia") from exc
    return P2Response(modelo=model["modelo"], version=model["version"],
                      riesgo=label, probabilidad_riesgo=prob)


@app.post("/v1/predict/p3", response_model=P3Response)
def predict_p3(data: P3Request):
    model = store().info("p3")
    value, _ = inference("p3", data.features)
    cluster = int(value)
    if cluster < 0 or value != cluster:
        raise HTTPException(status_code=500, detail="Clúster inválido")
    return P3Response(modelo=model["modelo"], version=model["version"], cluster=cluster)
