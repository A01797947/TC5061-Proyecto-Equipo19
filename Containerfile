# Imagen OCI compatible con Podman y Docker; ejecutar desde la raíz del proyecto.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MIBICI_MODEL_DIR=/app/model_store

WORKDIR /app

# Reproducibilidad: alinear runtime de sklearn/mlflow con el entrenamiento.
COPY requirements-serving.txt /app/requirements-serving.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-serving.txt \
    && groupadd --system appuser && useradd --system --gid appuser --home /app appuser

COPY deployment/__init__.py /app/deployment/__init__.py
COPY deployment/api /app/deployment/api
COPY deployment/verify_snapshot.py /app/deployment/verify_snapshot.py
COPY deployment/model_store /app/model_store
RUN test -s /app/model_store/manifest.json || ( \
    echo "ERROR: ejecuta 'python -m deployment.export_models' antes de podman build" >&2; exit 1)
# Validar durante podman build los mismos SHA256 que se comprobaran al iniciar.
# De este modo, un artefacto omitido por el contexto de build falla ANTES de podman run.
RUN python -m deployment.verify_snapshot

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
# La imagen slim no trae curl: se sondea /health con la biblioteca estándar.
# start-period amplio porque la API carga los 3 modelos al arrancar.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
# Un único proceso permite cargar los 3 modelos una sola vez en memoria.
CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
