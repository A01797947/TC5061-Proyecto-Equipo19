#!/usr/bin/env sh
set -eu

# Los snapshots se congelan al construir la imagen. No hay credenciales,
# Registry ni export_models en el contenedor de inferencia; fallar de forma
# explicita si un hash cambia es mas seguro que descargar otra version.
python -m deployment.verify_snapshot
exec uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000 --workers "${UVICORN_WORKERS:-1}"
