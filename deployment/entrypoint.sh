#!/usr/bin/env sh
set -e

if ! python -m deployment.verify_snapshot; then
  python -m deployment.export_models
  python -m deployment.verify_snapshot
fi

exec uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000 --workers "${UVICORN_WORKERS:-1}"