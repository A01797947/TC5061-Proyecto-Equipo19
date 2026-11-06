"""Verifica el contenedor real con los tres modelos exportados.

Uso (en otra terminal): python -m deployment.smoke_test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def request_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} en {url}: {exc.read().decode('utf-8')}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    files = {
        "p1": ROOT / "deployment/examples/p1.json",
        "p2": ROOT / "deployment/examples/p2.json",
        "p3": ROOT / "deployment/model_store/p3_ejemplo_peticion.json",
    }
    try:
        print("GET /health:", json.dumps(request_json("GET", f"{base}/health"), ensure_ascii=False))
        for key, file in files.items():
            payload = json.loads(file.read_text(encoding="utf-8"))
            response = request_json("POST", f"{base}/v1/predict/{key}", payload)
            print(f"POST /v1/predict/{key}:", json.dumps(response, ensure_ascii=False))
        print("[OK] Las tres predicciones respondieron correctamente.")
    except (FileNotFoundError, URLError, RuntimeError) as exc:
        raise SystemExit(f"[ERROR] No se pudo completar el smoke test: {exc}") from exc


if __name__ == "__main__":
    main()
