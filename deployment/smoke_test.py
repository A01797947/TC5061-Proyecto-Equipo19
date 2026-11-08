"""Verifica el contenedor real con los tres modelos exportados.

Uso (en otra terminal): python -m deployment.smoke_test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def validate_base_url(url: str) -> str:
    """Acepta solo http(s) con host: evita file:// u otros esquemas en urlopen (Bandit B310)."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"--base-url debe ser http:// o https:// con host, no {url!r}")
    return url.rstrip("/")


def request_json(method: str, url: str, payload: dict | None = None) -> dict:
    validate_base_url(url)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        # El esquema ya se restringió a http(s) en validate_base_url.
        with urlopen(request, timeout=30) as response:  # nosec B310
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} en {url}: {exc.read().decode('utf-8')}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    try:
        base = validate_base_url(args.base_url)
    except ValueError as exc:
        parser.error(str(exc))
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
