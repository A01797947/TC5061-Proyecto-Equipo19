"""Comprueba que el snapshot OCI coincide con el manifest generado desde MLflow.

Uso local: python -m deployment.verify_snapshot
Uso en build: RUN python -m deployment.verify_snapshot
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from deployment.api.inference import NAMES, folder_digest


def main() -> None:
    root = Path(os.getenv('MIBICI_MODEL_DIR', str(Path(__file__).resolve().parent / 'model_store')))
    manifest_file = root / 'manifest.json'
    if not manifest_file.is_file():
        raise SystemExit(f'ERROR: no existe {manifest_file}')
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    for key in NAMES:
        folder = root / key
        if not (folder / 'MLmodel').is_file():
            raise SystemExit(f'ERROR: falta {folder / "MLmodel"}')
        expected = manifest['models'][key]['sha256']
        actual = folder_digest(folder)
        if actual != expected:
            raise SystemExit(f'ERROR: snapshot {key} distinto al exportado. Esperado: {expected}, obtenido: {actual}. Verifica .containerignore/.dockerignore y reconstruye la imagen.')
        print(f'OK {key.upper()}: SHA256 {actual[:16]}... coincide con el manifest')
    print('OK: integridad de los tres modelos confirmada.')


if __name__ == '__main__':
    main()
