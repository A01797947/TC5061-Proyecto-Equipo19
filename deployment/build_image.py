"""Construye la imagen OCI desde un contexto aislado y verificable.

Evita que patrones de .containerignore/.dockerignore del repositorio filtren
artefactos que forman parte del snapshot MLflow.

Uso (raiz del repositorio):
    python -m deployment.build_image
    python -m deployment.build_image --tag mibici-equipo19-api:1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from deployment.api.inference import NAMES, folder_digest
import json

ROOT = Path(__file__).resolve().parents[1]


def _copy_file(source: Path, dest: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"No se encuentra el archivo requerido: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['podman', 'docker'], default='podman')
    parser.add_argument('--tag', default='mibici-equipo19-api:1.0')
    parser.add_argument('--no-cache', action='store_true', default=True)
    args = parser.parse_args()

    source_models = ROOT / 'deployment' / 'model_store'
    manifest_path = source_models / 'manifest.json'
    if not manifest_path.is_file():
        raise SystemExit('No existe el manifest. Ejecuta primero python -m deployment.export_models')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    # Verificar origen ANTES de copiar. No editar ni regenerar el snapshot.
    for key in NAMES:
        expected = manifest['models'][key]['sha256']
        actual = folder_digest(source_models / key)
        if actual != expected:
            raise SystemExit(f'Snapshot local {key} no coincide con su manifest: {actual} != {expected}')
    print('OK: snapshots locales P1/P2/P3 verificados.', flush=True)

    with tempfile.TemporaryDirectory(prefix='mibici-oci-') as td:
        context = Path(td)
        for relative in [
            'Containerfile',
            'requirements-serving.txt',
            'deployment/__init__.py',
            'deployment/verify_snapshot.py',
        ]:
            _copy_file(ROOT / relative, context / relative)

        api_src = ROOT / 'deployment' / 'api'
        if not api_src.is_dir():
            raise SystemExit(f'No existe {api_src}')
        shutil.copytree(api_src, context / 'deployment' / 'api',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(source_models, context / 'deployment' / 'model_store')

        # Los hashes de los modelos se calculan contra archivos ya copiados.
        for key in NAMES:
            expected = manifest['models'][key]['sha256']
            actual = folder_digest(context / 'deployment' / 'model_store' / key)
            if actual != expected:
                raise SystemExit(
                    f'Error al preparar contexto: {key} no coincide ({actual} != {expected}).'
                )
        print('OK: snapshots P1/P2/P3 íntegros en el contexto aislado.', flush=True)

        # No hay ignore heredado: el contexto solo contiene los archivos
        # explícitamente copiados, sin credenciales, notebooks ni datasets.
        command = [args.engine, 'build', '-f', 'Containerfile', '-t', args.tag]
        if args.no_cache:
            command.append('--no-cache')
        command.append('.')
        print('Ejecutando:', ' '.join(command), flush=True)
        print('Contexto temporal:', context, flush=True)
        try:
            subprocess.run(command, cwd=context, check=True)
        except FileNotFoundError as exc:
            raise SystemExit(f'No encuentro el ejecutable {args.engine}. Comprueba la instalación y PATH.') from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f'Falló la construcción OCI (código {exc.returncode}).') from exc

    print(f'OK: imagen construida: {args.tag}', flush=True)


if __name__ == '__main__':
    main()
