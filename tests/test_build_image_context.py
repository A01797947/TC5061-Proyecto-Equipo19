"""Regresión: el contexto aislado debe contener todos los archivos del Containerfile.

La prueba intercepta el comando Docker/Podman y no levanta contenedores.
"""
import sys
from types import SimpleNamespace

from deployment import build_image


def test_contexto_aislado_copia_entrypoint_y_snapshots(monkeypatch):
    inspected = {}

    def fake_run(command, cwd, check):
        assert command[:2] == ['docker', 'build']
        assert check
        for relative in ('deployment/entrypoint.sh', 'deployment/verify_snapshot.py',
                         'deployment/api/main.py', 'deployment/model_store/manifest.json',
                         'deployment/model_store/p1/model.skops',
                         'deployment/model_store/p2/model.skops',
                         'deployment/model_store/p3/model.skops'):
            assert (cwd / relative).is_file(), relative
        inspected['ok'] = True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build_image.subprocess, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['build_image.py', '--engine', 'docker', '--cache'])
    build_image.main()
    assert inspected['ok']


def test_entrypoint_no_intenta_exportar_si_falla_la_verificacion():
    content = (build_image.ROOT / 'deployment/entrypoint.sh').read_text(encoding='utf-8')
    assert 'python -m deployment.verify_snapshot' in content
    assert 'python -m deployment.export_models' not in content
