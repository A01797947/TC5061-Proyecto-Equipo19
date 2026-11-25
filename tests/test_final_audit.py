"""Asegura que la auditoría final no dependa de datasets, MLflow ni secretos."""
import hashlib
from pathlib import Path
import sqlite3

from quality.final_audit import audit, folder_sha256, read_registry_metadata


def test_sha_cambia_si_se_altera_un_snapshot(tmp_path):
    folder = tmp_path / 'p1'
    folder.mkdir()
    f = folder / 'model.skops'
    f.write_bytes(b'version original')
    digest = folder_sha256(folder)
    assert digest == folder_sha256(folder)
    assert len(digest) == 64
    f.write_bytes(b'otra version')
    assert digest != folder_sha256(folder)


def test_audit_no_inventa_evidencia(tmp_path):
    report = audit(tmp_path)
    by_name = {r['criterio']: r for r in report['criterios']}
    assert by_name['Evidencia real drift + A/B']['estado'] == 'pendiente'
    assert by_name['Locust y FinOps - evidencia medida']['estado'] == 'pendiente'
    assert by_name['Model Registry (metadatos SQLite)']['estado'] == 'pendiente'
    assert report['conteo']['error'] > 0


def test_metadatos_registry_sin_uris_ni_secrets(tmp_path):
    db = tmp_path / 'mlflow.db'
    con = sqlite3.connect(db)
    con.executescript('''
    CREATE TABLE model_versions (name TEXT, version INTEGER, run_id TEXT);
    CREATE TABLE runs (run_uuid TEXT, status TEXT);
    CREATE TABLE registered_model_aliases (name TEXT, alias TEXT, version INTEGER);
    CREATE TABLE model_version_tags (name TEXT, version INTEGER, key TEXT, value TEXT);
    INSERT INTO model_versions VALUES ('mibici-equipo19-p1-demanda', 1, 'abc');
    INSERT INTO runs VALUES ('abc', 'FINISHED');
    INSERT INTO registered_model_aliases VALUES ('mibici-equipo19-p1-demanda', 'champion', 1);
    INSERT INTO model_version_tags VALUES ('mibici-equipo19-p1-demanda', 1, 'git_commit', 'deadbeef');
    INSERT INTO model_version_tags VALUES ('mibici-equipo19-p1-demanda', 1, 'secret', 'no-exportar');
    ''')
    con.close()
    result = read_registry_metadata(db)
    assert result['modelos']['p1']['champion_registrado']
    assert result['modelos']['p1']['linaje_champion']['git_commit'] == 'deadbeef'
    assert 'no-exportar' not in str(result)
