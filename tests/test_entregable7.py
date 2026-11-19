"""Pruebas autocontenidas del pipeline de medición (sin daemon Docker ni datos)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from load_testing import benchmark, report_finops


def test_conversion_de_cpu_y_memoria():
    assert benchmark.ram_mib("512MiB / 8GiB") == 512
    assert benchmark.ram_mib("1.25GiB / 8GiB") == 1280
    assert abs(benchmark.ram_mib("500MB / 8GB") - (500e6 / 1024**2)) < 1e-8
    assert benchmark.cpu_percent("12.45%") == 12.45


def test_parseo_docker_stats(monkeypatch):
    id1, id2 = "abcdef0123456789", "123456abcdef7890"
    lines = '\n'.join([
        json.dumps({"ID": id1[:12], "CPUPerc": "105.2%", "MemUsage": "200MiB / 8GiB"}),
        json.dumps({"ID": id2[:12], "CPUPerc": "1.5%", "MemUsage": "50MiB / 8GiB"}),
    ])
    monkeypatch.setattr(benchmark, "run", lambda *a, **kw: SimpleNamespace(stdout=lines))
    data = benchmark.sample("docker", {"api": id1, "mlflow": id2})
    assert {d["service"] for d in data} == {"api", "mlflow"}
    assert next(x["cpu_percent"] for x in data if x["service"] == "api") == 105.2


def test_parseo_podman_stats(monkeypatch):
    cid = "a9f807ffaacdbbbbb"
    row = [{"id": cid[:12], "cpu_percent": "2.1%", "mem_usage": "3.092MB / 16.7GB"}]
    monkeypatch.setattr(benchmark, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(row)))
    data = benchmark.sample("podman", {"api": cid})
    assert len(data) == 1
    assert data[0]["mem_mib"] > 2


def test_report_finops_con_datos_de_prueba_sinteticos(tmp_path):
    run = "ensayo"
    pd.DataFrame([
        {"Type": "POST", "Name": "POST /v1/predict/p1", "Request Count": 30,
         "Failure Count": 1, "Average Response Time": 20, "Median Response Time": 18,
         "95%": 35, "99%": 40},
        {"Type": "", "Name": "Aggregated", "Request Count": 50,
         "Failure Count": 2, "Average Response Time": 25, "Median Response Time": 21,
         "95%": 55, "99%": 70},
    ]).to_csv(tmp_path / f"{run}_locust_stats.csv", index=False)
    pd.DataFrame([
        {"Timestamp": 1000, "Name": "Aggregated", "Total RPS": 2.2,
         "Total Failure/s": 0.1, "Total Average Response Time": 25},
        {"Timestamp": 1010, "Name": "Aggregated", "Total RPS": 3.5,
         "Total Failure/s": 0.0, "Total Average Response Time": 23},
    ]).to_csv(tmp_path / f"{run}_locust_stats_history.csv", index=False)
    pd.DataFrame([
        {"elapsed_s": 1, "service": "api", "cpu_percent": 20.0, "mem_mib": 210},
        {"elapsed_s": 11, "service": "api", "cpu_percent": 35.0, "mem_mib": 250},
        {"elapsed_s": 1, "service": "mlflow", "cpu_percent": 1.0, "mem_mib": 90},
    ]).to_csv(tmp_path / f"{run}_recursos.csv", index=False)
    report = report_finops.analyze(tmp_path, run, 20, 0.0835, 5, "docker")
    assert report["success"] == 48
    assert report["errors"] == 2
    assert report["rps_success"] == 2.4
    assert abs(report["usd_per_successful_inference"] - 0.0835 * 20 / 3600 / 48) < 1e-12
    assert (tmp_path / f"{run}_resumen.md").is_file()
    assert (tmp_path / f"{run}_resumen.json").is_file()
    for graph in report["graphs"]:
        assert (tmp_path / graph).stat().st_size > 200


def test_no_inventa_resultados_si_faltan_csv(tmp_path):
    with pytest.raises(FileNotFoundError):
        report_finops.analyze(tmp_path, "sin_medicion", 120, 0.0835, 5, "docker")


def test_pipeline_snapshot_y_compose_presentes():
    root = Path(__file__).resolve().parents[1]
    workflow = root / ".github/workflows/entregable7-cicd.yml"
    compose = root / "docker-compose.yml"
    assert workflow.is_file() and compose.is_file()
    assert "deployment.verify_snapshot" in workflow.read_text(encoding="utf-8")
    assert "locust:" in compose.read_text(encoding="utf-8")


def test_consolidado_exige_resultados_reales(tmp_path):
    from load_testing.consolidate import consolidate
    with pytest.raises(FileNotFoundError):
        consolidate(tmp_path, ['baja5', 'media20', 'alta50'])


def test_consolidado_realiza_comparacion_sin_inventar(tmp_path):
    from load_testing.consolidate import consolidate
    for run, users, rps in [('baja5', 5, 2.5), ('media20', 20, 7.5)]:
        (tmp_path / f'{run}_grafico.png').write_bytes(b'png-para-test')
        summary = {
            'run': run, 'users': users, 'duration_seconds_configured': 120,
            'success': int(rps * 120), 'rps_success': rps,
            'latency_ms': {'promedio': 25, 'p95': 40, 'p99': 60},
            'error_rate_pct': 0,
            'resource_avg_cpu_percent_by_service': {'api': 8},
            'resource_peak_mib_by_service': {'api': 300},
            'usd_per_successful_inference': 0.000005,
            'usd_per_1000_successful_inferences': 0.005,
            'graphs': [f'{run}_grafico.png'],
        }
        (tmp_path / f'{run}_resumen.json').write_text(json.dumps(summary), encoding='utf-8')
    report = consolidate(tmp_path, ['baja5', 'media20'])
    assert '7.500' in report.read_text(encoding='utf-8')
    assert 'PENDIENTE' in report.read_text(encoding='utf-8')
