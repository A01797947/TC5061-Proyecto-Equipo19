"""Compara candidatas por VALIDACIÓN; reserva TEST para evaluación final.

Ejemplo: python -m tc5061_proyecto_equipo19.modeling.compare_models
Los resultados se guardan en outputs/model_comparison.json (no versionado).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.dataset import leer_manifiesto

CRITERIA = {
    "p1": ("valid_mae", False),
    "p2": (f"valid_precision_at_{cfg.TOP_K}", True),
    "p3": ("valid_silhouette", True),
}


def compare(question="all"):
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=cfg.MLFLOW_TRACKING_URI)
    exp = client.get_experiment_by_name(cfg.MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise ValueError("No existe el experimento. Entrena primero los modelos")
    current_data_version = leer_manifiesto()["data_version"]
    results = {}
    for q in (CRITERIA if question == "all" else [question]):
        metric, higher_is_better = CRITERIA[q]
        runs = client.search_runs([exp.experiment_id],
                                  filter_string=f"tags.question = '{q}'",
                                  max_results=1000)
        candidates = []
        for run in runs:
            if run.info.status != "FINISHED":
                continue
            if run.data.tags.get("data_version") != current_data_version:
                continue  # No mezclar versiones de datos.
            val = run.data.metrics.get(metric)
            if val is None:
                continue  # No mezclar métricas de versiones antiguas de P3.
            candidates.append({
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "stage": run.data.tags.get("stage_of_work", ""),
                "model": run.data.tags.get("model", run.data.params.get("model", "")),
                "model_uri": run.data.tags.get("logged_model_uri", ""),
                "data_version": current_data_version,
                "valid_metric": metric,
                "valid_score": val,
                "test_metric": metric.replace("valid_", "test_", 1),
                "test_score": run.data.metrics.get(metric.replace("valid_", "test_", 1)),
                "train_duration_seconds": (run.info.end_time - run.info.start_time) / 1000
                if run.info.end_time and run.info.start_time else None,
            })
        candidates.sort(key=lambda item: item["valid_score"], reverse=higher_is_better)
        results[q] = candidates
        print(f"\n{q.upper()} | métrica de selección: {metric} | datos={current_data_version}")
        for item in candidates:
            print(f"  {item['valid_score']:.5f} | {item['run_name']:<43} | "
                  f"{item['run_id']} | registrable={bool(item['model_uri'])}")
        eligible = [item for item in candidates if item["model_uri"]]
        if eligible:
            print("  Ganador provisional por validación:", eligible[0]["run_id"])
        else:
            print("  Faltan candidatos registrables para esta pregunta")
    output = cfg.PROJ_ROOT / "outputs" / "model_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nReporte JSON:", output)
    print("IMPORTANTE: validar estabilidad, baseline y costo antes de publicar champion.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", choices=["all", *CRITERIA], default="all")
    args = parser.parse_args()
    compare(args.question)
