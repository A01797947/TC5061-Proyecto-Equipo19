"""Auditoría estática y segura de la entrega final MiBici.

No ejecuta entrenamiento, inferencia, descarga de datos ni consultas remotas.
No lee .env, secretos ni el contenido de datos de viajes.

Uso:
  python -m quality.final_audit
  python -m quality.final_audit --db mlflow.db --json reports/auditoria_local.json
  python -m quality.final_audit --strict  # falla si faltan evidencias opcionales
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {
    "p1": "mibici-equipo19-p1-demanda",
    "p2": "mibici-equipo19-p2-riesgo",
    "p3": "mibici-equipo19-p3-grupos",
}


def folder_sha256(folder: Path) -> str:
    """Hash compatible con el manifest Windows, sin cargar modelos Python."""
    if not folder.is_dir():
        return ""
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*"), key=lambda p: p.relative_to(folder).as_posix().casefold()):
        if path.is_symlink():
            raise ValueError(f"Enlace simbólico no permitido: {path.name}")
        if not path.is_file():
            continue
        digest.update(path.relative_to(folder).as_posix().encode("utf-8") + b"\0")
        with path.open("rb") as fp:
            for block in iter(lambda: fp.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def read_registry_metadata(db: Path) -> dict:
    """Lee solamente versiones, alias, tags de linaje y estados. No extrae URIs ni secretos."""
    result = {"fuente": "sqlite_local", "modelos": {}}
    with sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True) as con:
        for key, name in MODEL_NAMES.items():
            versions = [dict(zip(("version", "run_id", "estado"), row)) for row in con.execute(
                "SELECT mv.version, mv.run_id, COALESCE(r.status,'NO_RUN') "
                "FROM model_versions mv LEFT JOIN runs r ON r.run_uuid=mv.run_id "
                "WHERE mv.name=? ORDER BY mv.version", (name,)
            )]
            aliases = dict(con.execute(
                "SELECT alias, version FROM registered_model_aliases WHERE name=?", (name,)
            ).fetchall())
            champion = aliases.get("champion")
            lineage = dict(con.execute(
                "SELECT key,value FROM model_version_tags WHERE name=? AND version=? "
                "AND key IN ('question','source_run_id','source_experiment_id','data_version',"
                "'data_version_e8','git_commit','equipo')", (name, champion),
            ).fetchall()) if champion is not None else {}
            result["modelos"][key] = {
                "nombre": name, "versiones": versions,
                "aliases": aliases, "linaje_champion": lineage,
                "champion_registrado": champion is not None,
            }
    return result


def audit(root: Path, db: Path | None = None) -> dict:
    root = Path(root)
    checks: list[dict] = []

    def check(name: str, status: str, explanation: str, evidence: str = "") -> None:
        checks.append({"criterio": name, "estado": status, "detalle": explanation, "evidencia": evidence})

    def present(paths: list[str]) -> list[str]:
        return [p for p in paths if not (root / p).is_file()]

    required = [
        "MLproject", "python_env.yaml", "pyproject.toml", "Makefile",
        "references/data_manifest.json", "tc5061_proyecto_equipo19/dataset.py",
        "tc5061_proyecto_equipo19/features/base.py",
        "tc5061_proyecto_equipo19/features/p1.py",
        "tc5061_proyecto_equipo19/features/p2.py",
        "tc5061_proyecto_equipo19/features/p3.py",
        "tc5061_proyecto_equipo19/project_runner.py",
        "tc5061_proyecto_equipo19/modeling/train_P1.py",
        "tc5061_proyecto_equipo19/modeling/train_p2.py",
        "tc5061_proyecto_equipo19/modeling/train_P3.py",
        "tc5061_proyecto_equipo19/modeling/register_models.py",
    ]
    missing = present(required)
    check("Estructura, feature engineering y MLflow Project",
          "ok" if not missing else "error",
          "Componentes presentes" if not missing else f"Faltan: {', '.join(missing)}")

    deploy = ["Containerfile", "deployment/api/main.py", "deployment/api/inference.py",
              "deployment/entrypoint.sh", "deployment/smoke_test.py", "docker-compose.yml",
              "deployment/api/worker.py", "deployment/api/jobs.py"]
    missing = present(deploy)
    check("API, contenedor y Compose", "ok" if not missing else "error",
          "Código de despliegue presente; la ejecución no está probada aquí" if not missing
          else f"Faltan: {', '.join(missing)}")

    manifest_path = root / "deployment/model_store/manifest.json"
    snapshot_status = "error"
    snapshot_description = "Falta manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        issues = []
        for key, expected_name in MODEL_NAMES.items():
            model = manifest["models"][key]
            path = root / "deployment/model_store" / key
            if model["name"] != expected_name or model.get("alias") != "champion":
                issues.append(f"{key}: nombre/alias inconsistente")
            elif folder_sha256(path) != model["sha256"]:
                issues.append(f"{key}: SHA256 no coincide")
        snapshot_status = "error" if issues else "ok"
        snapshot_description = "; ".join(issues) if issues else "3 snapshots íntegros y versionados en el manifest"
    except (KeyError, OSError, ValueError, TypeError) as exc:
        snapshot_description = f"No se pudo verificar el manifest: {type(exc).__name__}"
    check("SHA256 snapshots MLflow P1/P2/P3", snapshot_status, snapshot_description,
          "deployment/model_store/manifest.json")

    workflows = [".github/workflows/entregable7-cicd.yml", ".github/workflows/quality-security.yml"]
    missing = present(workflows)
    check("Definiciones CI/CD y calidad", "ok" if not missing else "error",
          "Workflows presentes; comprobar ejecución verde en GitHub Actions" if not missing
          else f"Faltan: {', '.join(missing)}")

    check("Pruebas sintéticas y seguridad", "ok" if not present([
        "tests/test_dataset.py", "tests/test_deployment_api.py", "tests/test_monitoring_e8.py",
        "quality/secrets_scan.py", "requirements-quality.txt", "requirements-deepchecks.txt",
    ]) else "error", "Código disponible; ejecutar PyTest/escaneos en el entorno previsto")

    registry = None
    if db is None:
        db = root / "mlflow.db"
    if db.is_file():
        try:
            registry = read_registry_metadata(db)
            all_champions = all(item["champion_registrado"] for item in registry["modelos"].values())
            check("Model Registry (metadatos SQLite)", "ok" if all_champions else "error",
                  "Alias champion presente en P1/P2/P3 en este SQLite; confirmar carga de artefactos en MLflow",
                  db.name)
        except (sqlite3.Error, OSError) as exc:
            check("Model Registry (metadatos SQLite)", "error",
                  f"No se pudieron consultar metadatos: {type(exc).__name__}")
    else:
        check("Model Registry (metadatos SQLite)", "pendiente",
              "No se adjuntó mlflow.db (correcto para Git); verificar Registry remoto o local con verify_models")

    cards = [f"docs/model_cards/MODEL_CARD_{key.upper()}.md" for key in MODEL_NAMES]
    missing = present(cards)
    check("Model Cards y gobernanza", "ok" if not missing else "pendiente",
          "Model Cards disponibles; requieren aprobación humana" if not missing
          else f"Faltan: {', '.join(missing)}")

    code_monitor = ["tc5061_proyecto_equipo19/monitoring/drift.py",
                    "tc5061_proyecto_equipo19/monitoring/ab.py",
                    "tc5061_proyecto_equipo19/monitoring/report.py"]
    missing = present(code_monitor)
    check("Implementación Evidently y A/B offline", "ok" if not missing else "error",
          "Código presente, NO acredita reportes reales" if not missing else f"Faltan: {', '.join(missing)}")

    evidence_e8 = ["reports/entregable8/evidently_drift_calidad.html",
                   "reports/entregable8/resumen_drift.json",
                   "reports/entregable8/resultado_ab.json",
                   "reports/entregable8/informe_entregable8.md"]
    missing = present(evidence_e8)
    check("Evidencia real drift + A/B", "pendiente" if missing else "ok",
          "Faltan (omitidos del ZIP): " + ", ".join(missing) if missing
          else "Informes y métricas generados disponibles")

    load_code = ["load_testing/locustfile.py", "load_testing/benchmark.py",
                 "load_testing/consolidate.py", "load_testing/report_finops.py"]
    missing = present(load_code)
    check("Locust y FinOps - implementación", "ok" if not missing else "error",
          "Scripts presentes" if not missing else "Faltan: " + ", ".join(missing))
    results_e7 = ["reports/entregable7/RESUMEN_COMPARATIVO.md",
                  "reports/entregable7/INFORME_ENTREGABLE7.md"]
    missing = present(results_e7)
    check("Locust y FinOps - evidencia medida", "pendiente" if missing else "ok",
          "Faltan (omitidos del ZIP): " + ", ".join(missing) if missing
          else "Informes disponibles; revisar origen de mediciones")

    data = root / "data/processed"
    n_parquet = len(list(data.glob("trips_*.parquet"))) if data.is_dir() else 0
    check("Datos históricos / reproducción punta a punta", "pendiente" if n_parquet < 21 else "ok",
          f"{n_parquet} archivo(s) parquet mensual(es) en ZIP; requiere restaurar datasets para reentrenar")
    notebooks = list((root / "notebooks").glob("*.ipynb"))
    check("Notebooks y evidencias visuales", "pendiente" if not notebooks else "ok",
          f"{len(notebooks)} notebook(s) en el paquete; fueron retirados voluntariamente" if not notebooks
          else f"{len(notebooks)} notebook(s) incluidos")

    return {"proyecto": "TC5061-Proyecto-Equipo19", "tipo": "auditoria_estatica_zip",
            "nota": "Un check 'ok' de código no prueba funcionamiento real ni CI en verde.",
            "conteo": {s: sum(x["estado"] == s for x in checks) for s in ("ok", "pendiente", "error")},
            "criterios": checks, "registry_sqlite": registry}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--db", type=Path, default=None,
                        help="SQLite local opcional (no empaquetar para Git)")
    parser.add_argument("--json", type=Path, default=None, help="Guardar resumen sin secretos")
    parser.add_argument("--strict", action="store_true", help="Exigir todas las evidencias")
    args = parser.parse_args(argv)
    result = audit(args.root, args.db)
    for item in result["criterios"]:
        print(f"[{item['estado'].upper():9s}] {item['criterio']}: {item['detalle']}")
    print("Resumen:", result["conteo"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Reporte JSON:", args.json)
    return int(result["conteo"]["error"] > 0 or
               (args.strict and result["conteo"]["pendiente"] > 0))


if __name__ == "__main__":
    raise SystemExit(main())
