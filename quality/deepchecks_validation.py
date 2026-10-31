"""Validación P1/P2/P3 con DeepChecks 0.19.1 (Python 3.10 aislado).

El informe detecta drift sin asumir que los meses train/test tienen la misma
estacionalidad. Los errores de esquema, nulos y duplicados son bloqueantes.

Uso:
    python -m quality.deepchecks_validation --demo
    python -m quality.deepchecks_validation --train data/interim/quality_train_p1.csv \
        --test data/interim/quality_test_p1.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quality.data_validation import REQUIRED_FEATURES, TARGET, validate_p1_table


def demo_p1_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixtures reproducibles y pequeñas; NO son evidencia sobre MiBici real."""
    rng = np.random.default_rng(42)

    def make(start: str, days: int) -> pd.DataFrame:
        dates = pd.date_range(start, periods=days)
        date_index = np.repeat(dates, 2)
        station = np.tile(["101", "102"], days)
        return pd.DataFrame({
            "Año": date_index.year,
            "Mes": date_index.month,
            "Dia": date_index.day,
            "Dia_semana": date_index.weekday,
            "origin_station_id": station,
            "cantidad_viajes": 10 + 2 * (station == "101").astype(int)
            + (date_index.weekday < 5).astype(int) * 4
            + rng.integers(0, 3, size=days * 2),
        })
    return make("2025-01-01", 70), make("2025-10-01", 70)


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de validación: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"origin_station_id": "string", "station_id": "string"})
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Se aceptan únicamente archivos .csv o .parquet")


def run_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    report_dir: Path,
    *,
    mode: str,
    drift_threshold: float = 0.2,
    fail_on_drift: bool = False,
) -> dict:
    """Prevalidación estricta + DeepChecks: esquema, nulos, duplicados y drift."""
    # Import tardío para que sea explícita la necesidad de un entorno dedicado.
    try:
        from deepchecks.tabular import Dataset, Suite
        from deepchecks.tabular.checks import (
            DataDuplicates, FeatureDrift, MixedDataTypes, PercentOfNulls,
        )
    except ImportError as exc:
        raise RuntimeError(
            "DeepChecks no está disponible. Activa el entorno Python 3.10 e instala "
            "requirements-deepchecks.txt."
        ) from exc

    checks = [validate_p1_table(train, "train"), validate_p1_table(test, "test")]
    if set(train.columns) != set(test.columns):
        raise ValueError("Las columnas de train y test no coinciden")
    # Importante: no corregir silenciosamente anomalías o registros corruptos.
    train = train[REQUIRED_FEATURES + [TARGET]].copy()
    test = test[REQUIRED_FEATURES + [TARGET]].copy()
    for part in (train, test):
        part["origin_station_id"] = part["origin_station_id"].astype(str)
        part["Dia_semana"] = part["Dia_semana"].astype(int)

    dataset_kwargs = {"label": TARGET, "cat_features": ["origin_station_id", "Dia_semana"]}
    train_ds = Dataset(train, **dataset_kwargs)
    test_ds = Dataset(test, **dataset_kwargs)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    suite = Suite("MiBici P1 - integridad",
                  DataDuplicates(), PercentOfNulls(), MixedDataTypes())
    results = {}
    for name, ds in (("train", train_ds), ("test", test_ds)):
        result = suite.run(ds)
        result.save_as_html(str(report_dir / f"integridad_{name}.html"))
        results[name] = f"integridad_{name}.html"

    # Mes y año son distintos POR DISEÑO en una partición temporal, por lo que
    # reportarlos como 'drift accidental' generaría falsos positivos.
    drift = FeatureDrift(columns=["origin_station_id", "Dia_semana"])
    drift_result = drift.run(train_dataset=train_ds, test_dataset=test_ds)
    drift_result.save_as_html(str(report_dir / "drift_train_test.html"))
    scores = {
        col: float(info["Drift score"])
        for col, info in drift_result.value.items()
        if isinstance(info, dict) and "Drift score" in info
    }
    exceeds = {col: score for col, score in scores.items() if score > drift_threshold}
    summary = {
        "mode": mode,
        "checks": checks,
        "deepchecks": "ejecutado",
        "feature_drift_scores": scores,
        "drift_threshold": drift_threshold,
        "drift_above_threshold": exceeds,
        "drift_policy": "bloqueante" if fail_on_drift else "advertencia",
        "html_reports": [*results.values(), "drift_train_test.html"],
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if exceeds and fail_on_drift:
        raise ValueError(f"Drift excesivo: {exceeds}. Revisa {report_dir}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["P1", "P2", "P3"], default="P1",
                        help="Pregunta del proyecto (por defecto P1)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Datos sintéticos de prueba")
    source.add_argument("--train", type=Path, help="CSV/Parquet P1 de entrenamiento")
    parser.add_argument("--test", type=Path, help="CSV/Parquet P1 de prueba")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/quality"))
    parser.add_argument("--drift-threshold", type=float, default=0.2)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()
    if args.demo and args.test:
        parser.error("--test no aplica junto con --demo")
    if args.train and not args.test:
        parser.error("Indica --test cuando uses --train")
    if args.problem != "P1":
        from quality.p23_validation import (
            demo_p2_tables, demo_p3_tables, run_p23_validation,
        )
        if args.demo:
            train, test = (demo_p2_tables() if args.problem == "P2"
                           else demo_p3_tables())
            mode = "demo_sintetica"
        else:
            train, test = load_table(args.train), load_table(args.test)
            mode = "datos_suministrados"
            if args.problem == "P2":
                # El exportador P2 escribe ISO8601; convertir al cargar CSV.
                for frame in (train, test):
                    frame["station_id"] = frame["station_id"].astype("string")
                    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
            else:
                for frame in (train, test):
                    frame["station_id"] = frame["station_id"].astype("string")
        report_dir = (Path("reports/quality") / args.problem.lower()
                      if args.report_dir == Path("reports/quality") else args.report_dir)
        result = run_p23_validation(
            train, test, report_dir, problem=args.problem, mode=mode,
            drift_threshold=args.drift_threshold,
            fail_on_drift=args.fail_on_drift,
        )
    else:
        if args.demo:
            train, test = demo_p1_tables()
            mode = "demo_sintetica"
        else:
            train, test = load_table(args.train), load_table(args.test)
            mode = "datos_suministrados"
        result = run_validation(train, test, args.report_dir, mode=mode,
                                drift_threshold=args.drift_threshold,
                                fail_on_drift=args.fail_on_drift)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
