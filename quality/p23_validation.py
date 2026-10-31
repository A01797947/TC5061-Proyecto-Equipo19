"""Calidad específica de P2 (clasificación) y P3 (clustering por estación).

Validación de esquema independiente de DeepChecks para ejecutarse también desde PyTest.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.features.p2 import P2_FEATURE_COLUMNS

P2_TARGET = "y_riesgo"
P2_KEYS = ["station_id", "timestamp"]
P3_ID = "station_id"
P3_REQUIRED = ["station_id", "latitude", "longitude", "capacity",
               "viajes_iniciados", "viajes_finalizados", "total_viajes", "ratio_finde"]
P2_DRIFT_COLUMNS = ["hora", "capacidad", "ocupacion_est_h", "neto_h"]
P3_DRIFT_COLUMNS = ["total_viajes", "ratio_finde", "pct_salidas_hora_08"]


def _base_check(df: pd.DataFrame, name: str, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name}: faltan columnas: {missing}")
    if df.empty:
        raise ValueError(f"{name}: no contiene registros")


def _nonnull(df: pd.DataFrame, name: str, cols: list[str]) -> None:
    if df[cols].isna().any().any():
        raise ValueError(f"{name}: valores faltantes en columnas obligatorias")


def _station_ids(df: pd.DataFrame, name: str) -> None:
    if not is_string_dtype(df["station_id"]):
        raise ValueError(f"{name}: station_id debe ser texto (string)")
    if df["station_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}: station_id vacío")


def validate_p2_table(df: pd.DataFrame, name: str) -> dict:
    """P2 permite nulos en variables históricas (primeros 7 días), no en claves/etiqueta.

    No exige clases 0 y 1 dentro de *cada* partición: es válido un periodo sin riesgo.
    """
    cols = P2_KEYS + P2_FEATURE_COLUMNS + [P2_TARGET]
    _base_check(df, name, cols)
    _nonnull(df, name, P2_KEYS + [P2_TARGET, "capacidad", "ocupacion_est_h"])
    _station_ids(df, name)
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError(f"{name}: timestamp debe ser datetime")
    if df.duplicated(subset=P2_KEYS).any():
        raise ValueError(f"{name}: filas duplicadas por estación y timestamp")
    invalid_types = [c for c in P2_FEATURE_COLUMNS + [P2_TARGET]
                     if not is_numeric_dtype(df[c])]
    if invalid_types:
        raise ValueError(f"{name}: columnas no numéricas: {invalid_types}")
    if not df[P2_TARGET].isin([0, 1]).all():
        raise ValueError(f"{name}: y_riesgo debe ser binaria (0/1)")
    if (df["capacidad"] <= 0).any():
        raise ValueError(f"{name}: capacidad debe ser positiva")
    if not df["ocupacion_est_h"].between(0, 1).all():
        raise ValueError(f"{name}: ocupacion_est_h debe estar entre 0 y 1")
    if not df["hora"].between(0, 23).all() or not df["dia_semana"].between(0, 6).all():
        raise ValueError(f"{name}: valores de calendario fuera de rango")
    for c in ("y_riesgo_vacia", "y_riesgo_llena"):
        if c in df and not df[c].isin([0, 1]).all():
            raise ValueError(f"{name}: {c} debe ser binaria")
    if {"y_riesgo_vacia", "y_riesgo_llena"}.issubset(df.columns):
        logical = (df["y_riesgo_vacia"] == 1) | (df["y_riesgo_llena"] == 1)
        if not np.array_equal(logical.astype(int).to_numpy(), df[P2_TARGET].to_numpy()):
            raise ValueError(f"{name}: etiqueta y_riesgo inconsistente con vacía/llena")
    return {"dataset": name, "schema": "OK", "rows": len(df),
            "missing_feature_cells": int(df[P2_FEATURE_COLUMNS].isna().sum().sum()),
            "duplicate_keys": 0}


def validate_p3_table(df: pd.DataFrame, name: str) -> dict:
    """Perfiles por estación P3, sin target. Valida identidades y frecuencias relativas."""
    _base_check(df, name, P3_REQUIRED)
    _nonnull(df, name, P3_REQUIRED)
    _station_ids(df, name)
    if df.duplicated(subset=[P3_ID]).any():
        raise ValueError(f"{name}: estaciones duplicadas")
    numerical = [c for c in df.columns if c != P3_ID]
    invalid_types = [c for c in numerical if not is_numeric_dtype(df[c])]
    if invalid_types:
        raise ValueError(f"{name}: columnas no numéricas: {invalid_types}")
    if df[numerical].isna().any().any():
        raise ValueError(f"{name}: valores faltantes en perfil de estación")
    if (df["capacity"] <= 0).any():
        raise ValueError(f"{name}: capacity debe ser positiva")
    if (df[["viajes_iniciados", "viajes_finalizados", "total_viajes"]] < 0).any().any():
        raise ValueError(f"{name}: volumen de viajes negativo")
    if not np.allclose(df["viajes_iniciados"] + df["viajes_finalizados"],
                       df["total_viajes"], rtol=1e-6, atol=1e-7):
        raise ValueError(f"{name}: total_viajes no coincide con iniciados + finalizados")
    if (df["total_viajes"] <= 0).any():
        raise ValueError(f"{name}: total_viajes debe ser positivo")
    if not df["ratio_finde"].between(0, cfg.P3_RATIO_FINDE_MAX + 1e-6).all():
        raise ValueError(f"{name}: ratio_finde fuera de rango")
    for prefix in ("pct_salidas_hora_", "pct_llegadas_hora_"):
        cols = sorted(c for c in df.columns if c.startswith(prefix))
        if not cols:
            raise ValueError(f"{name}: faltan proporciones {prefix}")
        if not df[cols].ge(0).all().all() or not df[cols].le(1 + 1e-7).all().all():
            raise ValueError(f"{name}: proporciones horarias fuera de rango")
        sums = df[cols].sum(axis=1)
        # Estaciones que no registraron salidas/llegadas pueden sumar cero.
        if not (np.isclose(sums, 1, atol=1e-5) | np.isclose(sums, 0, atol=1e-5)).all():
            raise ValueError(f"{name}: proporciones horarias no suman 1 ni 0")
    return {"dataset": name, "schema": "OK", "rows": len(df),
            "duplicate_stations": 0, "missing_feature_cells": 0}


def demo_p2_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel pequeño pero diverso; prueba técnica, nunca dato real de MiBici."""
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    def make(start):
        dates = pd.date_range(start, periods=45, freq="h")
        n = len(dates) * 5
        times = pd.DatetimeIndex(np.tile(dates.to_numpy(), 5))
        stations = np.repeat([f"{i:03d}" for i in range(5)], len(dates))
        hours = times.hour.to_numpy()
        occupancy = np.clip(rng.normal(0.5, 0.22, n), 0, 1)
        target = (occupancy < 0.2) | (occupancy > 0.8)
        sample = pd.DataFrame({c: rng.normal(size=n).astype(float) for c in P2_FEATURE_COLUMNS})
        sample["station_id"] = pd.Series(stations, dtype="string")
        sample["timestamp"] = times
        sample["hora"] = hours
        sample["dia_semana"] = times.weekday
        sample["mes"] = times.month
        sample["fin_de_semana"] = (times.weekday >= 5).astype(int)
        sample["feriado"] = 0
        sample["hora_sin"] = np.sin(2 * np.pi * hours / 24)
        sample["hora_cos"] = np.cos(2 * np.pi * hours / 24)
        sample["capacidad"] = np.repeat([16, 18, 20, 22, 25], len(dates))
        sample["latitud"] = 20.6
        sample["longitud"] = -103.3
        sample["ocupacion_est_h"] = occupancy
        sample["y_riesgo"] = target.astype(int)
        sample["y_riesgo_vacia"] = (occupancy < 0.2).astype(int)
        sample["y_riesgo_llena"] = (occupancy > 0.8).astype(int)
        return sample

    return make("2025-02-01"), make("2025-10-01")


def demo_p3_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """60 perfiles válidos por periodo; proporciones horarias normalizadas."""
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    def make():
        count = 60
        started = rng.integers(30, 150, size=count).astype(float)
        ended = rng.integers(25, 140, size=count).astype(float)
        frame = pd.DataFrame({
            "station_id": pd.Series([f"{i:03d}" for i in range(count)], dtype="string"),
            "viajes_iniciados": started,
            "viajes_finalizados": ended,
            "total_viajes": started + ended,
            "latitude": rng.uniform(20.55, 20.75, count),
            "longitude": rng.uniform(-103.5, -103.2, count),
            "capacity": rng.integers(10, 35, size=count),
            "ratio_finde": rng.uniform(0.2, 1.3, count),
        })
        for prefix in ("pct_salidas_hora_", "pct_llegadas_hora_"):
            probs = rng.dirichlet(np.ones(24) * 2, count)
            for hour in range(24):
                frame[f"{prefix}{hour:02d}"] = probs[:, hour]
        return frame

    return make(), make()


def run_p23_validation(train: pd.DataFrame, test: pd.DataFrame, report_dir: Path,
                       *, problem: str, mode: str, drift_threshold: float = 0.2,
                       fail_on_drift: bool = False) -> dict:
    """Ejecuta validaciones de esquema, DeepChecks de integridad y de drift por P2/P3."""
    try:
        from deepchecks.tabular import Dataset, Suite
        from deepchecks.tabular.checks import (
            DataDuplicates, FeatureDrift, MixedDataTypes, PercentOfNulls,
        )
    except ImportError as exc:
        raise RuntimeError("Instala DeepChecks en el entorno Python 3.10") from exc
    if problem == "P2":
        checker, target, drift_cols = validate_p2_table, P2_TARGET, P2_DRIFT_COLUMNS
        feature_cols = P2_FEATURE_COLUMNS
        categorical = ["hora", "dia_semana"]
    elif problem == "P3":
        checker, target, drift_cols = validate_p3_table, None, P3_DRIFT_COLUMNS
        # Se retienen solo atributos de estación, no etiquetas o columnas de timestamp.
        feature_cols = [c for c in train.columns if c != P3_ID]
        categorical = []
    else:
        raise ValueError("problem debe ser P2 o P3")
    checks = [checker(train, "train"), checker(test, "test")]
    if set(train.columns) != set(test.columns):
        raise ValueError("train y test no contienen las mismas columnas")
    if not set(drift_cols).issubset(train.columns):
        raise ValueError(f"Faltan variables de drift {sorted(set(drift_cols) - set(train.columns))}")
    if problem == "P2":
        tr = train[feature_cols + [target]].copy()
        te = test[feature_cols + [target]].copy()
    else:
        tr = train[feature_cols].copy()
        te = test[feature_cols].copy()
    kwargs = {"cat_features": categorical}
    if target:
        kwargs["label"] = target
    train_ds, test_ds = Dataset(tr, **kwargs), Dataset(te, **kwargs)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    suite = Suite(f"MiBici {problem} - integridad", DataDuplicates(), PercentOfNulls(), MixedDataTypes())
    html = []
    for name, dataset in (("train", train_ds), ("test", test_ds)):
        report = suite.run(dataset)
        filename = f"integridad_{name}.html"
        report.save_as_html(str(report_dir / filename))
        html.append(filename)
    result = FeatureDrift(columns=drift_cols).run(train_dataset=train_ds, test_dataset=test_ds)
    result.save_as_html(str(report_dir / "drift_train_test.html"))
    html.append("drift_train_test.html")
    scores = {
        key: float(value["Drift score"])
        for key, value in result.value.items()
        if isinstance(value, dict) and "Drift score" in value
    }
    above = {key: value for key, value in scores.items() if value > drift_threshold}
    summary = {
        "problem": problem, "mode": mode, "validation": checks,
        "deepchecks": "ejecutado", "target": target,
        "drift_features": drift_cols, "feature_drift_scores": scores,
        "drift_threshold": drift_threshold, "drift_above_threshold": above,
        "drift_policy": "bloqueante" if fail_on_drift else "advertencia",
        "html_reports": html,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    if above and fail_on_drift:
        raise ValueError(f"{problem}: drift superior al umbral en {above}")
    return summary
