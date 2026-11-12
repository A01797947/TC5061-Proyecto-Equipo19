"""Reportes Evidently 0.7: deriva por variable y calidad comparada.

Snapshot dict aporta exactamente los scores calculados por Evidently;
no se reconstruyen tests o decisiones con una fórmula ajena.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from tc5061_proyecto_equipo19.features.p1 import P1_FEATURE_COLUMNS
from tc5061_proyecto_equipo19.monitoring.data import TARGET

DRIFT_THRESHOLD = 0.1  # Jensen-Shannon distance, versión 0.7
DRIFT_SHARE = 0.5
DRIFT_COLUMNS = list(P1_FEATURE_COLUMNS) + [TARGET]


def sample_equal(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Muestreo uniforme reproducible de filas estación/día sin reemplazo."""
    if n < 10:
        raise ValueError("--max-rows requiere un valor >= 10")
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def _parse_snapshot(snapshot_dict: dict) -> tuple[dict, pd.DataFrame]:
    """Extrae métricas ValueDrift desde la serialización oficial de Evidently 0.7.

    Fallamos explícitamente ante cambios de esquema: no marcar 'sin drift'
    cuando los resultados no se pudieron leer.
    """
    metrics = snapshot_dict.get("metrics", [])
    columns = []
    overall = {}
    for metric in metrics:
        name = str(metric.get("metric_name", ""))
        config = metric.get("config") or {}
        value = metric.get("value")
        if "DriftedColumnsCount" in name and isinstance(value, dict):
            overall = {
                "drifted_columns": int(value.get("count", 0)),
                "drift_share": float(value.get("share", 0)),
            }
        if "ValueDrift" not in name:
            continue
        column = config.get("column")
        if isinstance(column, dict):
            column = column.get("name") or column.get("column")
        if not column or value is None:
            continue
        try:
            score = float(value)
            threshold = float(config.get("threshold") or DRIFT_THRESHOLD)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        columns.append({
            "variable": str(column),
            "metodo": str(config.get("method") or "jensenshannon"),
            "score": score,
            "umbral": threshold,
            "drift": bool(score >= threshold),
            "es_temporal_contextual": str(column) == "Año",
        })
    result = pd.DataFrame(columns)
    expected = set(DRIFT_COLUMNS)
    found = set(result["variable"]) if not result.empty else set()
    if not expected.issubset(found):
        raise RuntimeError(
            "La salida de Evidently cambió o faltan métricas ValueDrift. "
            f"Esperadas: {sorted(expected)}; obtenidas: {sorted(found)}. "
            "Revisa el JSON/HTML de Evidently y su versión instalada."
        )
    result = result.drop_duplicates("variable").sort_values(["drift", "score"], ascending=False)
    if not overall:
        overall = {"drifted_columns": int(result["drift"].sum()),
                   "drift_share": float(result["drift"].mean())}
    overall["dataset_drift"] = bool(overall["drift_share"] >= DRIFT_SHARE)
    return overall, result.reset_index(drop=True)


def quality_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Calidad sin muestreo: nulos, cardinalidad y duplicados estación/día."""
    rows = []
    for name, df in (("referencia", reference), ("actual", current)):
        for col in DRIFT_COLUMNS:
            rows.append({
                "periodo": name, "variable": col, "filas": len(df),
                "nulos": int(df[col].isna().sum()),
                "porcentaje_nulos": round(100 * df[col].isna().mean(), 4),
                "valores_unicos": int(df[col].nunique(dropna=True)),
                "filas_estacion_dia_duplicadas": int(df.duplicated(P1_FEATURE_COLUMNS).sum()),
            })
    return pd.DataFrame(rows)


def generate_drift(reference: pd.DataFrame, current: pd.DataFrame, out: Path,
                   max_rows: int = 10000, seed: int = 42) -> dict:
    """Guarda Evidently HTML/JSON, métricas por columna y tabla de calidad.

    Los tipos se fijan: estación, año y calendario categóricos; demanda numérica.
    Año es contextual (2024→2026) y NO dispara reentrenamiento por sí solo.
    """
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset
    except ImportError as exc:
        raise RuntimeError(
            "Falta Evidently. Instala: python -m pip install -r requirements-monitoring.txt"
        ) from exc
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("referencia", reference), ("actual", current)):
        if df.empty or df[DRIFT_COLUMNS].isna().all().any():
            raise ValueError(f"{name} vacío o con columna completamente nula")
    quality_table(reference, current).to_csv(out / "calidad_datos.csv", index=False)
    ref_sample = sample_equal(reference[DRIFT_COLUMNS], max_rows, seed)
    curr_sample = sample_equal(current[DRIFT_COLUMNS], max_rows, seed)
    # Se asigna 'Año' como categórico para detectar cambios deliberados de año.
    schema = DataDefinition(
        categorical_columns=["Año", "Mes", "Dia", "Dia_semana", "origin_station_id"],
        numerical_columns=[TARGET],
    )
    ref = Dataset.from_pandas(ref_sample, data_definition=schema)
    curr = Dataset.from_pandas(curr_sample, data_definition=schema)
    report = Report([
        DataDriftPreset(columns=DRIFT_COLUMNS, method="jensenshannon",
                        threshold=DRIFT_THRESHOLD, drift_share=DRIFT_SHARE),
        DataSummaryPreset(columns=DRIFT_COLUMNS),
    ], include_tests=True)
    # Evidently API 0.7: current primero, reference segundo.
    snapshot = report.run(curr, ref)
    snapshot.save_html(str(out / "evidently_drift_calidad.html"))
    raw = snapshot.dict()
    (out / "evidently_snapshot.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    overall, by_column = _parse_snapshot(raw)
    by_column.to_csv(out / "drift_por_variable.csv", index=False)
    result = {
        "reference_rows": len(reference), "current_rows": len(current),
        "sample_reference": len(ref_sample), "sample_current": len(curr_sample),
        "method": "jensenshannon", "threshold": DRIFT_THRESHOLD,
        "seed": seed, "max_rows": max_rows, **overall,
        "actionable_drift_variables": by_column.loc[
            by_column["drift"] & ~by_column["es_temporal_contextual"], "variable"
        ].tolist(),
        "actionable_input_features": by_column.loc[
            by_column["drift"] & by_column["variable"].isin(
                ["Mes", "Dia", "Dia_semana", "origin_station_id"]
            ), "variable"
        ].tolist(),
        "target_drift": bool(by_column.loc[
            by_column["variable"] == TARGET, "drift"
        ].iloc[0]),
        "notas": [
            "Año cambia por diseño: su drift no se interpreta como degradación por sí solo.",
            "La muestra es uniforme por fila estación/día y no representa estaciones sin viajes.",
            "Cambios en variables explicativas no demuestran caída de desempeño; validar A/B.",
        ],
    }
    (out / "resumen_drift.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
