"""P1 por mes: misma limpieza/ingeniería de variables que entrenamiento original.

Se leen Parquet mensuales con proyección de cuatro columnas para evitar cargar
millones de filas y estaciones en memoria. Nunca se agrupa antes de limpiar.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.features.base import (
    agregar_variables_temporales,
    limpiar_datos_viajes,
)
from tc5061_proyecto_equipo19.features.p1 import P1_FEATURE_COLUMNS, construir_demanda_diaria

TRIP_COLUMNS = ["trip_id", "start_time", "end_time", "origin_station_id"]
TARGET = "cantidad_viajes"
REFERENCE_MONTHS = ("2024_07", "2024_08", "2024_09")
CURRENT_MONTHS = ("2026_07", "2026_08", "2026_09")
CANDIDATE_RECENT_MONTHS = ("2026_07", "2026_08")
EVALUATION_MONTHS = ("2026_09",)
HISTORICAL_TRAIN_MONTHS = tuple(
    f"{year}_{month:02d}" for year, last_month in ((2024, 12), (2025, 6))
    for month in range(1, last_month + 1)
)


def parse_months(text: str) -> tuple[str, ...]:
    """Acepta '2024_07,2024_08' y evita meses duplicados o mal formados."""
    import re

    months = tuple(s.strip() for s in text.split(",") if s.strip())
    if not months or len(months) != len(set(months)):
        raise ValueError("La lista de meses debe contener meses únicos y no estar vacía")
    for month in months:
        if not re.fullmatch(r"20\d{2}_(0[1-9]|1[0-2])", month):
            raise ValueError(f"Mes inválido: {month}. Ejemplo: 2026_09")
    return tuple(sorted(months))


def validate_periods(reference: tuple[str, ...], current: tuple[str, ...],
                     evaluation: tuple[str, ...]) -> None:
    if max(reference) >= min(current):
        raise ValueError("Todos los meses de referencia deben ser anteriores al periodo actual")
    if not set(evaluation).issubset(set(current)):
        raise ValueError("La evaluación debe pertenecer al periodo actual")
    if set(reference) & set(evaluation):
        raise ValueError("Referencia y evaluación deben ser periodos independientes")


def missing_parquets(months: tuple[str, ...], processed_dir: Path) -> list[Path]:
    return [processed_dir / f"trips_{m}.parquet" for m in months
            if not (processed_dir / f"trips_{m}.parquet").is_file()]


def load_p1_months(months: tuple[str, ...], processed_dir: Path | None = None) -> pd.DataFrame:
    """Agrupado observado estación×día; coincide con `construir_demanda_diaria`.

    Cada archivo se comprueba por mes real para evitar contenido de otro periodo.
    Las estaciones con cero viajes NO existen en el P1 original: limitación explícita.
    """
    processed_dir = Path(processed_dir or cfg.PROCESSED_DATA_DIR)
    missing = missing_parquets(months, processed_dir)
    if missing:
        names = ", ".join(p.name for p in missing[:12])
        raise FileNotFoundError(
            f"Faltan {len(missing)} archivos en {processed_dir}: {names}. "
            "Restáuralos en data/processed o ejecuta la ingesta de esos meses. "
            "Este Entregable NO descarga datos ni usa fechas sustitutas de manera silenciosa."
        )
    frames: list[pd.DataFrame] = []
    for month in months:
        path = processed_dir / f"trips_{month}.parquet"
        trips = pd.read_parquet(path, columns=TRIP_COLUMNS)
        year, mo = (int(x) for x in month.split("_"))
        # La lógica se alinea con la limpieza empleada por features/base.py.
        # P1 histórico solo descarta fechas inválidas/negativas y duplicados.
        # No se introduce aquí el corte opcional de 24h de mascara_viajes_consistentes.
        trips = trips.loc[(trips["start_time"].dt.year == year) &
                          (trips["start_time"].dt.month == mo)]
        trips = limpiar_datos_viajes(trips)
        if trips.empty:
            raise ValueError(f"No hay viajes válidos en {path.name}")
        trips["origin_station_id"] = trips["origin_station_id"].astype("string").str.strip()
        trips = trips.loc[trips["origin_station_id"].notna() &
                          (trips["origin_station_id"] != "")]
        if trips.empty:
            raise ValueError(f"No hay estaciones de origen válidas en {path.name}")
        enriched = agregar_variables_temporales(trips)
        _, _, grouped = construir_demanda_diaria(enriched)
        frames.append(grouped)
    result = pd.concat(frames, ignore_index=True)
    result = result[P1_FEATURE_COLUMNS + [TARGET]].copy()
    for col in ("Año", "Mes", "Dia", "Dia_semana"):
        result[col] = result[col].astype("int32")
    result["origin_station_id"] = result["origin_station_id"].astype(str)
    result[TARGET] = result[TARGET].astype("int64")
    assert not result.duplicated(subset=P1_FEATURE_COLUMNS).any(), "Estación-día duplicado"
    return result.sort_values(P1_FEATURE_COLUMNS).reset_index(drop=True)


def dataframe_to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if df.empty:
        raise ValueError("Datos vacíos")
    required = P1_FEATURE_COLUMNS + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Columnas faltantes P1: {missing}")
    x = df[P1_FEATURE_COLUMNS].copy()
    x["origin_station_id"] = x["origin_station_id"].astype(str)
    return x, pd.to_numeric(df[TARGET], errors="raise")


def data_fingerprint(months: tuple[str, ...], processed_dir: Path | None = None) -> dict:
    """Huella SHA256 del contenido exacto (archivos de hasta cientos de MB)."""
    processed_dir = Path(processed_dir or cfg.PROCESSED_DATA_DIR)
    files = missing_parquets(months, processed_dir)
    if files:
        raise FileNotFoundError(f"No existe: {files[0]}")
    entries = []
    combined = hashlib.sha256()
    for month in sorted(set(months)):
        path = processed_dir / f"trips_{month}.parquet"
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(2**20), b""):
                h.update(chunk)
        entry = {"month": month, "sha256": h.hexdigest(), "bytes": path.stat().st_size}
        entries.append(entry)
        combined.update(json.dumps(entry, sort_keys=True).encode())
    return {"version": combined.hexdigest()[:16], "sources": entries}
