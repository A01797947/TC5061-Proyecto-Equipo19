"""Ingesta y normalización de los datos de MiBici."""

from __future__ import annotations
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from tc5061_proyecto_equipo19.config import (
    RAW_DIR,
    PROCESSED_DIR,
    REFERENCES_DIR,
    MANIFEST_PATH,
    CSV_ENCODING,
    PARQUET_COMPRESSION,
    DATA_SCHEMA_VERSION
)

logger = logging.getLogger(__name__)

# Nombres originales aceptados para cada campo canónico.
TRIP_COLUMNS = {
    "Viaje_Id": "trip_id",
    "Usuario_Id": "user_id",
    "Genero": "gender",
    "Año_de_nacimiento": "birth_year",
    "AÃ±o_de_nacimiento": "birth_year",
    "Inicio_del_viaje": "start_time",
    "Fin_del_viaje": "end_time",
    "Origen_Id": "origin_station_id",
    "Destino_Id": "destination_station_id",
}

STATION_COLUMNS = {
    "id": "station_id",
    "name": "station_name",
    "latitude": "latitude",
    "longitude": "longitude",
    "dpcapacity": "capacity",
}

# Columnas mínimas para el análisis inicial.
ANALYTICAL_TRIP_COLUMNS = [
    "trip_id",
    "start_time",
    "end_time",
    "origin_station_id",
    "destination_station_id",
]


def sha256_file(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo por bloques."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()



def read_csv_checked(path: Path) -> pd.DataFrame:
    """Lee un CSV probando codificaciones conocidas y registra cuál funcionó."""

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding, encoding_errors="strict", low_memory=False)
            logger.info("Archivo %s leído correctamente con %s", path.name, encoding)
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"No se pudo leer {path.name} con las codificaciones evaluadas.")


def normalize_columns(df: pd.DataFrame, mapping: dict[str, str], required: list[str],) -> pd.DataFrame:
    """Normaliza nombres de columnas y valida campos obligatorios."""
    result = df.copy()
    result.columns = [str(column).strip() for column in result.columns]

    # No permitir que dos columnas originales terminen con el mismo nombre.
    normalized = [mapping.get(column, column) for column in result.columns]

    if len(normalized) != len(set(normalized)):
        raise ValueError("Hay columnas duplicadas después de normalizar los nombres.")

    result.columns = normalized

    missing = sorted(set(required) - set(result.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}. Columnas encontradas: {result.columns.tolist()}")

    return result


def process_trips(path: Path) -> tuple[Path, dict]:
    """Normaliza un archivo mensual de viajes y lo guarda en Parquet."""
    df = read_csv_checked(path)

    df = normalize_columns(df, TRIP_COLUMNS, ANALYTICAL_TRIP_COLUMNS,)

    # Conservar solo campos necesarios para el análisis inicial.
    df = df[ANALYTICAL_TRIP_COLUMNS].copy()

    # Identificadores como texto para evitar problemas de tipos y ceros.
    for column in [
        "trip_id",
        "origin_station_id",
        "destination_station_id",
    ]:
        df[column] = df[column].astype("string").str.strip()

    # Conversión explícita de fechas.
    for column in ["start_time", "end_time"]:
        df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")

    invalid_dates = int((df["start_time"].isna() | df["end_time"].isna()).sum())

    # Detectar registros duplicados sin eliminarlos silenciosamente.
    duplicate_ids = int(df["trip_id"].duplicated(keep=False).sum())

    # Los casos problemáticos deben analizarse y documentarse.
    output_name = path.stem.replace("datos_abiertos_", "trips_") + ".parquet"

    output_path = PROCESSED_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False, compression=PARQUET_COMPRESSION)

    metadata = {
        "source_file": path.name,
        "source_sha256": sha256_file(path),
        "output_file": output_path.name,
        "output_sha256": sha256_file(output_path),
        "records": int(len(df)),
        "invalid_date_rows": invalid_dates,
        "rows_with_duplicate_trip_ids": duplicate_ids,
        "start_time_min": (str(df["start_time"].min()) if df["start_time"].notna().any() else None),
        "start_time_max": (str(df["start_time"].max()) if df["start_time"].notna().any() else None),
    }

    logger.info("Procesado %s: %s registros", path.name, len(df))

    return output_path, metadata


def process_stations(path: Path) -> tuple[Path, dict]:
    """Normaliza el catálogo de estaciones y lo guarda en Parquet."""
    df = read_csv_checked(path)

    df = normalize_columns(df, STATION_COLUMNS, list(STATION_COLUMNS.values()))

    df = df[list(STATION_COLUMNS.values())].copy()

    df["station_id"] = df["station_id"].astype("string").str.strip()
    df["station_name"] = df["station_name"].astype("string").str.strip()

    for column in ["latitude", "longitude", "capacity"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    output_path = PROCESSED_DIR / "stations.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False, compression=PARQUET_COMPRESSION)

    metadata = {
        "source_file": path.name,
        "source_sha256": sha256_file(path),
        "output_file": output_path.name,
        "output_sha256": sha256_file(output_path),
        "records": int(len(df)),
        "duplicate_station_ids": int(df["station_id"].duplicated(keep=False).sum()),
        "missing_coordinates": int((df["latitude"].isna() | df["longitude"].isna()).sum())
    }

    logger.info("Procesadas %s estaciones", len(df))
    return output_path, metadata


def generate_manifest(trip_metadata: list[dict], station_metadata: dict | None) -> None:
    """Registra la versión de datos, hashes y estadísticas de ingesta."""
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trips": trip_metadata,
        "stations": station_metadata,
    }

    # Escritura atómica para evitar dejar un JSON incompleto.
    temporary_path = MANIFEST_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(MANIFEST_PATH)


def run_ingestion() -> None:
    """Procesa los CSV locales y genera Parquet y manifiesto."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    trip_files = sorted(RAW_DIR.glob("datos_abiertos_*.csv"))

    if not trip_files:
        raise FileNotFoundError(f"No se encontraron datos_abiertos_*.csv en {RAW_DIR}")

    station_files = sorted(RAW_DIR.glob("nomenclatura_*.csv"))

    if len(station_files) > 1:
        raise ValueError("Se encontraron varios catálogos de estaciones. Deja uno seleccionado para esta primera versión.")

    trip_metadata = [process_trips(path)[1] for path in trip_files]

    station_metadata = None
    
    if station_files:
        station_metadata = process_stations(station_files[0])[1]

    generate_manifest(trip_metadata, station_metadata)

    logger.info("Ingesta completada. Manifiesto: %s", MANIFEST_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_ingestion()