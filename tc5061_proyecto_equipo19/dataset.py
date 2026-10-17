"""Ingesta y normalización de los datos de MiBici."""

from __future__ import annotations
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import re
import time

import pandas as pd
import requests


from tc5061_proyecto_equipo19.config import (
    DATA_SCHEMA_VERSION,
    DOWNLOAD_RETRIES,
    DOWNLOAD_TIMEOUT_SECONDS, 
    DOWNLOAD_URL_OVERRIDES,
    DOWNLOAD_USER_AGENT,
    START_MONTH,
    END_MONTH,
    MANIFEST_PATH,
    MIBICI_BASE_URL,
    MIBICI_DATA_PAGE,
    PARQUET_COMPRESSION,
    PARTIAL_MANIFEST_PATH,
    PROCESSED_DIR,
    STATIONS_FILE,
    RAW_DIR,    
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

# Enlace de descarga en al página de MiBici: /site/assets/files/<id>/
_LINK_PATTERN = re.compile(
    r"(/site/assets/files/\d+/((?:datos_abiertos|nomenclatura)_\d{4}_\d{2})[^\"'\s<>]*\.csv)"
)

# Formato válido de mes para el rango de ingesta AAAA-MM
_MONTH_PATTERN = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


# ===================================================
# Hashes y manifest
# ===================================================

def sha256_file(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo por bloques.
    
    Args:
        path: Ruta de archivos a procesar
        
    Return: 
        hash SHA-256 en hexadecimal"""

    
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

def load_manifest(path: Path) -> dict:
    """Lee un manifiesto; devuelve {} si aún no existe
    
    Args: 
        path: Ruta del archivo de manifiesto JSON
    
    Return:
        Contenido del manifiesto o un diccionario vacio si el archvo no existe
        """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def source_hashes(manifest: dict) -> dict[str, str]:
    """Regresa {archivo : sha256} de todos los archivos que registra el manifest
    
    Args:
        manifest: manifiesto leido con load_manifest
        
    Return:
        Diccionario {nombre_del_csv: sha256} con los viajes y el catálogo de estaciones"""
    entries = list(manifest.get("trips", []))
    if manifest.get("stations"):
        entries.append(manifest["stations"])
    return {entry["source_file"]: entry["source_sha256"] for entry in entries}

def generate_manifest(trip_metadata: list[dict], station_metadata: dict | None,
                      path: Path = MANIFEST_PATH, date_range: tuple[str, str] | None = None,
                      ) -> None:
    """Registra la versión de datos, hashes y estadísticas de ingesta."""
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": (
            {"start_month": date_range[0], "end_month": date_range[1]}
        ),
        "trips": trip_metadata,
        "stations": station_metadata,
    }

    # Escritura atómica para evitar dejar un JSON incompleto.
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


# ===================================================
# Lectura y normalización
# ===================================================

def read_csv_checked(path: Path) -> pd.DataFrame:
    """Lee un CSV probando codificaciones conocidas y registra cuál funcionó.
    
    Args: 
        path: archivo CSV a leer
        
    Returns:
        Contenido del csv, sin transformar
        
    Raises:
        
        ValueError: si ninguna de las codificaiones evaluadas puede leer el archivo
        """

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
    """Normaliza nombres de columnas y valida campos obligatorios.
    
    Args:
        df: tabla original
        mapping: {nombre_original: nombre_canónico}. Las columnas que no aparecen mantinene el nombre original
        requiered: nombres canónicos que deben existir después de renombrar
        
    Return:
        Copia de df con las columnas renombradas
    
    Raises:
        ValueError: si dos columnas terminan con el mismo nombre o falta alguna obligatoria"""
    
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
    """Normaliza un archivo mensual de viajes y lo guarda en Parquet.
    
    Args:
        path: csv mensual data/raw
        
    Return: 
        Ruta del parquet generado y un diccionario con hashes y estadísticas para el manifest
        
    Raises:
        ValueError: si el csv no se puede leer o faltan columnas obligatorias"""
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
    """Normaliza el catálogo de estaciones y lo guarda en Parquet.
    
        Args: 
            path: ruta del csv mensual en data/raw
            
        Return;
            Ruta del parquet generado y un diccionario con hashes y estadísticas para el manifiesto
            
        Raises:
            ValueError: Si el csv no se puede leer o le faltan columnas obligatorias"""
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

#======================================
# Descarga
#======================================

def iter_months(start:str, end:str) -> list[str]:
    """Devuelve la lista de meses entre start y end, en formato AAAA-MM
    
    Args:
        start: primer mes, formato AAAA-MM
        end: último mes, formato AAAA-MM
    """

    parsed = []
    for label, value in (("inicio", start), ("fin", end)):
        match = _MONTH_PATTERN.match(value)
        if not match:
            raise ValueError(f"Mes de {label}, inválido: {value}")
        parsed.append((int(match.group(1)), int(match.group(2))))

    (year, month), (end_year, end_month) = parsed
    if (year, month) > (end_year, end_month):
        raise ValueError(f"El mes de inicio {year}-{month} es mayor al final {end_year}-{end_month}")


    months = []
    while(year, month) <= (end_year, end_month):
        months.append(f"{year:04d}_{month:02d}")
        month+=1
        if month > 12:
            year, month = year+1, 1
    return months

def parse_download_links(html: str) -> dict[str, str]:
    """Extrae del HTML de la página {nombre_de_archivo: url}
    
    Args: 
        html_ contenido de la página de datos abiertos de mibici
        
    Return:
        Diccionario {nombre_del_archivo_ url}"""
    
    links = {}
    for path, name in _LINK_PATTERN.findall(html):
        links[f"{name}.csv"] = f"{MIBICI_BASE_URL}{path}"
    return links

def resolve_download_urls(session: requests.Session) -> dict[str, str]:
    """Lee la página de datos y devuelve {nombre_de_archivo: url}
    
    Args:
        session: sesión de requests con el UserAgent
    
    Return:
        Diccionario {nombre_del_archivo: url}. Las entradas de DOWNLOAD_URL_OVERRIDES remplazan las leidas
        
    Raisses:
        request.HTTPError si la página responde con un error HTTP"""
    response = session.get(MIBICI_DATA_PAGE, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()

    urls = parse_download_links(response.text)
    logger.info("MiBici: Total de %s archivos", len(urls))

    urls.update(DOWNLOAD_URL_OVERRIDES)
    return urls

def download_file(url: str, destination: Path, session: requests.Session) -> None:
    """Descarga contenido de url a ubicación de destino y con reintentos
    
        Args:
            url: dirección del archivo
            destination: ruta donde se guarda el archivo
            session: sesión de requests con el user agent

        Raisses:
            requests.RequestException: Si fallan los reintentos

    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try: 
            with session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                response.raise_for_status()
                with partial.open("wb") as file:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        file.write(block)
                partial.replace(destination)
                return
        except requests.RequestException as error:
            logger.warning(
                "Intento %s/%s fallido para %s: %s",
                attempt, DOWNLOAD_RETRIES, destination.name, error
            )
            if attempt == DOWNLOAD_RETRIES:
                raise
            time.sleep(2**attempt)

def ensure_raw_files(file_names: list[str], expected_hashes: dict[str, str])-> None:
    """Asegura que cada csv de file_name se encuentra en la carpeta data/raw
    
        Args: 
            file_names: nombres de los csv necesarios
            expected_hashes: {nombre_de_archivo: sha256} formato del manifiesto

        Raises:
            LookUpError: si la página de MiBici no publica alguno de los archivos
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    missing = [name for name in file_names if not (RAW_DIR / name).exists()]
    if not missing:
        logger.info("Todos los archivos existen en data/raw")
        return
    
    session = requests.Session()
    session.headers["User-Agent"] = DOWNLOAD_USER_AGENT
    urls = resolve_download_urls(session)

    changed = []
    for name in missing:
        if name not in urls:
            raise LookupError(f"No se encontro el archivo {name} en {urls}")
        path = RAW_DIR/name
        logger.info("Descargando %s", name)
        download_file(urls[name], path, session)

        expected_hash = expected_hashes.get(name)
        if expected_hash is not None and sha256_file(path) != expected_hash:
            changed.append(name) 

    if changed:
        logger.warning(f"""Los siguientes archivos no coinciden con el los hashes del manifets,
                MiBici pudo haberlos modificado {changed}""")

# ===================================================
# Flow principal
# ===================================================


def run_ingestion(start: str | None = None, end: str | None = None) -> None:
    """Descarga, normaliza y versiona los datos a partir del rango de meses
    
    Args: 
        start: primer mes, formato 'AAAA-MM' (default: config[START_MONTH])
        end: último mes, formato 'AAAA-MM' (default: config[START_MONTH])

    Return:
        Ruta del manifest escrito
    """
    
    start = start or START_MONTH
    end = end or END_MONTH
    months = iter_months(start, end)

    trip_names = [f"datos_abiertos_{month}.csv" for month in months]

    ensure_raw_files(
        trip_names + [STATIONS_FILE],
        source_hashes(load_manifest(MANIFEST_PATH)),
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    trip_metadata = [process_trips(RAW_DIR/name)[1] for name in trip_names]
    station_metadata = process_stations(RAW_DIR/STATIONS_FILE)[1]

    is_default_range = (start, end) == (START_MONTH, END_MONTH)
    manifest_path = MANIFEST_PATH if is_default_range else PARTIAL_MANIFEST_PATH
    generate_manifest(trip_metadata, station_metadata, manifest_path, (start, end))

    logger.info(f"Ingesta completada: %s meses. Manifesto: %s", len(months), manifest_path)
    return manifest_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_ingestion()