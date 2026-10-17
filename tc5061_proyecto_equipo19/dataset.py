"""Ingesta reproducible y versionada de los datos de MiBici.

Uso desde la terminal (en la raíz del repositorio):

    python -m tc5061_proyecto_equipo19.dataset              # descarga + procesa + manifiesto
    python -m tc5061_proyecto_equipo19.dataset descargar    # solo descarga a data/raw
    python -m tc5061_proyecto_equipo19.dataset procesar     # solo CSV -> Parquet + manifiesto

Etapas:
1. Descarga los meses definidos en ``config.MESES`` y la nomenclatura de estaciones
   en ``data/raw`` (los archivos ya descargados no se vuelven a bajar).
2. Lee cada CSV detectando su codificación, normaliza nombres de columnas
   (acentos, mayúsculas, texto mal codificado) y convierte fechas a ``datetime``.
3. Excluye los datos personales (usuario, género y año de nacimiento) y guarda
   los datos limpios en Parquet en ``data/processed``.
4. Escribe ``references/data_manifest.json`` con el número de registros y el
   hash SHA-256 de cada archivo. La versión de datos (``data_version``) es un hash
   corto del contenido de todos los Parquet y se registra en MLflow.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Annotated
import unicodedata

from loguru import logger
import pandas as pd
import requests
import typer

from tc5061_proyecto_equipo19.config import (
    ARCHIVO_ESTACIONES,
    DATA_SCHEMA_VERSION,
    MANIFEST_PATH,
    MESES,
    MIBICI_DATOS_ABIERTOS_URL,
    PARQUET_COMPRESSION,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    URL_ESTACIONES,
    URLS_VIAJES,
)

app = typer.Typer(help="Ingesta de datos de MiBici.", invoke_without_command=True)

# Nombres canónicos. La llave es el nombre original ya normalizado por `clave_columna`.
TRIP_COLUMNS = {
    "viaje_id": "trip_id",
    "usuario_id": "user_id",
    "genero": "gender",
    "ano_de_nacimiento": "birth_year",
    "inicio_del_viaje": "start_time",
    "fin_del_viaje": "end_time",
    "origen_id": "origin_station_id",
    "destino_id": "destination_station_id",
}

STATION_COLUMNS = {
    "id": "station_id",
    "name": "station_name",
    "latitude": "latitude",
    "longitude": "longitude",
    "dpcapacity": "capacity",
}

# Columnas que se conservan. Se excluyen user_id, gender y birth_year
# por ser datos personales que ninguna de las tres preguntas necesita.
ANALYTICAL_TRIP_COLUMNS = [
    "trip_id",
    "start_time",
    "end_time",
    "origin_station_id",
    "destination_station_id",
]

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
FORMATOS_FECHA = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%Y/%m/%d %H:%M:%S",
]
HTTP_TIMEOUT = 120
USER_AGENT = "TC5061-Equipo19-ingesta/1.0"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo leyéndolo por bloques de 1 MB."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clave_columna(nombre: str) -> str:
    """Normaliza un nombre de columna para compararlo sin importar su codificación.

    Repara texto UTF-8 leído como Latin-1 (``AÃ±o`` -> ``Año``), quita acentos,
    pasa a minúsculas y reemplaza espacios por guiones bajos. Así ``Año_de_nacimiento``,
    ``AÃ±o_de_nacimiento`` y ``Ano de nacimiento`` terminan en la misma llave.
    """
    texto = str(nombre).strip()
    try:
        texto = texto.encode("latin-1").decode("utf-8")
    except UnicodeEncodeError, UnicodeDecodeError:
        pass
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[\s\-]+", "_", texto.lower())
    return texto


# ---------------------------------------------------------------------------
# 1. Descarga
# ---------------------------------------------------------------------------
def buscar_enlaces_publicados() -> dict[str, str]:
    """Lee la página de datos abiertos y devuelve {nombre_de_archivo: url}.

    Permite seguir funcionando si MiBici cambia la carpeta de un archivo. Si la
    página no responde se devuelve un diccionario vacío y se usan las URLs de config.
    """
    try:
        respuesta = requests.get(
            MIBICI_DATOS_ABIERTOS_URL, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}
        )
        respuesta.raise_for_status()
    except requests.RequestException as error:
        logger.warning(f"No se pudo leer la página de datos abiertos ({error}); uso config.")
        return {}

    enlaces = {}
    for href in re.findall(r'href="([^"]+\.csv)"', respuesta.text):
        url = requests.compat.urljoin(MIBICI_DATOS_ABIERTOS_URL, href)
        enlaces[url.rsplit("/", 1)[-1]] = url
    return enlaces


def descargar_archivo(url: str, destino: Path, forzar: bool = False) -> Path:
    """Descarga `url` a `destino` en streaming; omite la descarga si ya existe."""
    if destino.exists() and not forzar:
        logger.info(f"Ya existe {destino.name}, se omite la descarga.")
        return destino

    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(destino.suffix + ".part")
    logger.info(f"Descargando {url}")
    with requests.get(
        url, stream=True, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}
    ) as respuesta:
        respuesta.raise_for_status()
        with temporal.open("wb") as archivo:
            for bloque in respuesta.iter_content(chunk_size=1024 * 1024):
                archivo.write(bloque)
    temporal.replace(destino)
    return destino


@app.command("descargar")
def descargar(
    meses: Annotated[list[str], typer.Option(help="Meses AAAA_MM a descargar.")] = MESES,
    forzar: bool = typer.Option(False, help="Volver a descargar aunque el archivo exista."),
) -> None:
    """Descarga los meses elegidos y la nomenclatura de estaciones a data/raw."""
    publicados = buscar_enlaces_publicados()

    pendientes = {f"datos_abiertos_{mes}.csv": URLS_VIAJES.get(mes) for mes in meses}
    pendientes[ARCHIVO_ESTACIONES] = URL_ESTACIONES

    for nombre, url_config in pendientes.items():
        url = publicados.get(nombre, url_config)
        if url is None:
            raise typer.BadParameter(f"No hay URL conocida para {nombre}.")
        descargar_archivo(url, RAW_DATA_DIR / nombre, forzar=forzar)

    logger.success(f"Descarga completa en {RAW_DATA_DIR}")


# ---------------------------------------------------------------------------
# 2. Normalización
# ---------------------------------------------------------------------------
def read_csv_checked(path: Path) -> tuple[pd.DataFrame, str]:
    """Lee un CSV probando codificaciones conocidas y devuelve también la que funcionó."""
    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=encoding, encoding_errors="strict", low_memory=False)
            logger.info(f"{path.name} leído con {encoding}")
            return df, encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer {path.name} con las codificaciones {ENCODINGS}.")


def normalize_columns(
    df: pd.DataFrame, mapping: dict[str, str], required: list[str]
) -> pd.DataFrame:
    """Renombra columnas a su nombre canónico y valida que existan las obligatorias."""
    result = df.copy()
    normalized = [mapping.get(clave_columna(c), clave_columna(c)) for c in result.columns]

    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Columnas duplicadas tras normalizar: {normalized}")
    result.columns = normalized

    missing = sorted(set(required) - set(result.columns))
    if missing:
        raise ValueError(f"Faltan columnas {missing}. Encontradas: {result.columns.tolist()}")
    return result


def parsear_fechas(serie: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Convierte texto a fecha probando formatos explícitos y cuenta cuántas filas usó cada uno.

    ``format="mixed"`` interpreta ``12/10/2024`` como 10 de diciembre (mes primero),
    lo que en 2024-09 y 2024-10 produjo viajes con fin anterior al inicio. Aquí cada
    formato se prueba en orden sobre las filas que aún no se pudieron leer, siempre
    con día antes que mes en las fechas con diagonal (convención en México). El
    conteo por formato queda en el manifiesto para revisar cada archivo.
    """
    texto = serie.astype("string").str.strip()
    resultado = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    conteo = {}
    for formato in FORMATOS_FECHA:
        pendientes = resultado.isna() & texto.notna()
        if not pendientes.any():
            break
        convertidas = pd.to_datetime(texto[pendientes], format=formato, errors="coerce")
        resultado.loc[pendientes] = convertidas
        if (n := int(convertidas.notna().sum())) > 0:
            conteo[formato] = n
    return resultado, conteo


def process_trips(path: Path) -> dict:
    """Normaliza un archivo mensual de viajes, lo guarda en Parquet y devuelve sus metadatos."""
    df, encoding = read_csv_checked(path)
    df = normalize_columns(df, TRIP_COLUMNS, ANALYTICAL_TRIP_COLUMNS)
    df = df[ANALYTICAL_TRIP_COLUMNS].copy()

    # Identificadores como texto para evitar problemas de tipos y ceros a la izquierda.
    for column in ["trip_id", "origin_station_id", "destination_station_id"]:
        df[column] = df[column].astype("string").str.strip()

    formatos = {}
    for column in ["start_time", "end_time"]:
        df[column], formatos[column] = parsear_fechas(df[column])

    output_path = PROCESSED_DATA_DIR / (
        path.stem.replace("datos_abiertos_", "trips_") + ".parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression=PARQUET_COMPRESSION)

    con_fecha = df["start_time"].notna().any()
    metadata = {
        "source_file": path.name,
        "source_encoding": encoding,
        "source_sha256": sha256_file(path),
        "output_file": output_path.name,
        "output_sha256": sha256_file(output_path),
        "records": len(df),
        # Los casos problemáticos se cuantifican aquí y se limpian en features.py.
        "invalid_date_rows": int((df["start_time"].isna() | df["end_time"].isna()).sum()),
        "rows_with_duplicate_trip_ids": int(df["trip_id"].duplicated(keep=False).sum()),
        "date_formats": formatos["start_time"],
        "start_time_min": str(df["start_time"].min()) if con_fecha else None,
        "start_time_max": str(df["start_time"].max()) if con_fecha else None,
    }
    logger.info(f"Procesado {path.name}: {len(df):,} registros")
    return metadata


def process_stations(path: Path) -> dict:
    """Normaliza el catálogo de estaciones, lo guarda en Parquet y devuelve sus metadatos."""
    df, encoding = read_csv_checked(path)
    df = normalize_columns(df, STATION_COLUMNS, list(STATION_COLUMNS.values()))
    df = df[list(STATION_COLUMNS.values())].copy()

    df["station_id"] = df["station_id"].astype("string").str.strip()
    df["station_name"] = df["station_name"].astype("string").str.strip()
    for column in ["latitude", "longitude", "capacity"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    output_path = PROCESSED_DATA_DIR / "stations.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression=PARQUET_COMPRESSION)

    logger.info(f"Procesadas {len(df)} estaciones")
    return {
        "source_file": path.name,
        "source_encoding": encoding,
        "source_sha256": sha256_file(path),
        "output_file": output_path.name,
        "output_sha256": sha256_file(output_path),
        "records": len(df),
        "duplicate_station_ids": int(df["station_id"].duplicated(keep=False).sum()),
        "missing_coordinates": int((df["latitude"].isna() | df["longitude"].isna()).sum()),
    }


# ---------------------------------------------------------------------------
# 3. Versionado
# ---------------------------------------------------------------------------
def calcular_version_datos(trips: list[dict], stations: dict | None) -> str:
    """Devuelve un identificador corto que cambia si cambia cualquier archivo procesado.

    Es el SHA-256 (12 caracteres) de los hashes de los Parquet de viajes ordenados,
    seguidos del hash del Parquet de estaciones. No usa la fecha de generación del
    manifiesto, así que dos ingestas con los mismos archivos dan la misma versión.
    Es el valor que se registra en MLflow como ``data_version`` (P2 lo calcula igual).
    """
    hashes = sorted(t["output_sha256"] for t in trips)
    if stations:
        hashes.append(stations["output_sha256"])
    return hashlib.sha256("".join(hashes).encode()).hexdigest()[:12]


def generate_manifest(trips: list[dict], stations: dict | None) -> dict:
    """Escribe el manifiesto de datos de forma atómica y lo devuelve."""
    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "data_version": calcular_version_datos(trips, stations),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "months": len(trips),
        "total_trips": int(sum(t["records"] for t in trips)),
        "start_time_min": min(
            (t["start_time_min"] for t in trips if t["start_time_min"]), default=None
        ),
        "start_time_max": max(
            (t["start_time_max"] for t in trips if t["start_time_max"]), default=None
        ),
        "trips": trips,
        "stations": stations,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporal = MANIFEST_PATH.with_suffix(".tmp")
    temporal.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporal.replace(MANIFEST_PATH)
    return manifest


def leer_manifiesto() -> dict:
    """Lee el manifiesto versionado en Git. Lo usan los runs de MLflow."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"No existe {MANIFEST_PATH}. Ejecuta primero la ingesta.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if "data_version" not in manifest:  # manifiestos anteriores a la versión 1.1.0
        manifest["data_version"] = calcular_version_datos(
            manifest["trips"], manifest.get("stations")
        )
    return manifest


@app.command("procesar")
def procesar(
    meses: Annotated[list[str], typer.Option(help="Meses AAAA_MM a procesar.")] = MESES,
) -> None:
    """Convierte los CSV de data/raw a Parquet y genera el manifiesto con hashes."""
    trip_files = [RAW_DATA_DIR / f"datos_abiertos_{mes}.csv" for mes in meses]
    faltantes = [p.name for p in trip_files if not p.exists()]
    if faltantes:
        raise FileNotFoundError(f"Faltan en {RAW_DATA_DIR}: {faltantes}. Ejecuta `descargar`.")

    station_path = RAW_DATA_DIR / ARCHIVO_ESTACIONES
    trips = [process_trips(path) for path in trip_files]
    stations = process_stations(station_path) if station_path.exists() else None

    manifest = generate_manifest(trips, stations)
    logger.success(
        f"Ingesta completa: {manifest['total_trips']:,} viajes, versión {manifest['data_version']}"
    )


@app.callback()
def main(ctx: typer.Context) -> None:
    """Sin subcomando ejecuta la ingesta completa: descargar y procesar."""
    if ctx.invoked_subcommand is None:
        descargar(meses=MESES, forzar=False)
        procesar(meses=MESES)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_ingestion()