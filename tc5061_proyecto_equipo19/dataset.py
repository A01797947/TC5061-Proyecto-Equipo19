"""Ingesta reproducible y versionada de los datos de MiBici.

Uso desde la terminal (en la raíz del repositorio):

    python -m tc5061_proyecto_equipo19.dataset                  # descarga + procesa + manifiesto
    python -m tc5061_proyecto_equipo19.dataset descargar        # solo descarga a data/raw
    python -m tc5061_proyecto_equipo19.dataset procesar         # solo CSV -> Parquet + manifiesto
    python -m tc5061_proyecto_equipo19.dataset --inicio 2025-12 --fin 2025-12   # un solo mes

Etapas:
1. Descarga los meses del rango (por defecto el de ``config``) y la nomenclatura de
   estaciones en ``data/raw``. Los archivos ya descargados no se vuelven a bajar.
2. Lee cada CSV detectando su codificación, normaliza nombres de columnas
   (acentos, mayúsculas, texto mal codificado) y convierte fechas a ``datetime``.
3. Excluye los datos personales (usuario, género y año de nacimiento) y guarda
   los datos limpios en Parquet en ``data/processed``.
4. Escribe ``references/data_manifest.json`` con el número de registros y el
   hash SHA-256 de cada archivo. La versión de datos (``data_version``) es un hash
   corto del contenido de todos los Parquet y se registra en MLflow. Si el rango
   no es el oficial, escribe ``PARTIAL_MANIFEST_PATH`` para no pisar el oficial.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Annotated
import unicodedata

from loguru import logger
import pandas as pd
import requests
import typer

from tc5061_proyecto_equipo19.config import (
    ARCHIVO_ESTACIONES,
    DATA_SCHEMA_VERSION,
    FECHA_FIN,
    FECHA_INICIO,
    MANIFEST_PATH,
    MIBICI_BASE_URL,
    MIBICI_DATOS_ABIERTOS_URL,
    PARQUET_COMPRESSION,
    PARTIAL_MANIFEST_PATH,
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

# Descarga
HTTP_TIMEOUT = 120
USER_AGENT = "TC5061-Equipo19-ingesta/1.0"
DOWNLOAD_RETRIES = 3

# Rango por defecto (AAAA-MM), derivado de las fechas de config.
START_MONTH = FECHA_INICIO[:7]
END_MONTH = FECHA_FIN[:7]

# Enlace de descarga en la página de MiBici: /site/assets/files/<id>/<archivo>.csv
_LINK_PATTERN = re.compile(
    r"(/site/assets/files/\d+/((?:datos_abiertos|nomenclatura)_\d{4}_\d{2})[^\"'\s<>]*\.csv)"
)
# Formato válido de mes para el rango de ingesta (AAAA-MM).
_MONTH_PATTERN = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


# ---------------------------------------------------------------------------
# Hashes y manifiesto
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo leyéndolo por bloques de 1 MB.

    Args:
        path: Ruta del archivo.

    Returns:
        Hash SHA-256 en hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    """Lee un manifiesto; devuelve {} si aún no existe.

    Args:
        path: Ruta del manifiesto JSON.

    Returns:
        Contenido del manifiesto o un diccionario vacío si el archivo no existe.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes(manifest: dict) -> dict[str, str]:
    """Devuelve {archivo: sha256} de todos los CSV que registra el manifiesto.

    Args:
        manifest: Manifiesto leído con `load_manifest`.

    Returns:
        Diccionario {nombre_del_csv: sha256} con viajes y catálogo de estaciones.
    """
    entries = list(manifest.get("trips", []))
    if manifest.get("stations"):
        entries.append(manifest["stations"])
    return {entry["source_file"]: entry["source_sha256"] for entry in entries}


def calcular_version_datos(trips: list[dict], stations: dict | None) -> str:
    """Devuelve un identificador corto que cambia si cambia cualquier archivo procesado.

    Es el SHA-256 (12 caracteres) de los hashes de los Parquet de viajes ordenados,
    seguidos del hash del Parquet de estaciones. No usa la fecha de generación del
    manifiesto, así que dos ingestas con los mismos archivos dan la misma versión.
    Es el valor que se registra en MLflow como ``data_version`` (P2 lo calcula igual).

    Args:
        trips: Metadatos de cada mes procesado.
        stations: Metadatos del catálogo de estaciones, o None.

    Returns:
        Versión de datos de 12 caracteres hexadecimales.
    """
    hashes = sorted(t["output_sha256"] for t in trips)
    if stations:
        hashes.append(stations["output_sha256"])
    return hashlib.sha256("".join(hashes).encode()).hexdigest()[:12]


def generate_manifest(
    trips: list[dict],
    stations: dict | None,
    path: Path = MANIFEST_PATH,
    date_range: tuple[str, str] | None = None,
) -> dict:
    """Escribe el manifiesto de datos de forma atómica y lo devuelve.

    Args:
        trips: Metadatos de cada mes procesado.
        stations: Metadatos del catálogo de estaciones, o None.
        path: Destino del manifiesto (oficial o parcial).
        date_range: Meses (inicio, fin) en formato AAAA-MM que cubre esta ingesta.

    Returns:
        El manifiesto escrito.
    """
    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "data_version": calcular_version_datos(trips, stations),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "date_range": (
            {"start_month": date_range[0], "end_month": date_range[1]} if date_range else None
        ),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    # Escritura atómica para no dejar un JSON incompleto.
    temporal = path.with_suffix(".tmp")
    temporal.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporal.replace(path)
    return manifest


def leer_manifiesto() -> dict:
    """Lee el manifiesto oficial versionado en Git. Lo usan los runs de MLflow.

    Returns:
        Contenido del manifiesto, con ``data_version`` siempre presente.

    Raises:
        FileNotFoundError: Si el manifiesto no existe.
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"No existe {MANIFEST_PATH}. Ejecuta primero la ingesta.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if "data_version" not in manifest:  # manifiestos anteriores a la versión 1.1.0
        manifest["data_version"] = calcular_version_datos(
            manifest["trips"], manifest.get("stations")
        )
    return manifest


# ---------------------------------------------------------------------------
# Lectura y normalización
# ---------------------------------------------------------------------------
def clave_columna(nombre: str) -> str:
    """Normaliza un nombre de columna para compararlo sin importar su codificación.

    Repara texto UTF-8 leído como Latin-1 (``AÃ±o`` -> ``Año``), quita acentos,
    pasa a minúsculas y reemplaza espacios por guiones bajos. Así ``Año_de_nacimiento``,
    ``AÃ±o_de_nacimiento`` y ``Ano de nacimiento`` terminan en la misma llave.

    Args:
        nombre: Nombre de columna tal como viene en el CSV.

    Returns:
        Llave normalizada.
    """
    texto = str(nombre).strip()
    try:
        texto = texto.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[\s\-]+", "_", texto.lower())


def read_csv_checked(path: Path) -> tuple[pd.DataFrame, str]:
    """Lee un CSV probando codificaciones conocidas.

    Args:
        path: CSV a leer.

    Returns:
        Tabla sin transformar y la codificación que funcionó.

    Raises:
        ValueError: Si ninguna codificación puede leer el archivo.
    """
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
    """Renombra columnas a su nombre canónico y valida que existan las obligatorias.

    Args:
        df: Tabla original.
        mapping: {llave_normalizada: nombre_canónico}.
        required: Nombres canónicos que deben existir tras renombrar.

    Returns:
        Copia de `df` con las columnas renombradas.

    Raises:
        ValueError: Si hay columnas duplicadas o falta alguna obligatoria.
    """
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

    Args:
        serie: Columna con fechas en texto.

    Returns:
        Serie de fechas y diccionario {formato: filas convertidas}.
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
    """Normaliza un archivo mensual de viajes, lo guarda en Parquet y devuelve sus metadatos.

    Args:
        path: CSV mensual en data/raw.

    Returns:
        Metadatos (hashes, conteos, rango de fechas) para el manifiesto.

    Raises:
        ValueError: Si el CSV no se puede leer o faltan columnas obligatorias.
    """
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
    """Normaliza el catálogo de estaciones, lo guarda en Parquet y devuelve sus metadatos.

    Args:
        path: CSV del catálogo en data/raw.

    Returns:
        Metadatos (hashes y conteos) para el manifiesto.

    Raises:
        ValueError: Si el CSV no se puede leer o faltan columnas obligatorias.
    """
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
# Descarga
# ---------------------------------------------------------------------------
def iter_months(start: str, end: str) -> list[str]:
    """Devuelve los meses entre `start` y `end` (incluidos) como ``AAAA_MM``.

    Args:
        start: Primer mes, formato AAAA-MM.
        end: Último mes, formato AAAA-MM.

    Returns:
        Lista de meses, por ejemplo ``["2025_11", "2025_12"]``.

    Raises:
        ValueError: Si un mes no tiene formato AAAA-MM o el inicio es posterior al fin.
    """
    parsed = []
    for label, value in (("inicio", start), ("fin", end)):
        match = _MONTH_PATTERN.match(value)
        if not match:
            raise ValueError(f"Mes de {label} inválido: {value} (formato AAAA-MM)")
        parsed.append((int(match.group(1)), int(match.group(2))))

    (year, month), (end_year, end_month) = parsed
    if (year, month) > (end_year, end_month):
        raise ValueError(f"El mes de inicio {start} es posterior al de fin {end}")

    months = []
    while (year, month) <= (end_year, end_month):
        months.append(f"{year:04d}_{month:02d}")
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return months


def parse_download_links(html: str) -> dict[str, str]:
    """Extrae del HTML de la página de MiBici {nombre_de_archivo: url}.

    Args:
        html: Contenido de la página de datos abiertos.

    Returns:
        Diccionario {nombre_del_archivo: url_completa}.
    """
    links = {}
    for path, name in _LINK_PATTERN.findall(html):
        links[f"{name}.csv"] = f"{MIBICI_BASE_URL}{path}"
    return links


def resolve_download_urls(session: requests.Session) -> dict[str, str]:
    """Devuelve {nombre_de_archivo: url}, priorizando lo publicado en la página.

    Si MiBici cambia la carpeta de un archivo, la página trae la URL vigente. Si la
    página no responde, se usan las URLs guardadas en config.

    Args:
        session: Sesión de requests con el User-Agent.

    Returns:
        Diccionario {nombre_del_archivo: url}.
    """
    urls = {f"datos_abiertos_{mes}.csv": url for mes, url in URLS_VIAJES.items()}
    urls[ARCHIVO_ESTACIONES] = URL_ESTACIONES
    try:
        response = session.get(MIBICI_DATOS_ABIERTOS_URL, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as error:
        logger.warning(f"No se pudo leer la página de datos abiertos ({error}); uso config.")
        return urls

    published = parse_download_links(response.text)
    logger.info(f"MiBici publica {len(published)} archivos")
    urls.update(published)
    return urls


def download_file(url: str, destination: Path, session: requests.Session) -> None:
    """Descarga `url` a `destination` con reintentos y escritura atómica (.part).

    Args:
        url: Dirección del archivo.
        destination: Ruta donde se guarda.
        session: Sesión de requests con el User-Agent.

    Raises:
        requests.RequestException: Si se agotan los reintentos.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=HTTP_TIMEOUT) as response:
                response.raise_for_status()
                with partial.open("wb") as file:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        file.write(block)
            partial.replace(destination)
            return
        except requests.RequestException as error:
            logger.warning(f"Intento {attempt}/{DOWNLOAD_RETRIES} fallido ({destination.name}): {error}")
            if attempt == DOWNLOAD_RETRIES:
                raise
            time.sleep(2**attempt)


def ensure_raw_files(file_names: list[str], expected_hashes: dict[str, str]) -> None:
    """Descarga a data/raw los CSV que falten; los que ya existen no se tocan.

    Args:
        file_names: Nombres de los CSV necesarios.
        expected_hashes: {archivo: sha256} del manifiesto, para avisar si MiBici
            modificó un archivo desde la última ingesta.

    Raises:
        LookupError: Si no hay URL conocida para algún archivo faltante.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    missing = [name for name in file_names if not (RAW_DATA_DIR / name).exists()]
    if not missing:
        logger.info("Todos los archivos ya existen en data/raw")
        return

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    urls = resolve_download_urls(session)

    changed = []
    for name in missing:
        if name not in urls:
            raise LookupError(f"No hay URL conocida para {name}.")
        path = RAW_DATA_DIR / name
        logger.info(f"Descargando {name}")
        download_file(urls[name], path, session)

        expected = expected_hashes.get(name)
        if expected is not None and sha256_file(path) != expected:
            changed.append(name)

    if changed:
        logger.warning(f"Hash distinto al del manifiesto (MiBici pudo modificarlos): {changed}")


# ---------------------------------------------------------------------------
# Flujo completo
# ---------------------------------------------------------------------------
def download_range(start: str, end: str) -> None:
    """Descarga los CSV del rango y el catálogo de estaciones.

    Args:
        start: Primer mes, formato AAAA-MM.
        end: Último mes, formato AAAA-MM.
    """
    names = [f"datos_abiertos_{month}.csv" for month in iter_months(start, end)]
    ensure_raw_files(names + [ARCHIVO_ESTACIONES], source_hashes(load_manifest(MANIFEST_PATH)))
    logger.success(f"Descarga completa en {RAW_DATA_DIR}")


def process_range(start: str, end: str) -> Path:
    """Convierte a Parquet los CSV del rango y escribe el manifiesto.

    Si el rango es el oficial se escribe ``MANIFEST_PATH``; si no, el manifiesto
    parcial, para no sobrescribir el que está versionado en Git.

    Args:
        start: Primer mes, formato AAAA-MM.
        end: Último mes, formato AAAA-MM.

    Returns:
        Ruta del manifiesto escrito.

    Raises:
        FileNotFoundError: Si faltan CSV en data/raw.
    """
    trip_files = [RAW_DATA_DIR / f"datos_abiertos_{m}.csv" for m in iter_months(start, end)]
    station_path = RAW_DATA_DIR / ARCHIVO_ESTACIONES

    faltantes = [p.name for p in [*trip_files, station_path] if not p.exists()]
    if faltantes:
        raise FileNotFoundError(f"Faltan en {RAW_DATA_DIR}: {faltantes}. Ejecuta `descargar`.")

    trips = [process_trips(path) for path in trip_files]
    stations = process_stations(station_path)

    is_default_range = (start, end) == (START_MONTH, END_MONTH)
    manifest_path = MANIFEST_PATH if is_default_range else PARTIAL_MANIFEST_PATH
    manifest = generate_manifest(trips, stations, manifest_path, (start, end))

    logger.success(
        f"Ingesta completa: {manifest['total_trips']:,} viajes, "
        f"versión {manifest['data_version']}. Manifiesto: {manifest_path}"
    )
    return manifest_path


def run_ingestion(start: str | None = None, end: str | None = None) -> Path:
    """Descarga y procesa los datos del rango de meses.

    Args:
        start: Primer mes, formato AAAA-MM (por defecto el de config).
        end: Último mes, formato AAAA-MM (por defecto el de config).

    Returns:
        Ruta del manifiesto escrito.
    """
    start = start or START_MONTH
    end = end or END_MONTH
    download_range(start, end)
    return process_range(start, end)


# ---------------------------------------------------------------------------
# Línea de comandos
# ---------------------------------------------------------------------------
Inicio = Annotated[str, typer.Option("--inicio", metavar="AAAA-MM", help="Primer mes.")]
Fin = Annotated[str, typer.Option("--fin", metavar="AAAA-MM", help="Último mes.")]


def _validar_rango(inicio: str, fin: str) -> None:
    """Convierte un rango inválido en un error de uso de la CLI."""
    try:
        iter_months(inicio, fin)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


@app.command("descargar")
def descargar(inicio: Inicio = START_MONTH, fin: Fin = END_MONTH) -> None:
    """Descarga los CSV del rango (y las estaciones) a data/raw."""
    _validar_rango(inicio, fin)
    download_range(inicio, fin)


@app.command("procesar")
def procesar(inicio: Inicio = START_MONTH, fin: Fin = END_MONTH) -> None:
    """Convierte los CSV de data/raw a Parquet y genera el manifiesto con hashes."""
    _validar_rango(inicio, fin)
    process_range(inicio, fin)


@app.callback()
def main(ctx: typer.Context, inicio: Inicio = START_MONTH, fin: Fin = END_MONTH) -> None:
    """Sin subcomando ejecuta la ingesta completa: descargar y procesar."""
    if ctx.invoked_subcommand is None:
        _validar_rango(inicio, fin)
        run_ingestion(inicio, fin)


if __name__ == "__main__":
    app()