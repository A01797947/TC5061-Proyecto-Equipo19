"""Configuración central del proyecto: rutas, fechas, semillas, fuentes y MLflow.

Cualquier valor que afecte la reproducibilidad (rango de fechas, particiones,
semillas, URLs de los datos o nombre del experimento) se define aquí y el resto
de los módulos lo importa. Así un cambio de configuración queda en un solo commit.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Variables de entorno opcionales (por ejemplo MLFLOW_TRACKING_URI) desde .env
load_dotenv()

# Paths

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")


DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REFERENCES_DIR = PROJ_ROOT / "references"
MANIFEST_PATH = REFERENCES_DIR / "data_manifest.json"
PARTIAL_MANIFEST_PATH = INTERIM_DATA_DIR / "data_manifest_partial.json"

# ---------------------------------------------------------------------------
# Rango de fechas y partición temporal
# ---------------------------------------------------------------------------
# 24 meses completos: dos ciclos anuales para capturar estacionalidad.
FECHA_INICIO = "2024-01-01"
FECHA_FIN = "2025-12-31"
MESES = [f"{anio}_{mes:02d}" for anio in (2024, 2025) for mes in range(1, 13)]

# Partición estrictamente temporal (sin mezclar el futuro en entrenamiento).
# Entrenamiento: 2024-01 a 2025-06 | Validación: 2025-07 a 2025-09 | Prueba: 2025-10 a 2025-12
FIN_ENTRENAMIENTO = "2025-06-30 23:59:59"
INICIO_VALIDACION = "2025-07-01"
FIN_VALIDACION = "2025-09-30 23:59:59"
INICIO_PRUEBA = "2025-10-01"


# Días feriados oficiales (Ley Federal del Trabajo, art. 74) en el periodo de estudio.
FERIADOS = [
    "2024-01-01",
    "2024-02-05",
    "2024-03-18",
    "2024-05-01",
    "2024-09-16",
    "2024-10-01",
    "2024-11-18",
    "2024-12-25",
    "2025-01-01",
    "2025-02-03",
    "2025-03-17",
    "2025-05-01",
    "2025-09-16",
    "2025-11-17",
    "2025-12-25",
]
# ---------------------------------------------------------------------------
# Semillas y parámetros generales
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# Franja temporal de agregación del panel estación × hora.
FRECUENCIA_PANEL = "h"

# Límite de filas para entrenar los modelos candidatos (muestreo con semilla).
MAX_FILAS_ENTRENAMIENTO = 2_000_000

# ---------------------------------------------------------------------------
# Fuentes de datos de MiBici
# ---------------------------------------------------------------------------
MIBICI_DATOS_ABIERTOS_URL = "https://www.mibici.net/es/datos-abiertos/"
MIBICI_BASE_URL = "https://mibici.net"
MIBICI_ASSETS_URL = "https://www.mibici.net/site/assets/files"

# Enlaces publicados en la página de datos abiertos. Si MiBici los cambia,
# dataset.py los vuelve a buscar en la página antes de usar estos.
URLS_VIAJES = {
    "2024_01": f"{MIBICI_ASSETS_URL}/61450/datos_abiertos_2024_01.csv",
    "2024_02": f"{MIBICI_ASSETS_URL}/77251/datos_abiertos_2024_02.csv",
    "2024_03": f"{MIBICI_ASSETS_URL}/77252/datos_abiertos_2024_03.csv",
    "2024_04": f"{MIBICI_ASSETS_URL}/80028/datos_abiertos_2024_04.csv",
    "2024_05": f"{MIBICI_ASSETS_URL}/81663/datos_abiertos_2024_05.csv",
    "2024_06": f"{MIBICI_ASSETS_URL}/83901/datos_abiertos_2024_06.csv",
    "2024_07": f"{MIBICI_ASSETS_URL}/85500/datos_abiertos_2024_07.csv",
    "2024_08": f"{MIBICI_ASSETS_URL}/87665/datos_abiertos_2024_08.csv",
    "2024_09": f"{MIBICI_ASSETS_URL}/89736/datos_abiertos_2024_09.csv",
    "2024_10": f"{MIBICI_ASSETS_URL}/91758/datos_abiertos_2024_10.csv",
    "2024_11": f"{MIBICI_ASSETS_URL}/93516/datos_abiertos_2024_11.csv",
    "2024_12": f"{MIBICI_ASSETS_URL}/94962/datos_abiertos_2024_12.csv",
    "2025_01": f"{MIBICI_ASSETS_URL}/97653/datos_abiertos_2025_01.csv",
    "2025_02": f"{MIBICI_ASSETS_URL}/99879/datos_abiertos_2025_02.csv",
    "2025_03": f"{MIBICI_ASSETS_URL}/101986/datos_abiertos_2025_03.csv",
    "2025_04": f"{MIBICI_ASSETS_URL}/103813/datos_abiertos_2025_04.csv",
    "2025_05": f"{MIBICI_ASSETS_URL}/105513/datos_abiertos_2025_05.csv",
    "2025_06": f"{MIBICI_ASSETS_URL}/107088/datos_abiertos_2025_06.csv",
    "2025_07": f"{MIBICI_ASSETS_URL}/108898/datos_abiertos_2025_07.csv",
    "2025_08": f"{MIBICI_ASSETS_URL}/111021/datos_abiertos_2025_08.csv",
    "2025_09": f"{MIBICI_ASSETS_URL}/113216/datos_abiertos_2025_09.csv",
    "2025_10": f"{MIBICI_ASSETS_URL}/115104/datos_abiertos_2025_10.csv",
    "2025_11": f"{MIBICI_ASSETS_URL}/116773/datos_abiertos_2025_11.csv",
    "2025_12": f"{MIBICI_ASSETS_URL}/118153/datos_abiertos_2025_12.csv",
}
ARCHIVO_ESTACIONES = "nomenclatura_2026_09.csv"
URL_ESTACIONES = f"{MIBICI_ASSETS_URL}/1118/{ARCHIVO_ESTACIONES}"

DATA_SCHEMA_VERSION = "1.1.0"
PARQUET_COMPRESSION = "snappy"

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
EQUIPO = "equipo19"
MLFLOW_EXPERIMENT_NAME = "mibici-equipo19"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACT_LOCATION = (PROJ_ROOT / "mlartifacts").as_uri()

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
