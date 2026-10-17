from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJECT_ROOT path is: {PROJECT_ROOT}")


DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
PARTIAL_MANIFEST_PATH = INTERIM_DIR/"data_manifest_partial.json"


REFERENCES_DIR = PROJECT_ROOT / "references"
MANIFEST_PATH = REFERENCES_DIR / "data_manifest.json"

DATA_SCHEMA_VERSION = "1.0.0"
CSV_ENCODING = "utf-8-sig"
PARQUET_COMPRESSION = "snappy"


#===========Datos de MiBici==============
START_MONTH = "2024-01"
END_MONTH = "2025-12"
STATIONS_FILE = "nomenclatura_2026_09.csv"

MIBICI_BASE_URL = "https://mibici.net"
MIBICI_DATA_PAGE = f"{MIBICI_BASE_URL}/es/datos-abiertos/"
DOWNLOAD_USER_AGENT = "TC5061-Proyecto-Equipo19"
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_RETRIES = 3

DOWNLOAD_URL_OVERRIDES: dict[str, str] = {}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
