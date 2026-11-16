"""Descarga y procesa los meses recientes necesarios para el A/B offline.

Uso (desde la raíz del repositorio):

    python -m tc5061_proyecto_equipo19.monitoring.descargar_2026
    python -m tc5061_proyecto_equipo19.monitoring.descargar_2026 --inicio 2026-07 --fin 2026-09

Por defecto cubre CURRENT_MONTHS. Descarga los CSV a data/raw y genera los Parquet
en data/processed. Al no ser el rango oficial, el manifiesto se escribe en
PARTIAL_MANIFEST_PATH y no se sobrescribe references/data_manifest.json.
"""
from __future__ import annotations

import argparse

from loguru import logger

from tc5061_proyecto_equipo19.dataset import run_ingestion
from tc5061_proyecto_equipo19.monitoring.data import CURRENT_MONTHS


def _a_aaaa_mm(month: str) -> str:
    """Convierte '2026_07' en '2026-07'."""
    return month.replace("_", "-")


def ejecutar_descarga(inicio: str | None = None, fin: str | None = None):
    """Descarga y procesa el rango [inicio, fin] (AAAA-MM); por defecto CURRENT_MONTHS."""
    inicio = inicio or _a_aaaa_mm(min(CURRENT_MONTHS))
    fin = fin or _a_aaaa_mm(max(CURRENT_MONTHS))
    logger.info(f"Descarga aditiva para monitoreo: {inicio} .. {fin}")
    manifiesto = run_ingestion(start=inicio, end=fin)
    logger.success(f"Descarga completada. Manifiesto parcial: {manifiesto}")
    return manifiesto


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inicio", metavar="AAAA-MM", default=None, help="Primer mes")
    parser.add_argument("--fin", metavar="AAAA-MM", default=None, help="Último mes")
    args = parser.parse_args(argv)
    ejecutar_descarga(args.inicio, args.fin)


if __name__ == "__main__":
    main()
