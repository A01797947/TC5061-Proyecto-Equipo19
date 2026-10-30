"""Exporta train/test P1 de parquet local para ejecutarlos con DeepChecks en 3.10.

Ejecutar en el entorno principal Python 3.14; no escribe en el repositorio Git.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.features.p1 import construir_demanda_diaria, P1_FEATURE_COLUMNS


def export_sample(source: Path, output_dir: Path, max_rows: int = 3000) -> dict:
    if max_rows < 1:
        raise ValueError("max_rows debe ser positivo")
    if not source.is_file():
        raise FileNotFoundError(f"No existe {source}; genera primero las features")
    viajes = pd.read_parquet(source, columns=P1_FEATURE_COLUMNS)
    _, _, datos = construir_demanda_diaria(viajes)
    fechas = pd.to_datetime({"year": datos["Año"], "month": datos["Mes"], "day": datos["Dia"]})
    # Usamos fronteras [inicio, fin) para abarcar todas las fracciones de segundo.
    parts = {
        "train": datos.loc[fechas < pd.Timestamp(cfg.INICIO_VALIDACION)],
        "test": datos.loc[fechas >= pd.Timestamp(cfg.INICIO_PRUEBA)],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, frame in parts.items():
        if frame.empty:
            raise ValueError(f"No existen registros en la partición {name}")
        # Muestreo determinista de las filas agregadas P1 (no de usuarios ni viajes).
        sample = frame.sample(n=min(len(frame), max_rows), random_state=cfg.RANDOM_SEED)
        path = output_dir / f"quality_{name}_p1.csv"
        sample.to_csv(path, index=False)
        result[name] = {"path": str(path), "rows": len(sample)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=cfg.INTERIM_DATA_DIR / "viajes_limpios_enriquecidos.parquet")
    parser.add_argument("--output-dir", type=Path, default=cfg.INTERIM_DATA_DIR)
    parser.add_argument("--max-rows", type=int, default=3000)
    args = parser.parse_args()
    print(export_sample(args.source, args.output_dir, args.max_rows))


if __name__ == "__main__":
    main()
