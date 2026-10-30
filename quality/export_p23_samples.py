"""Exporta muestras agregadas para DeepChecks P2 y P3 desde datos locales.

Se ejecuta en mlops_tec (Python 3.14), nunca en el entorno DeepChecks.
No requiere MLflow, Supabase ni conexión de red. No exporta datos personales.

P2 utiliza data/processed/p2_features.parquet, que debe estar generado.
P3 utiliza trips_AAAA_MM.parquet y stations.parquet para crear perfiles por
estación de train y test. --max-trips-per-month=0 procesa todos los viajes
(consume más memoria); con un límite mayor que cero es una muestra exploratoria.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.features.p2 import P2_FEATURE_COLUMNS
from tc5061_proyecto_equipo19.features.p3 import construir_features_estaciones


def _sample(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit > 0 and len(df) > limit:
        return df.sample(n=limit, random_state=seed)
    return df


def _write_parts(frames: dict, output_dir: Path, problem: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, frame in frames.items():
        if frame.empty:
            raise ValueError(f"{problem}: la partición {split} está vacía")
        path = output_dir / f"quality_{split}_{problem.lower()}.csv"
        frame.to_csv(path, index=False)
        files[split] = {"file": str(path), "rows": len(frame)}
    return files


def export_p2(source: Path, output_dir: Path, max_rows: int) -> dict:
    """Proyección de columnas: evita cargar toda la tabla horaria a memoria."""
    if not source.is_file():
        raise FileNotFoundError(f"Falta {source}; ejecuta primero 'make features_p2'")
    mandatory = ["station_id", "timestamp", "y_riesgo"] + P2_FEATURE_COLUMNS
    optional = ["y_riesgo_vacia", "y_riesgo_llena"]
    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(source).schema.names)
    missing = set(mandatory) - available
    if missing:
        raise ValueError(f"P2: faltan columnas en Parquet: {sorted(missing)}")
    df = pd.read_parquet(source, columns=mandatory + [c for c in optional if c in available])
    dates = pd.to_datetime(df["timestamp"])
    splits = {
        "train": df.loc[dates < pd.Timestamp(cfg.INICIO_VALIDACION)],
        "test": df.loc[dates >= pd.Timestamp(cfg.INICIO_PRUEBA)],
    }
    # Muestreo determinista, excluye IDs de viaje y usuario.
    sample = {key: _sample(frame, max_rows, cfg.RANDOM_SEED).copy()
              for key, frame in splits.items()}
    for frame in sample.values():
        frame["station_id"] = frame["station_id"].astype("string")
    return _write_parts(sample, output_dir, "P2")


def export_p3(processed_dir: Path, output_dir: Path, max_trips_per_month: int) -> dict:
    """Perfiles de train/test; se ajustan solo con viajes de su propio periodo."""
    stations_path = processed_dir / "stations.parquet"
    if not stations_path.is_file():
        raise FileNotFoundError(f"Falta {stations_path}")
    stations = pd.read_parquet(stations_path,
                               columns=["station_id", "latitude", "longitude", "capacity"])
    trip_cols = ["start_time", "end_time", "origin_station_id", "destination_station_id"]
    periods = {
        "train": (pd.Timestamp(cfg.FECHA_INICIO), pd.Timestamp(cfg.INICIO_VALIDACION)),
        "test": (pd.Timestamp(cfg.INICIO_PRUEBA), pd.Timestamp(cfg.FECHA_FIN) + pd.Timedelta(days=1)),
    }
    profiles = {}
    for split, (start, end) in periods.items():
        files = []
        for year_month in cfg.MESES:
            month_start = pd.Timestamp(year_month.replace("_", "-") + "-01")
            if not (start <= month_start < end):
                continue
            path = processed_dir / f"trips_{year_month}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Falta {path}. Ejecuta primero la ingesta")
            month = pd.read_parquet(path, columns=trip_cols)
            month["start_time"] = pd.to_datetime(month["start_time"], errors="coerce")
            month = month.loc[(month["start_time"] >= start) & (month["start_time"] < end)]
            files.append(_sample(month, max_trips_per_month, cfg.RANDOM_SEED))
        if not files:
            raise ValueError(f"P3: sin viajes procesados en {split}")
        trips = pd.concat(files, ignore_index=True)
        profile = construir_features_estaciones(trips, stations)
        days = (end - start).days
        count_cols = ["viajes_iniciados", "viajes_finalizados", "total_viajes"]
        count_cols += [c for c in profile if c.startswith(("salidas_hora_", "salidas_dia_"))]
        profile[count_cols] = profile[count_cols].astype(float) / days
        profile["station_id"] = profile["station_id"].astype("string")
        profiles[split] = profile
    return _write_parts(profiles, output_dir, "P3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=["P2", "P3"])
    parser.add_argument("--processed-dir", type=Path, default=cfg.PROCESSED_DATA_DIR)
    parser.add_argument("--p2-source", type=Path, default=cfg.FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=cfg.INTERIM_DATA_DIR)
    parser.add_argument("--max-rows", type=int, default=3000,
                        help="Límite de filas P2 por partición; 0 = completo")
    parser.add_argument("--max-trips-per-month", type=int, default=20000,
                        help="Muestreo P3 por mes; 0 = usar todos los viajes")
    args = parser.parse_args()
    if args.max_rows < 0 or args.max_trips_per_month < 0:
        parser.error("Los límites deben ser >= 0")
    result = (export_p2(args.p2_source, args.output_dir, args.max_rows)
              if args.problem == "P2" else
              export_p3(args.processed_dir, args.output_dir, args.max_trips_per_month))
    print(result)
    if args.problem == "P3" and args.max_trips_per_month:
        print("AVISO: P3 usa una muestra de viajes por mes; los volúmenes no representan el total real.")


if __name__ == "__main__":
    main()
