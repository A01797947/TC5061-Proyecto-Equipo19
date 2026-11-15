"""Análisis por segmentos del A/B offline: ¿dónde mejora (o empeora) el candidato?

Lee ``predicciones_ab.csv`` (lo genera el comando ``ab``) y desglosa el error de la
versión A (champion) y B (candidato) por tipo de día, por estaciones vistas o no al
entrenar y por estación. Conecta el drift de ``origin_station_id`` con el error real.

Supuesto: el champion se entrenó con ``HISTORICAL_TRAIN_MONTHS`` y el candidato con esos
meses más ``CANDIDATE_RECENT_MONTHS`` (mismo diseño que ``ab.train_candidate``).
"""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.monitoring.data import (
    CANDIDATE_RECENT_MONTHS,
    HISTORICAL_TRAIN_MONTHS,
    missing_parquets,
)

PREDICTIONS_FILE = "predicciones_ab.csv"
REQUIRED_COLUMNS = [
    "Dia_semana", "origin_station_id", "cantidad_viajes",
    "abs_error_baseline", "abs_error_candidate",
]
GROUP_GLOBAL = "global"
GROUP_WEEKDAY = "tipo_dia"
GROUP_STATION_SEEN = "estacion_vista_al_entrenar"
SEEN_BY_BOTH = "vista por A y B"
SEEN_ONLY_BY_B = "solo vista por B (nueva en jul-ago 2026)"
SEEN_BY_NEITHER = "nueva para A y B"


def stations_in_months(months: tuple[str, ...], processed_dir: Path | None = None) -> set[str]:
    """Estaciones de origen presentes en los Parquet mensuales (solo lee esa columna)."""
    processed_dir = Path(processed_dir or cfg.PROCESSED_DATA_DIR)
    missing = missing_parquets(months, processed_dir)
    if missing:
        names = ", ".join(path.name for path in missing[:12])
        raise FileNotFoundError(
            f"Faltan {len(missing)} Parquet en {processed_dir}: {names}. Restáuralos o "
            "descárgalos con `python -m tc5061_proyecto_equipo19.monitoring.descargar_2026 "
            "--inicio AAAA-MM --fin AAAA-MM`."
        )
    found: set[str] = set()
    for month in months:
        column = pd.read_parquet(
            processed_dir / f"trips_{month}.parquet", columns=["origin_station_id"]
        )["origin_station_id"]
        column = column.astype("string").str.strip().dropna()
        found.update(column[column != ""].unique())
    return {str(station) for station in found}


def load_predictions(out: Path) -> pd.DataFrame:
    path = Path(out) / PREDICTIONS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero el comando `ab`.")
    frame = pd.read_csv(path, dtype={"origin_station_id": str})
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} no tiene las columnas esperadas: {missing}")
    return frame


def _group_metrics(frame: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    grouped = frame.groupby(group, observed=True)
    table = grouped.agg(
        n_filas=("cantidad_viajes", "size"),
        n_estaciones=("origin_station_id", "nunique"),
        mae_a=("abs_error_baseline", "mean"),
        mae_b=("abs_error_candidate", "mean"),
    )
    table["mejora_mae"] = table["mae_a"] - table["mae_b"]
    table["mejora_relativa"] = np.where(
        table["mae_a"] > 0, table["mejora_mae"] / table["mae_a"], np.nan
    )
    return table.reset_index(names="grupo")


def compute_segments(
    predictions: pd.DataFrame, stations_a: set[str], stations_b: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (tabla de segmentos en formato largo, tabla por estación)."""
    frame = predictions.copy()
    frame["origin_station_id"] = frame["origin_station_id"].astype(str)
    weekend = frame["Dia_semana"].astype(int) >= 5  # 0 = lunes
    day_type = pd.Series(np.where(weekend, "fin de semana", "entre semana"), index=frame.index)
    seen = np.select(
        [frame["origin_station_id"].isin(stations_a), frame["origin_station_id"].isin(stations_b)],
        [SEEN_BY_BOTH, SEEN_ONLY_BY_B],
        default=SEEN_BY_NEITHER,
    )
    pieces = []
    for name, group in (
        (GROUP_GLOBAL, pd.Series(GROUP_GLOBAL, index=frame.index)),
        (GROUP_WEEKDAY, day_type),
        (GROUP_STATION_SEEN, pd.Series(seen, index=frame.index)),
    ):
        table = _group_metrics(frame, group)
        table.insert(0, "segmento", name)
        pieces.append(table)
    segments = pd.concat(pieces, ignore_index=True)
    by_station = _group_metrics(frame, frame["origin_station_id"]).rename(
        columns={"grupo": "estacion"}
    )
    by_station = by_station.drop(columns=["n_estaciones"]).sort_values(
        "n_filas", ascending=False
    ).reset_index(drop=True)
    return segments, by_station


def _plot(segments: pd.DataFrame, by_station: pd.DataFrame, out: Path, top_n: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = segments[segments["segmento"] != GROUP_GLOBAL]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    positions = np.arange(len(selected))
    axes[0].bar(positions - 0.2, selected["mae_a"], width=0.4, label="A (champion)")
    axes[0].bar(positions + 0.2, selected["mae_b"], width=0.4, label="B (candidato)")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        [
            f"{textwrap.fill(g, 16, break_on_hyphens=False)}\n(n={n:,})"
            for g, n in zip(selected["grupo"], selected["n_filas"])
        ],
        fontsize=7,
    )
    axes[0].set_ylabel("MAE (menor es mejor)")
    axes[0].set_title("MAE por tipo de día y estaciones vistas")
    axes[0].legend()
    top = by_station.head(top_n).iloc[::-1]
    stations = np.arange(len(top))
    axes[1].barh(stations - 0.2, top["mae_a"], height=0.4, label="A (champion)")
    axes[1].barh(stations + 0.2, top["mae_b"], height=0.4, label="B (candidato)")
    axes[1].set_yticks(stations)
    axes[1].set_yticklabels(top["estacion"].astype(str), fontsize=7)
    axes[1].set_xlabel("MAE (menor es mejor)")
    axes[1].set_title(f"Las {top_n} estaciones con más filas")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(Path(out) / "segmentos_ab.png", dpi=170)
    plt.close(fig)


def generate_segments(out: Path, processed_dir: Path | None = None, top_n: int = 15) -> dict:
    """Escribe segmentos_ab.csv, mae_por_estacion.csv, segmentos_ab.png y el resumen JSON."""
    out = Path(out)
    predictions = load_predictions(out)
    stations_a = stations_in_months(HISTORICAL_TRAIN_MONTHS, processed_dir)
    stations_b = stations_a | stations_in_months(CANDIDATE_RECENT_MONTHS, processed_dir)
    segments, by_station = compute_segments(predictions, stations_a, stations_b)
    segments.to_csv(out / "segmentos_ab.csv", index=False)
    by_station.to_csv(out / "mae_por_estacion.csv", index=False)
    _plot(segments, by_station, out, top_n)
    seen = segments[segments["segmento"] == GROUP_STATION_SEEN]
    total_rows = len(predictions)
    new_rows = int(seen.loc[seen["grupo"] != SEEN_BY_BOTH, "n_filas"].sum())
    worse = by_station[by_station["mejora_mae"] < 0]
    summary = {
        "n_filas": total_rows,
        "n_estaciones_holdout": int(predictions["origin_station_id"].nunique()),
        "estaciones_nuevas_para_A": len(set(predictions["origin_station_id"]) - stations_a),
        "estaciones_nuevas_para_A_y_B": len(
            set(predictions["origin_station_id"]) - stations_b
        ),
        "proporcion_filas_en_estaciones_nuevas_para_A": new_rows / total_rows,
        "estaciones_donde_B_empeora": len(worse),
        "supuesto": (
            "A se entrenó con HISTORICAL_TRAIN_MONTHS y B con esos meses más "
            "CANDIDATE_RECENT_MONTHS."
        ),
    }
    (out / "resumen_segmentos.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
