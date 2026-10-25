"""Features de P3: perfil de uso por estación para el clustering."""

import re

import pandas as pd

from tc5061_proyecto_equipo19 import config as cfg


def _percentage_by_hour(hourly: pd.DataFrame, prefix: str)-> pd.DataFrame:
    """Obtiene el porcentaje de uso por hora """
    pct_hourly = hourly.copy()
    pct_hourly = pct_hourly.div(pct_hourly.sum(axis=1), axis=0).fillna(0.0)

    pct_hourly.rename(columns=lambda x: re.sub(prefix,f'pct_{prefix}',x), inplace=True)
    return pct_hourly

def _weekend_ratio(daily: pd.DataFrame) -> pd.Series:

    weekly = daily[[f"salidas_dia_{d}" for d in range(5)]]
    weekend = daily[[f"salidas_dia_{d}" for d in range(5, 7)]]
    weekly_avg = weekly.mean(axis= 1)
    weekend_avg = weekend.mean(axis=1)
    ratio = (weekend_avg / weekly_avg).where(weekly_avg > 0, 0.0)
    ratio = ratio.clip(upper=cfg.P3_RATIO_FINDE_MAX)
    return ratio.rename("ratio_finde")

def construir_features_estaciones(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Construye una fila por estación con su uso total, perfil horario y semanal.

    Incluye viajes iniciados y finalizados, salidas por hora (0-23) y por día de la
    semana (0-6), y las coordenadas y capacidad de la estación.
    """
    trips, stations = trips.copy(), stations.copy()

    for c in ["origin_station_id", "destination_station_id"]:
        trips[c] = trips[c].astype("string").str.strip()

    stations["station_id"] = stations["station_id"].astype("string").str.strip()
    trips["start_time"] = pd.to_datetime(trips["start_time"], errors="coerce")
    trips["end_time"] = pd.to_datetime(trips["end_time"], errors="coerce")
    trips = trips.dropna(
        subset=["start_time", "origin_station_id", "destination_station_id"]
    ).copy()

    for c in ["latitude", "longitude", "capacity"]:
        stations[c] = pd.to_numeric(stations[c], errors="coerce")

    stations = stations.dropna(subset=["station_id"]).drop_duplicates("station_id")

    trips["hour"] = trips["start_time"].dt.hour
    trips["arrive_hour"] = trips["end_time"].dt.hour
    trips["weekday"] = trips["start_time"].dt.dayofweek

    out = (
        trips.groupby("origin_station_id")
        .size()
        .rename_axis("station_id")
        .rename("viajes_iniciados")
    )
    inc = (
        trips.groupby("destination_station_id")
        .size()
        .rename_axis("station_id")
        .rename("viajes_finalizados")
    )

    usage = pd.concat([out, inc], axis=1).fillna(0)
    usage["total_viajes"] = usage["viajes_iniciados"] + usage["viajes_finalizados"]

    hourly = pd.crosstab(trips["origin_station_id"], trips["hour"]).reindex(
        columns=range(24), fill_value=0
    )
    hourly.columns = [f"salidas_hora_{h:02d}" for h in range(24)]
    hourly.index.name = "station_id"

    arrivals = pd.crosstab(trips["destination_station_id"], trips["arrive_hour"]).reindex(
        columns=range(24), fill_value=0
    )
    arrivals.columns = [f"llegadas_hora_{h:02d}" for h in range(24)]
    arrivals.index.name = "station_id"

    daily = pd.crosstab(trips["origin_station_id"], trips["weekday"]).reindex(
        columns=range(7), fill_value=0
    )
    daily.columns = [f"salidas_dia_{d}" for d in range(7)]
    daily.index.name = "station_id"

    features = (
            usage.join(hourly, how="outer")
            .join(daily, how="outer")
            .join(_percentage_by_hour(hourly, "salidas"), how="outer")
            .join(_percentage_by_hour(arrivals, "llegadas"), how="outer")
            .join(_weekend_ratio(daily), how="outer")
            .fillna(0)
            .reset_index()
    )
    features = features.merge(
        stations[["station_id", "latitude", "longitude", "capacity"]],
        on="station_id",
        how="left",
        validate="one_to_one",
    )
    features = features.dropna(subset=["latitude", "longitude"])
    features = features.loc[features["total_viajes"] > 0].reset_index(drop=True)

    if features.empty:
        raise ValueError("No hay estaciones con viajes y coordenadas válidas.")

    return features
