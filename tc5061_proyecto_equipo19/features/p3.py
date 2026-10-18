"""Features de P3: perfil de uso por estación para el clustering."""

import pandas as pd


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
    trips = trips.dropna(
        subset=["start_time", "origin_station_id", "destination_station_id"]
    ).copy()

    for c in ["latitude", "longitude", "capacity"]:
        stations[c] = pd.to_numeric(stations[c], errors="coerce")

    stations = stations.dropna(subset=["station_id"]).drop_duplicates("station_id")

    trips["hour"] = trips["start_time"].dt.hour
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

    daily = pd.crosstab(trips["origin_station_id"], trips["weekday"]).reindex(
        columns=range(7), fill_value=0
    )
    daily.columns = [f"salidas_dia_{d}" for d in range(7)]
    daily.index.name = "station_id"

    features = usage.join(hourly, how="outer").join(daily, how="outer").fillna(0).reset_index()
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
