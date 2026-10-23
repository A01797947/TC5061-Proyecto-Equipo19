"""Features de P1: demanda diaria de viajes por estación de origen."""

import pandas as pd

P1_FEATURE_COLUMNS = ["Año", "Mes", "Dia", "Dia_semana", "origin_station_id"]


def construir_demanda_diaria(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Agrupa los viajes por fecha y estación de origen y separa X e y.

    Returns:
        ``X`` con las columnas de ``P1_FEATURE_COLUMNS``, ``y`` con la cantidad de
        viajes y la tabla agrupada completa.
    """
    df_agrupado = df.groupby(P1_FEATURE_COLUMNS).size().reset_index(name="cantidad_viajes")
    X = df_agrupado[P1_FEATURE_COLUMNS]
    y = df_agrupado["cantidad_viajes"]
    return X, y, df_agrupado
