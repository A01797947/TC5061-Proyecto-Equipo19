"""Pruebas de limpieza, variables, demanda P1 y particionado temporal."""
import numpy as np
import pandas as pd

from tc5061_proyecto_equipo19.features.base import (
    agregar_variables_temporales, dividir_particiones,
    limpiar_datos_viajes, mascara_viajes_consistentes,
)
from tc5061_proyecto_equipo19.features.p1 import construir_demanda_diaria


def test_limpiar_datos_elimina_duplicados_invalidos_y_conserva_duracion_cero():
    viajes = pd.DataFrame({
        "trip_id": ["ok", "ok", "negativo", "sin_fecha", "cero", "sin_estacion"],
        "start_time": pd.to_datetime([
            "2025-01-05 12:00", "2025-01-05 12:00", "2025-01-05 12:00",
            None, "2025-01-05 12:00", "2025-01-05 12:00",
        ]),
        "end_time": pd.to_datetime([
            "2025-01-05 12:10", "2025-01-05 12:10", "2025-01-05 11:59",
            "2025-01-05 13:00", "2025-01-05 12:00", "2025-01-05 12:30",
        ]),
        "origin_station_id": ["1", "1", "1", "1", "2", None],
    })
    limpios = limpiar_datos_viajes(viajes)
    assert limpios["trip_id"].tolist() == ["ok", "cero"]
    assert len(limpios) == 2


def test_mascara_viajes_consistentes_casos_limite():
    start = pd.to_datetime(["2025-01-01", "2025-01-01", "2025-02-01", None])
    end = pd.to_datetime(["2025-01-02", "2025-01-02 00:00:01", "2025-02-01 00:01", None], format="mixed")
    trips = pd.DataFrame({"start_time": start, "end_time": end})
    assert mascara_viajes_consistentes(trips, 2025, 1).tolist() == [True, False, False, False]


def test_variables_temporales_y_demanda_diaria():
    viajes = pd.DataFrame({
        "start_time": pd.to_datetime(["2025-01-06 08:00", "2025-01-06 09:00", "2025-01-06 10:00"]),
        "end_time": pd.to_datetime(["2025-01-06 08:10", "2025-01-06 09:20", "2025-01-06 10:30"]),
        "origin_station_id": ["A", "A", "B"],
    })
    enriquecido = agregar_variables_temporales(viajes)
    assert enriquecido["Dia_semana"].tolist() == [0, 0, 0]
    assert enriquecido["duracion_viaje"].tolist() == [10, 20, 30]
    assert "Año" not in viajes.columns  # la función no muta su entrada
    X, y, agrupado = construir_demanda_diaria(enriquecido)
    assert X.columns.tolist() == ["Año", "Mes", "Dia", "Dia_semana", "origin_station_id"]
    assert dict(zip(agrupado.origin_station_id, y)) == {"A": 2, "B": 1}


def test_dividir_particiones_respetando_cortes_y_huecos():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-06-30 23:59:59", "2025-07-01", "2025-09-30 23:59:59",
            "2025-10-01", "2025-06-30 23:59:59.500",
        ], format="mixed"),
        "id": [1, 2, 3, 4, 5],
    })
    cortes = dividir_particiones(df)
    assert cortes["valid"]["id"].tolist() == [2, 3]
    assert cortes["test"]["id"].tolist() == [4]
    # Un milisegundo posterior a 23:59:59 también forma parte del entrenamiento.
    assert cortes["train"]["id"].tolist() == [1, 5]
