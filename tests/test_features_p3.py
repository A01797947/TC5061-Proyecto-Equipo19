from operator import index

import pandas as pd
import pytest

from tc5061_proyecto_equipo19.features.p3 import (
    construir_features_estaciones,
)

class TestConstruirFeaturesEstaciones:
    def test_incluye_columnas_esperadoas(self, viajes, estaciones):
        features = construir_features_estaciones(viajes, estaciones)
        assert not features.empty
        assert "viajes_iniciados" in features.columns
        assert "viajes_finalizados" in features.columns
        assert "salidas_hora_00" in features.columns


    def test_excluye_estaciones_sin_viajes(self, viajes, estaciones):
        features = construir_features_estaciones(viajes, estaciones)
        assert all(features["total_viajes"] > 0)

    def test_excluye_estaciones_sin_coordenadas(self, viajes):
        estaciones_sin_coordenadas = pd.DataFrame({
            "station_id": [1, 2],
            "latitude": [None, None],
            "longitude": [None, None],
            "capacity": [10, 15]
        })
        with pytest.raises(ValueError, match="No hay estaciones con viajes y coordenadas válidas."):
            construir_features_estaciones(viajes, estaciones_sin_coordenadas)

    def test_limpia_espacios_en_id_station(self, viajes):
        estaciones_con_espacios = pd.DataFrame({
            "station_id": [" 1", "2 ", " 3 "],
            "latitude": [20.67, 20.68, 20.69],
            "longitude": [-103.35, -103.36, -103.37],
            "capacity": [10, 15, 20]
        })
        features = construir_features_estaciones(viajes, estaciones_con_espacios)
        assert all(features["station_id"].str.strip() == features["station_id"])

    def test_construir_features_estaciones_sin_viajes(self, estaciones):
        viajes_vacios = pd.DataFrame(
            columns=["trip_id", "origin_station_id", "destination_station_id", "start_time", "end_time"]
        )
        with pytest.raises(ValueError, match="No hay estaciones con viajes y coordenadas válidas."):
            construir_features_estaciones(viajes_vacios, estaciones)