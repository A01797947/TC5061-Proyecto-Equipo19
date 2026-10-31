from operator import index

import pandas as pd
import pytest

from tc5061_proyecto_equipo19.features.base import (
    limpiar_datos_viajes,
    agregar_variables_temporales,
    mascara_viajes_consistentes,
    merge_viajes_con_estaciones,
    particion_temporal,
    dividir_particiones,
    agregar_calendario,
    cargar_datos_viajes_y_estaciones,
)


class TestLimpiarDatosViajes:
    def test_conserva_datos_validos(self, viajes):
        viajes_limpios = limpiar_datos_viajes(viajes)
        assert len(viajes_limpios) == 4
        assert set(viajes_limpios["trip_id"]) == {1, 2, 3, 4}

    def test_elimina_fecha_fin_anterior_inicio(self, viajes):
        viajes = viajes.copy()
        viajes.loc[1, "end_time"] = pd.to_datetime("2025-03-03 08:00")
        viajes_limpios = limpiar_datos_viajes(viajes)
        assert len(viajes_limpios) == 3
        assert set(viajes_limpios["trip_id"]) == {1, 3, 4}

    @pytest.mark.parametrize("columna", ["start_time", "end_time", "origin_station_id"])
    def test_elimina_viajes_con_valores_nulos_columnas_obligatorias(
        self, viajes, columna
    ):
        viajes = viajes.copy()
        viajes.loc[0, columna] = None
        viajes_limpios = limpiar_datos_viajes(viajes)
        assert len(viajes_limpios) == 3
        assert set(viajes_limpios["trip_id"]) == {2, 3, 4}

    def test_elimina_trip_id_duplicados(self, viajes):
        viajes = viajes.copy()
        viajes.loc[1, "trip_id"] = 1
        viajes_limpios = limpiar_datos_viajes(viajes)
        assert len(viajes_limpios) == 3
        assert set(viajes_limpios["trip_id"]) == {1, 3, 4}
        assert viajes_limpios.loc[viajes_limpios["trip_id"] == 1, "start_time"].iloc[
            0
        ] == pd.Timestamp("2025-03-03 08:00")

    def test_dataframe_vacio(self):
        viajes = pd.DataFrame(
        columns=[
                "trip_id",
                "origin_station_id",
                "destination_station_id",
                "start_time",
                "end_time",
            ]
        )
        viajes_limpios = limpiar_datos_viajes(viajes)
        assert viajes_limpios.empty


class TestAgregarVariablesTemporales:
    def test_calcula_columnas(self, viajes):
        viajes_con_variables = agregar_variables_temporales(viajes)

        assert viajes_con_variables["Año"].unique().tolist() == [2025]
        assert viajes_con_variables["Mes"].unique().tolist() == [3]
        assert viajes_con_variables["Dia"].tolist() == [3, 3, 3, 8]
        assert viajes_con_variables["Dia_semana"].tolist() == [0, 0, 0, 5]
        assert viajes_con_variables["Hora"].tolist() == [8, 8, 18, 12]
        assert viajes_con_variables["duracion_viaje"].tolist() == pytest.approx([20, 20, 15, 30])

    def test_no_modifica_dataframe_original(self, viajes):
        viajes_original = viajes.copy()
        agregar_variables_temporales(viajes)
        pd.testing.assert_frame_equal(viajes, viajes_original)

    def test_media_noche(self, viajes):
        viajes = viajes.copy()
        viajes.loc[0, "start_time"] = pd.to_datetime("2025-03-03 23:50")
        viajes.loc[0, "end_time"] = pd.to_datetime("2025-03-04 00:10")
        viajes_con_variables = agregar_variables_temporales(viajes)

        assert viajes_con_variables.loc[0, "Año"] == 2025
        assert viajes_con_variables.loc[0, "Mes"] == 3
        assert viajes_con_variables.loc[0, "Dia"] == 3
        assert viajes_con_variables.loc[0, "Dia_semana"] == 0
        assert viajes_con_variables.loc[0, "Hora"] == 23
        assert viajes_con_variables.loc[0, "duracion_viaje"] == pytest.approx(20)

class TestMascaraViajesConsistentes():
    def test_viajes_consistentes(self, viajes):
        mascara = mascara_viajes_consistentes(viajes, 2025, 3)
        assert mascara.tolist() == [True, True, True, True]

    def test_mes_equivocado(self, viajes):
        mascara = mascara_viajes_consistentes(viajes, 2025, 4)
        assert mascara.tolist() == [False, False, False, False]

    def test_anio_equivocado(self, viajes):
        mascara = mascara_viajes_consistentes(viajes, 2024, 3)
        assert mascara.tolist() == [False, False, False, False]

    def test_duracion_maxima(self, viajes):
        viajes = viajes.copy()
        viajes.loc[0, "end_time"] = pd.to_datetime("2025-03-03 09:30")
        mascara = mascara_viajes_consistentes(viajes, 2025, 3, duracion_maxima=pd.Timedelta(minutes=25))
        assert mascara.tolist() == [False, True, True, False]

    def test_fin_nulo(self, viajes):
        viajes = viajes.copy()
        viajes.loc[0, "end_time"] = pd.NaT
        mascara = mascara_viajes_consistentes(viajes, 2025, 3)
        assert mascara.tolist() == [False, True, True, True]


    @pytest.mark.parametrize("duracion, esperado", 
        [
            (pd.Timedelta(seconds=-1), False),
            (pd.Timedelta(0), True),
            (pd.Timedelta(days=1), True),
            (pd.Timedelta(days=1, seconds=1), False),
        ],
        ids=["negativo", "cero", "maximo", "superior_maximo"]
        )
    def test_duracion_limites(self, viajes, duracion, esperado):
        viajes = viajes.copy()
        viajes.loc[0, "end_time"] = viajes.loc[0, "start_time"] + duracion
        mascara = mascara_viajes_consistentes(viajes, 2025, 3)
        assert mascara.tolist()[0] == esperado

class TestMergeViajesConEstaciones:
    def test_agrega_columnas_de_origen_y_destino(self, viajes, estaciones):
        viajes_con_estaciones = merge_viajes_con_estaciones(viajes, estaciones)
        assert "origin_latitude" in viajes_con_estaciones.columns
        assert "origin_longitude" in viajes_con_estaciones.columns
        assert "origin_capacity" in viajes_con_estaciones.columns
        assert "destination_latitude" in viajes_con_estaciones.columns
        assert "destination_longitude" in viajes_con_estaciones.columns
        assert "destination_capacity" in viajes_con_estaciones.columns

    def test_asigna_a_cada_viaje_los_datos_de_sus_estaciones(self, viajes, estaciones):
        viajes_con_estaciones = merge_viajes_con_estaciones(viajes, estaciones)

        esperado = pd.DataFrame(
            {
                "origin_latitude": [20.67, 20.67, 20.68, 20.69],
                "origin_longitude": [-103.35, -103.35, -103.36, -103.37],
                "origin_capacity": [10, 10, 15, 20],
                "destination_latitude": [20.68, 20.69, 20.67, 20.67],
                "destination_longitude": [-103.36, -103.37, -103.35, -103.35],
                "destination_capacity": [15, 20, 10, 10],
            }
        )

        pd.testing.assert_frame_equal(viajes_con_estaciones[esperado.columns], esperado)


    def test_viaje_no_existe_en_estaciones(self, viajes, estaciones):
        viajes = viajes.copy()
        viajes.loc[0, "origin_station_id"] = 999
        viajes.loc[1, "destination_station_id"] = 999
        viajes_con_estaciones = merge_viajes_con_estaciones(viajes, estaciones)

        assert len(viajes_con_estaciones) == 4
        assert pd.isna(viajes_con_estaciones.loc[0, "origin_latitude"])
        assert pd.isna(viajes_con_estaciones.loc[1, "destination_latitude"])
        assert viajes_con_estaciones.loc[2, "origin_latitude"] == 20.68

    def test_no_modifica_dataframe_original(self, viajes, estaciones):
        viajes_original = viajes.copy()
        estaciones_original = estaciones.copy()
        merge_viajes_con_estaciones(viajes, estaciones)
        pd.testing.assert_frame_equal(viajes, viajes_original)
        pd.testing.assert_frame_equal(estaciones, estaciones_original)


class TestParticionTemporal:
    def test_particion_temporal(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2025-06-30 23:59:59", "2025-07-01", "2025-09-30 23:59:59",
                "2025-10-01", "2025-06-30 23:59:59.500", "2025-09-30 23:59:59.500",
            ], format="mixed"),
            "id": [1, 2, 3, 4, 5, 6],
        })
        particiones = particion_temporal(df, "timestamp")
        assert particiones.iloc[0] == "train"
        assert particiones.iloc[1] == "valid"
        assert particiones.iloc[2] == "valid"
        assert particiones.iloc[3] == "test"
        assert particiones.iloc[4] == "train"
        assert particiones.iloc[5] == "valid"

class TestDividirParticiones:
    def test_dividir_particiones_respetando_cortes_y_huecos(self):
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

class TestAgregarCalendario:
    def test_agregar_calendario_agrega_columnas_correctamente(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2025-01-01 00:00", "2025-01-02 12:00", "2025-01-03 23:59"
            ])
        })
        df_con_calendario = agregar_calendario(df, "timestamp")
        assert "mes" in df_con_calendario.columns
        assert "fin_de_semana" in df_con_calendario.columns
        assert "dia_semana" in df_con_calendario.columns
        assert df_con_calendario["mes"].tolist() == [1, 1, 1]
        assert df_con_calendario["fin_de_semana"].tolist() == [0, 0, 0]
        assert df_con_calendario["dia_semana"].tolist() == [2, 3, 4]  # Miércoles, Jueves, Viernes


class TestCargarDatos:
    def test_cargar_datos(self):
        df_trips, df_stations = cargar_datos_viajes_y_estaciones()
        assert not df_trips.empty
        assert not df_stations.empty
        assert "trip_id" in df_trips.columns
        assert "station_id" in df_stations.columns

        
    def test_sin_archivos_lanza_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="trips_"):
            cargar_datos_viajes_y_estaciones(tmp_path)