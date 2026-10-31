"""Casos normales y límite de la ingesta MiBici, sin lectura de CSV grandes."""
import pandas as pd
import pytest

from tc5061_proyecto_equipo19.dataset import (
    calcular_version_datos,
    clave_columna,
    normalize_columns,
    parsear_fechas,
)


def test_clave_columna_tolera_acentos_espacios_y_texto_mal_codificado():
    assert clave_columna(" Año de Nacimiento ") == "ano_de_nacimiento"
    assert clave_columna("AÃ±o-de-nacimiento") == "ano_de_nacimiento"


def test_normalize_columns_normal_y_falta_obligatoria():
    origen = pd.DataFrame({"Id Viaje": [1], "Fecha Inicio": ["2025-01-02"]})
    mapping = {"id_viaje": "trip_id", "fecha_inicio": "start_time"}
    tabla = normalize_columns(origen, mapping, ["trip_id", "start_time"])
    assert tabla.columns.tolist() == ["trip_id", "start_time"]
    with pytest.raises(ValueError, match="Faltan columnas"):
        normalize_columns(origen, mapping, ["trip_id", "end_time"])


def test_normalize_columns_detecta_colision_por_acentos():
    origen = pd.DataFrame({"Año": [2024], "Ano": [2025]})
    with pytest.raises(ValueError, match="Columnas duplicadas"):
        normalize_columns(origen, {}, [])


def test_parsear_fechas_dia_primero_y_fechas_invalidas():
    fechas, conteos = parsear_fechas(pd.Series([
        "12/10/2024 13:15:00", "2024-10-13 14:00:00", "fecha_no_valida", None,
    ]))
    assert fechas.iloc[0] == pd.Timestamp("2024-10-12 13:15:00")
    assert fechas.iloc[1] == pd.Timestamp("2024-10-13 14:00:00")
    assert fechas.iloc[2:].isna().all()
    assert sum(conteos.values()) == 2


def test_version_datos_invariante_al_orden_y_sensible_a_cambios():
    viajes = [
        {"output_file": "trips_2024_02.parquet", "output_sha256": "bbb"},
        {"output_file": "trips_2024_01.parquet", "output_sha256": "aaa"},
    ]
    estaciones = {"output_file": "stations.parquet", "output_sha256": "ccc"}
    first = calcular_version_datos(viajes, estaciones)
    assert first == calcular_version_datos(list(reversed(viajes)), estaciones)
    assert first != calcular_version_datos(viajes, {**estaciones, "output_sha256": "ddd"})
    assert len(first) == 12
