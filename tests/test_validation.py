"""Regresiones de esquema: los datos corruptos deben hacer FALLAR los chequeos."""
import pandas as pd
import pytest
from quality.data_validation import validate_p1_table


@pytest.fixture
def valid_table():
    return pd.DataFrame({
        "Año": [2025, 2025], "Mes": [1, 1], "Dia": [2, 3],
        "Dia_semana": [3, 4], "origin_station_id": ["1", "1"],
        "cantidad_viajes": [12, 15],
    })


def test_tabla_p1_correcta(valid_table):
    assert validate_p1_table(valid_table, "train")["schema"] == "OK"


@pytest.mark.parametrize("problema,match", [
    ("missing", "valores faltantes"),
    ("duplicate", "duplicadas"),
    ("wrong_type", "tipos no numéricos"),
    ("bad_target", "no positiva"),
    ("missing_column", "faltan columnas"),
    ("station_type", "debe ser texto"),
])
def test_tabla_p1_rechaza_errores(valid_table, problema, match):
    df = valid_table.copy()
    if problema == "missing":
        df.loc[0, "cantidad_viajes"] = None
    elif problema == "duplicate":
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    elif problema == "wrong_type":
        df["Mes"] = ["enero", "enero"]
    elif problema == "station_type":
        df["origin_station_id"] = [1, 1]
    elif problema == "bad_target":
        df.loc[0, "cantidad_viajes"] = 0
    else:
        df = df.drop(columns="origin_station_id")
    with pytest.raises(ValueError, match=match):
        validate_p1_table(df, "train")
