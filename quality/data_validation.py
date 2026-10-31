"""Validaciones estrictas del conjunto agregado P1 (sin depender de DeepChecks).

DeepChecks añade diagnósticos de drift; los errores de esquema/integridad son
bloqueantes y se pueden probar en la versión principal de Python del proyecto.
"""
from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

REQUIRED_FEATURES = ["Año", "Mes", "Dia", "Dia_semana", "origin_station_id"]
TARGET = "cantidad_viajes"


def validate_p1_table(table: pd.DataFrame, name: str) -> dict:
    """Levanta ValueError si hay esquema inválido, nulos o filas duplicadas."""
    required = REQUIRED_FEATURES + [TARGET]
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"{name}: faltan columnas: {missing}")
    if table.empty:
        raise ValueError(f"{name}: no contiene registros")
    if table[required].isna().any().any():
        raise ValueError(f"{name}: existen valores faltantes en columnas requeridas")
    numerics = ["Año", "Mes", "Dia", "Dia_semana", TARGET]
    wrong = [col for col in numerics if not is_numeric_dtype(table[col])]
    if wrong:
        raise ValueError(f"{name}: tipos no numéricos en {wrong}")
    if not is_string_dtype(table["origin_station_id"]):
        raise ValueError(f"{name}: identificador de estación debe ser texto")
    if table["origin_station_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}: estación vacía")
    if table.duplicated(subset=REQUIRED_FEATURES).any():
        raise ValueError(f"{name}: filas duplicadas por fecha y estación")
    if (table[TARGET] < 1).any():
        raise ValueError(f"{name}: demanda no positiva")
    if not table["Mes"].between(1, 12).all() or not table["Dia"].between(1, 31).all():
        raise ValueError(f"{name}: fecha fuera de rango")
    if not table["Dia_semana"].between(0, 6).all():
        raise ValueError(f"{name}: día de semana fuera de rango")
    return {"dataset": name, "rows": len(table), "nulls": 0, "duplicates": 0,
            "schema": "OK"}
