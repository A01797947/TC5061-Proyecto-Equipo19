"""Ingeniería de características de MiBici: módulo reutilizable por P1, P2 y P3.

Lo común (carga, limpieza, variables temporales, unión con estaciones y partición
temporal) vive en ``features.base``; lo propio de cada pregunta, en su módulo.
"""

from tc5061_proyecto_equipo19.features.base import (
    agregar_calendario,
    agregar_variables_temporales,
    cargar_datos_viajes_y_estaciones,
    dividir_particiones,
    limpiar_datos_viajes,
    mascara_viajes_consistentes,
    merge_viajes_con_estaciones,
    particion_temporal,
)
from tc5061_proyecto_equipo19.features.p1 import P1_FEATURE_COLUMNS, construir_demanda_diaria
from tc5061_proyecto_equipo19.features.p3 import construir_features_estaciones

__all__ = [
    "P1_FEATURE_COLUMNS",
    "agregar_calendario",
    "agregar_variables_temporales",
    "cargar_datos_viajes_y_estaciones",
    "construir_demanda_diaria",
    "construir_features_estaciones",
    "dividir_particiones",
    "limpiar_datos_viajes",
    "mascara_viajes_consistentes",
    "merge_viajes_con_estaciones",
    "particion_temporal",
]
