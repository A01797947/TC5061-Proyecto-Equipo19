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
    merge_viajes_con_estaciones,
    particion_temporal,
)

__all__ = [
    "agregar_calendario",
    "agregar_variables_temporales",
    "cargar_datos_viajes_y_estaciones",
    "dividir_particiones",
    "limpiar_datos_viajes",
    "merge_viajes_con_estaciones",
    "particion_temporal",
]
