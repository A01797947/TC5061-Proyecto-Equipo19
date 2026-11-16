"""Configuración de monitorización de la calidad de los datos.

Las constantes viven en ``monitoring.data`` (única fuente de verdad); este módulo
solo las reexporta para no mantener dos copias que puedan divergir.
"""
from tc5061_proyecto_equipo19.monitoring.data import (
    CANDIDATE_RECENT_MONTHS,
    CURRENT_MONTHS,
    EVALUATION_MONTHS,
    HISTORICAL_TRAIN_MONTHS,
    REFERENCE_MONTHS,
    TARGET,
    TRIP_COLUMNS,
)

__all__ = [
    "CANDIDATE_RECENT_MONTHS",
    "CURRENT_MONTHS",
    "EVALUATION_MONTHS",
    "HISTORICAL_TRAIN_MONTHS",
    "REFERENCE_MONTHS",
    "TARGET",
    "TRIP_COLUMNS",
]
