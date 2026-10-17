"""
Parámetros propios de P2 (riesgo de estación vacía o llena).

Fechas, particiones, semilla y MLflow vienen de ``config.py``; aquí solo vive la
aproximación del riesgo.
"""

from tc5061_proyecto_equipo19.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR

# Aproximación del riesgo.
# El día de servicio empieza a las 5:00 y ahí se reinicia la ocupación estimada.
SERVICE_DAY_START_HOUR = 5
# Ocupación supuesta al iniciar el día (fracción de la capacidad), tras el rebalanceo nocturno.
INITIAL_OCCUPANCY = 0.5
# Umbrales de ocupación estimada que definen "sin bicicletas" y "sin anclajes".
EMPTY_THRESHOLD = 0.10
FULL_THRESHOLD = 0.90
# Horizonte: se mira la ventana h+1 .. h+HORIZON_HOURS, el tiempo para que llegue un camión.
HORIZON_HOURS = 3
# Horas del reloj en que se emite una predicción; la ventana termina antes de medianoche.
EMISSION_HOURS = list(range(6, 21))
# Estaciones que la flota puede atender por hora (precisión en top-k).
TOP_K = 10

FEATURES_PATH = PROCESSED_DATA_DIR / "p2_features.parquet"
P2_MODELS_DIR = MODELS_DIR / "p2"
P2_FIGURES_DIR = FIGURES_DIR / "p2"
