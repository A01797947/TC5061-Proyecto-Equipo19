# Congiguración de la monitorización de la calidad de los datos

TRIP_COLUMNS = ["trip_id", "start_time", "end_time", "origin_station_id"]
TARGET = "cantidad_viajes"

REFERENCE_MONTHS = ("2024_07", "2024_08", "2024_09")
CURRENT_MONTHS = ("2026_07", "2026_08", "2026_09")
CANDIDATE_RECENT_MONTHS = ("2026_07", "2026_08")
EVALUATION_MONTHS = ("2026_09",)

HISTORICAL_TRAIN_MONTHS = tuple(
    f"{year}_{month:02d}" for year, last_month in ((2024, 12), (2025, 6))
    for month in range(1, last_month + 1)
)

# Nueva variable para orquestar la descarga
MESES_A_DESCARGAR = list(CURRENT_MONTHS)