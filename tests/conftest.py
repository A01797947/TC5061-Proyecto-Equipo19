"""Carga el paquete directamente desde el checkout; no requiere descargar datos."""
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def viajes():

    return pd.DataFrame(
        {
            "trip_id": [1, 2, 3, 4],
            "origin_station_id": [1, 1, 2, 3],
            "destination_station_id": [2, 3, 1, 1],
            "start_time": pd.to_datetime(
                ["2025-03-03 08:00", "2025-03-03 08:30", "2025-03-03 18:00", "2025-03-08 12:00"]),
            "end_time": pd.to_datetime(
                ["2025-03-03 08:20", "2025-03-03 08:50", "2025-03-03 18:15", "2025-03-08 12:30"])
        }
    )

@pytest.fixture
def estaciones():
    """Catálogo de las tres estaciones usadas en `viajes`."""
    return pd.DataFrame(
        {
            "station_id": [1, 2, 3],
            "latitude": [20.67, 20.68, 20.69],
            "longitude": [-103.35, -103.36, -103.37],
            "capacity": [10, 15, 20]
        }
    )
