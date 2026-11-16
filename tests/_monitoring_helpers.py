"""Datos mínimos compartidos por las pruebas de segmentos y ablación del A/B."""

import pandas as pd


def predictions_frame(n_days: int = 10, stations=("1", "2", "3", "4")) -> pd.DataFrame:
    """Predicciones A/B sintéticas con el mismo formato que `predicciones_ab.csv`."""
    rows = []
    for day in range(1, n_days + 1):
        weekday = (day + 1) % 7  # 0 = lunes; hay fines de semana en los datos
        for position, station in enumerate(stations):
            real = 20 + 5 * position + (weekday >= 5) * -4
            baseline = real + 6 + position
            candidate = real + 2
            rows.append({
                "Año": 2026, "Mes": 9, "Dia": day, "Dia_semana": weekday,
                "origin_station_id": station, "cantidad_viajes": real,
                "pred_baseline": baseline, "pred_candidate": candidate,
                "abs_error_baseline": abs(real - baseline),
                "abs_error_candidate": abs(real - candidate),
            })
    return pd.DataFrame(rows)


def write_month_parquets(directory, months, stations):
    """Un Parquet mensual mínimo por mes con las columnas que lee `load_p1_months`."""
    for month in months:
        year, mo = (int(x) for x in month.split("_"))
        rows = []
        for day in range(1, 11):
            for station in stations:
                for trip in range(3):
                    start = pd.Timestamp(year=year, month=mo, day=day, hour=8 + trip)
                    rows.append((f"{month}-{day}-{station}-{trip}", start,
                                 start + pd.Timedelta(minutes=10), station))
        pd.DataFrame(
            rows, columns=["trip_id", "start_time", "end_time", "origin_station_id"]
        ).to_parquet(directory / f"trips_{month}.parquet", index=False)
