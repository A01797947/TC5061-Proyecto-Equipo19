"""
Variables y etiqueta de P2: riesgo de que una estación se quede sin bicicletas o sin anclajes.

Uso desde la terminal (desde la raíz del repositorio)::

    python -m tc5061_proyecto_equipo19.features.p2
"""

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# P2. Riesgo de que una estación se quede sin bicicletas o sin anclajes
# =============================================================================
# Los datos abiertos no publican el inventario de cada estación, así que el riesgo se
# aproxima con el balance de viajes. Para cada día de servicio (desde las 5:00) se acumula
# llegadas - salidas por hora sobre una ocupación inicial supuesta (50 % de la capacidad,
# tras el rebalanceo nocturno), acotada entre 0 y la capacidad:
#
#     ocupacion_est(s, 5:00) = 0.5
#     ocupacion_est(s, t) = clip(ocupacion_est(s, t-1) + (llegadas - salidas)(s, t) / capacidad(s), 0, 1)
#
# Una fila se emite al cierre de la hora h y su etiqueta mira la ventana h+1 .. h+3:
#     riesgo_vacia = min ocupacion_est ≤ 0.10   (sin bicicletas)
#     riesgo_llena = max ocupacion_est ≥ 0.90   (sin anclajes)
#     y = riesgo_vacia o riesgo_llena
# Los parámetros viven en config_p2.py.

P2_FEATURE_COLUMNS = [
    "hora",
    "dia_semana",
    "fin_de_semana",
    "feriado",
    "mes",
    "hora_sin",
    "hora_cos",
    "capacidad",
    "latitud",
    "longitud",
    "ocupacion_est_h",
    "neto_h",
    "neto_h_1",
    "neto_ult_3h",
    "salidas_h",
    "llegadas_h",
    "neto_ventana_ayer",
    "neto_ventana_semana_pasada",
    "ocup_min_ventana_semana_pasada",
    "ocup_max_ventana_semana_pasada",
    "riesgo_semana_pasada",
    "salidas_sistema_h",
    "actividad_sistema_vs_semana_pasada",
]

# Días feriados oficiales (Ley Federal del Trabajo, art. 74) en el periodo de estudio.
P2_HOLIDAYS = [
    "2024-01-01",
    "2024-02-05",
    "2024-03-18",
    "2024-05-01",
    "2024-09-16",
    "2024-10-01",
    "2024-11-18",
    "2024-12-25",
    "2025-01-01",
    "2025-02-03",
    "2025-03-17",
    "2025-05-01",
    "2025-09-16",
    "2025-11-17",
    "2025-12-25",
]


def _swap_day_month(series: pd.Series) -> pd.Series:
    """Intercambia día y mes de una serie de fechas; deja NaT si la fecha no existe."""
    swapped = pd.to_datetime(
        {
            "year": series.dt.year,
            "month": series.dt.day,
            "day": series.dt.month,
            "hour": series.dt.hour,
            "minute": series.dt.minute,
            "second": series.dt.second,
        },
        errors="coerce",
    )
    return swapped


def corregir_fechas_invertidas(
    df: pd.DataFrame, year: int, month: int
) -> tuple[pd.DataFrame, int]:
    """
    Repara viajes cuyo día y mes quedaron invertidos al leer el CSV.

    La ingesta usa ``format="mixed"`` y en algunos archivos (2024-09 y 2024-10) lee
    ``09/12`` como 9 de diciembre en lugar de 12 de septiembre, lo que produce duraciones
    negativas. Si el inicio de un viaje cae fuera del mes de su archivo y al intercambiar
    día y mes cae dentro, se corrige; lo mismo con el fin cuando así la duración se vuelve
    razonable. Para P2 importa porque una fecha mal leída mueve una salida o una llegada a
    otro mes y deforma el balance de la estación.

    Returns:
        El DataFrame corregido y el número de viajes modificados.
    """
    df = df.copy()
    start = df["start_time"]
    in_month = (start.dt.year == year) & (start.dt.month == month)

    swapped_start = _swap_day_month(start)
    fix_start = ~in_month & (swapped_start.dt.year == year) & (swapped_start.dt.month == month)
    df.loc[fix_start, "start_time"] = swapped_start[fix_start]

    # El fin se corrige si el viaje sigue con duración negativa o de más de un día.
    duration = df["end_time"] - df["start_time"]
    suspicious_end = (duration < pd.Timedelta(0)) | (duration > pd.Timedelta(days=1))
    swapped_end = _swap_day_month(df["end_time"])
    new_duration = swapped_end - df["start_time"]
    fix_end = (
        suspicious_end & (new_duration >= pd.Timedelta(0)) & (new_duration <= pd.Timedelta(days=1))
    )
    df.loc[fix_end, "end_time"] = swapped_end[fix_end]

    return df, int((fix_start | fix_end).sum())


def cargar_viajes_p2(processed_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    Carga los Parquet mensuales con solo las columnas que usa P2 y limpia las fechas.

    Lee archivo por archivo para conocer el mes al que pertenece cada viaje, corrige la
    inversión de día y mes y descarta viajes que siguen siendo inconsistentes: inicio fuera
    del mes de su archivo, duración negativa o mayor a 24 horas. Un viaje se asigna al
    periodo de su ``start_time`` para que no se filtre a otro conjunto de la partición.

    Returns:
        Viajes limpios (inicio, fin, origen, destino) y un resumen de la limpieza.
    """
    columns = ["start_time", "end_time", "origin_station_id", "destination_station_id"]
    frames, stats = [], {"viajes_leidos": 0, "fechas_corregidas": 0, "viajes_descartados": 0}

    for path in sorted(Path(processed_dir).glob("trips_*.parquet")):
        year, month = (int(part) for part in path.stem.split("_")[1:3])
        df = pd.read_parquet(path, columns=columns)
        stats["viajes_leidos"] += len(df)

        df, fixed = corregir_fechas_invertidas(df, year, month)
        stats["fechas_corregidas"] += fixed

        duration = df["end_time"] - df["start_time"]
        valid = (
            (df["start_time"].dt.year == year)
            & (df["start_time"].dt.month == month)
            & (duration >= pd.Timedelta(0))
            & (duration <= pd.Timedelta(days=1))
        )
        stats["viajes_descartados"] += int((~valid).sum())
        frames.append(df[valid])

    if not frames:
        raise FileNotFoundError(f"No hay archivos trips_*.parquet en {processed_dir}")

    trips = pd.concat(frames, ignore_index=True)
    stats["viajes_validos"] = len(trips)
    return trips, stats


def construir_flujos_horarios(
    trips: pd.DataFrame, stations: pd.DataFrame, date_start: str, date_end: str
) -> dict:
    """
    Cuenta salidas y llegadas por estación y hora en matrices estación × hora.

    Las salidas se cuentan en la hora de ``start_time`` en la estación de origen y las
    llegadas en la hora de ``end_time`` en la de destino. Una hora sin viajes vale cero, lo
    que es una observación válida. Solo se conservan estaciones del catálogo con capacidad
    positiva, porque la capacidad es el denominador del riesgo; los viajes de estaciones
    que no están en el catálogo se cuentan en el resumen y se excluyen.

    Returns:
        Diccionario con ``salidas`` y ``llegadas`` (arreglos estación × hora), el índice de
        horas, el catálogo de estaciones alineado y el número de viajes excluidos.
    """
    catalog = stations[stations["capacity"] > 0].copy()
    catalog["station_id"] = catalog["station_id"].astype(str)
    catalog = catalog.drop_duplicates("station_id").reset_index(drop=True)
    station_pos = pd.Series(np.arange(len(catalog)), index=catalog["station_id"])

    hours = pd.date_range(date_start, pd.Timestamp(date_end) + pd.Timedelta(hours=23), freq="h")
    n_stations, n_hours = len(catalog), len(hours)

    def count(station_col: str, time_col: str) -> tuple[np.ndarray, int]:
        station_idx = trips[station_col].astype(str).map(station_pos)
        hour_idx = (trips[time_col].dt.floor("h") - hours[0]) // pd.Timedelta(hours=1)
        ok = station_idx.notna() & (hour_idx >= 0) & (hour_idx < n_hours)
        flat = station_idx[ok].astype(np.int64).to_numpy() * n_hours + hour_idx[ok].to_numpy()
        counts = np.bincount(flat, minlength=n_stations * n_hours)
        return counts.reshape(n_stations, n_hours).astype(np.int32), int((~ok).sum())

    departures, lost_dep = count("origin_station_id", "start_time")
    arrivals, lost_arr = count("destination_station_id", "end_time")

    return {
        "salidas": departures,
        "llegadas": arrivals,
        "horas": hours,
        "estaciones": catalog,
        "salidas_excluidas": lost_dep,
        "llegadas_excluidas": lost_arr,
    }


def _por_dia_de_servicio(matrix: np.ndarray, start_hour: int) -> np.ndarray:
    """Reacomoda un arreglo estación × hora en estación × día de servicio × 24 horas."""
    n_stations, n_hours = matrix.shape
    n_days = int(np.ceil((n_hours - start_hour) / 24))
    padded = np.zeros((n_stations, n_days * 24), dtype=float)
    padded[:, : n_hours - start_hour] = matrix[:, start_hour:]
    return padded.reshape(n_stations, n_days, 24)


def _desplazar_dias(cube: np.ndarray, days: int) -> np.ndarray:
    """Devuelve el valor de ``days`` días antes (NaN cuando no existe)."""
    shifted = np.full_like(cube, np.nan, dtype=float)
    shifted[:, days:] = cube[:, :-days]
    return shifted


def estimar_ocupacion(net: np.ndarray, initial_occupancy: float) -> np.ndarray:
    """
    Estima la ocupación (fracción de la capacidad) al cierre de cada hora del día de servicio.

    Parte de ``initial_occupancy`` a las 5:00 y suma el balance neto hora por hora, acotando
    el resultado a [0, 1]. Una estación no puede tener menos de cero bicicletas ni más que su
    capacidad: si los viajes observados la empujarían fuera de ese rango es porque el operador
    repuso o retiró bicicletas, y desde ahí la estimación continúa. Sin la cota, una estación
    que se vació en la mañana quedaría marcada en riesgo el resto del día aunque reciba
    llegadas.

    Args:
        net: balance (llegadas - salidas) / capacidad, con forma estación × día × 24 horas.
    """
    occupancy = np.empty_like(net)
    level = np.full(net.shape[:2], initial_occupancy)
    for hour in range(net.shape[2]):
        level = np.clip(level + net[:, :, hour], 0.0, 1.0)
        occupancy[:, :, hour] = level
    return occupancy


def construir_dataset_p2(
    flujos: dict,
    service_start_hour: int,
    initial_occupancy: float,
    empty_threshold: float,
    full_threshold: float,
    horizon: int,
    emission_hours: list[int],
) -> pd.DataFrame:
    """
    Construye la tabla estación × hora de emisión con la etiqueta de riesgo y sus variables.

    Etiqueta (ver encabezado de la sección): la ocupación estimada se reinicia al inicio de
    cada día de servicio y avanza con el balance llegadas - salidas, acotada a [0, 1]. La fila emitida al
    cierre de la hora h es positiva si en h+1 .. h+horizon la ocupación estimada baja del
    umbral de vacía o sube del de llena. Como la ventana no cruza el fin del día, ninguna
    etiqueta usa horas de otro día y los cortes de la partición (a medianoche) no se filtran.

    Variables (todas con información hasta h, sin mirar el futuro):

    - Calendario: hora, día de la semana, fin de semana, feriado, mes y la hora en forma
      cíclica. La demanda tiene ciclos diarios y semanales muy marcados.
    - Estación: capacidad y coordenadas. Una estación chica se vacía con menos viajes y la
      ubicación distingue zonas residenciales de laborales.
    - Estado actual: ocupación estimada en h, balance neto en h, h-1 y últimas 3 horas, y
      salidas y llegadas en h. El riesgo es acumulativo: lo que se ha vaciado hasta ahora es
      el mejor indicio de lo que pasará en las siguientes horas.
    - Patrón histórico: balance de la misma ventana ayer y hace una semana, ocupación
      mínima y máxima de esa ventana hace una semana y si entonces hubo riesgo. Capturan la
      repetición del patrón de cada estación a la misma hora.
    - Sistema: salidas totales en h y actividad del día respecto a la semana anterior.
      Detectan días atípicos (feriados, lluvia, eventos) que cambian a toda la red.

    Los balances se expresan como fracción de la capacidad para comparar estaciones de
    distinto tamaño. Las filas fuera del periodo de vida de cada estación (antes de su
    primer viaje o después del último) se descartan para no inflar los negativos con
    estaciones que todavía no existían.
    """
    catalog = flujos["estaciones"]
    capacity = catalog["capacity"].to_numpy(dtype=float)[:, None, None]
    dep = _por_dia_de_servicio(flujos["salidas"], service_start_hour)
    arr = _por_dia_de_servicio(flujos["llegadas"], service_start_hour)
    n_stations, n_days, _ = dep.shape

    net = (arr - dep) / capacity
    occupancy = estimar_ocupacion(net, initial_occupancy)

    offsets = np.array([h - service_start_hour for h in emission_hours])
    window = np.stack([offsets + k for k in range(1, horizon + 1)], axis=1)  # (E, H)

    occ_window = occupancy[:, :, window]  # (S, D, E, H)
    occ_min, occ_max = occ_window.min(axis=3), occ_window.max(axis=3)
    risk_empty = occ_min <= empty_threshold
    risk_full = occ_max >= full_threshold
    net_window = net[:, :, window].sum(axis=3)

    cum_net = np.cumsum(net, axis=2)
    net_last3 = cum_net - np.concatenate(
        [np.zeros((n_stations, n_days, 3)), cum_net[:, :, :-3]], axis=2
    )
    net_prev = np.concatenate([np.zeros((n_stations, n_days, 1)), net[:, :, :-1]], axis=2)

    system_dep = dep.sum(axis=0)  # (D, 24)
    system_cum = np.cumsum(system_dep, axis=1)
    system_cum_lw = np.full_like(system_cum, np.nan)
    system_cum_lw[7:] = system_cum[:-7]
    activity_ratio = (system_cum + 1) / (system_cum_lw + 1)

    risk_any = (risk_empty | risk_full).astype(float)
    cubes = {
        "y_riesgo": risk_any,
        "y_riesgo_vacia": risk_empty.astype(float),
        "y_riesgo_llena": risk_full.astype(float),
        "ocupacion_est_h": occupancy[:, :, offsets],
        "neto_h": net[:, :, offsets],
        "neto_h_1": net_prev[:, :, offsets],
        "neto_ult_3h": net_last3[:, :, offsets],
        "salidas_h": dep[:, :, offsets],
        "llegadas_h": arr[:, :, offsets],
        "neto_ventana_ayer": _desplazar_dias(net_window, 1),
        "neto_ventana_semana_pasada": _desplazar_dias(net_window, 7),
        "ocup_min_ventana_semana_pasada": _desplazar_dias(occ_min, 7),
        "ocup_max_ventana_semana_pasada": _desplazar_dias(occ_max, 7),
        "riesgo_semana_pasada": _desplazar_dias(risk_any, 7),
        "salidas_sistema_h": np.broadcast_to(system_dep[None, :, offsets], risk_any.shape),
        "actividad_sistema_vs_semana_pasada": np.broadcast_to(
            activity_ratio[None, :, offsets], risk_any.shape
        ),
    }

    n_emit = len(offsets)
    station_idx = np.repeat(np.arange(n_stations), n_days * n_emit)
    day_idx = np.tile(np.repeat(np.arange(n_days), n_emit), n_stations)
    emit_idx = np.tile(offsets, n_stations * n_days)

    first_hour = flujos["horas"][0] + pd.Timedelta(hours=service_start_hour)
    timestamps = first_hour + pd.to_timedelta(day_idx * 24 + emit_idx, unit="h")

    df = pd.DataFrame({name: cube.reshape(-1) for name, cube in cubes.items()})
    df.insert(0, "station_id", catalog["station_id"].to_numpy()[station_idx])
    df.insert(1, "timestamp", timestamps)
    df["capacidad"] = capacity.reshape(-1)[station_idx]
    df["latitud"] = catalog["latitude"].to_numpy()[station_idx]
    df["longitud"] = catalog["longitude"].to_numpy()[station_idx]

    df = agregar_calendario_p2(df)

    # Periodo de vida de cada estación según sus viajes observados.
    total = flujos["salidas"] + flujos["llegadas"]
    active = total > 0
    first = np.where(active.any(axis=1), active.argmax(axis=1), total.shape[1])
    last = np.where(active.any(axis=1), total.shape[1] - 1 - active[:, ::-1].argmax(axis=1), -1)
    hour_idx = ((df["timestamp"] - flujos["horas"][0]) // pd.Timedelta(hours=1)).to_numpy()
    alive = (hour_idx >= first[station_idx]) & (hour_idx <= last[station_idx])

    # Las horas de la ventana deben existir en el periodo de datos.
    in_range = df["timestamp"] + pd.Timedelta(hours=horizon) <= flujos["horas"][-1]
    df = df[alive & in_range.to_numpy()].reset_index(drop=True)

    for col in ["y_riesgo", "y_riesgo_vacia", "y_riesgo_llena"]:
        df[col] = df[col].astype("int8")
    float_cols = df.select_dtypes("float64").columns.difference(["latitud", "longitud"])
    df[float_cols] = df[float_cols].astype("float32")
    return df


def agregar_calendario_p2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables de calendario de la hora de emisión.

    La hora en seno y coseno permite que un modelo lineal entienda que las 23:00 y las 0:00
    son contiguas. El fin de semana y los feriados cambian por completo el patrón de viajes
    al trabajo y a la escuela, que es lo que vacía y llena las estaciones.
    """
    ts = df["timestamp"]
    df["hora"] = ts.dt.hour.astype("int8")
    df["dia_semana"] = ts.dt.weekday.astype("int8")
    df["fin_de_semana"] = (df["dia_semana"] >= 5).astype("int8")
    df["feriado"] = ts.dt.normalize().isin(pd.to_datetime(P2_HOLIDAYS)).astype("int8")
    df["mes"] = ts.dt.month.astype("int8")
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    return df


def tasa_historica_riesgo(train: pd.DataFrame, target: str = "y_riesgo") -> pd.DataFrame:
    """
    Calcula la proporción histórica de riesgo por estación, tipo de día y hora.

    Se ajusta solo con el conjunto de entrenamiento y se aplica sin reajustar a validación y
    prueba. Es la base del baseline de P2: si una estación suele estar en riesgo a esa hora,
    se espera que lo vuelva a estar.
    """
    return (
        train.groupby(["station_id", "fin_de_semana", "hora"], observed=True)[target]
        .mean()
        .rename("tasa_hist_riesgo")
        .reset_index()
    )


def build_p2_features(
    processed_dir: Path,
    output_path: Path,
    date_start: str,
    date_end: str,
    service_start_hour: int,
    initial_occupancy: float,
    empty_threshold: float,
    full_threshold: float,
    horizon: int,
    emission_hours: list[int],
) -> tuple[pd.DataFrame, dict]:
    """Ejecuta el pipeline de datos de P2 de punta a punta y guarda la tabla en Parquet."""
    trips, stats = cargar_viajes_p2(processed_dir)
    stations = pd.read_parquet(Path(processed_dir) / "stations.parquet")
    flujos = construir_flujos_horarios(trips, stations, date_start, date_end)
    stats["salidas_excluidas"] = flujos["salidas_excluidas"]
    stats["llegadas_excluidas"] = flujos["llegadas_excluidas"]
    stats["estaciones"] = len(flujos["estaciones"])

    df = construir_dataset_p2(
        flujos,
        service_start_hour,
        initial_occupancy,
        empty_threshold,
        full_threshold,
        horizon,
        emission_hours,
    )
    stats["filas"] = len(df)
    stats["estaciones_con_filas"] = int(df["station_id"].nunique())
    stats["prevalencia_riesgo"] = float(df["y_riesgo"].mean())
    stats["prevalencia_vacia"] = float(df["y_riesgo_vacia"].mean())
    stats["prevalencia_llena"] = float(df["y_riesgo_llena"].mean())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df, stats


if __name__ == "__main__":
    import json

    from tc5061_proyecto_equipo19 import config_p2 as cfg
    from tc5061_proyecto_equipo19.config import FECHA_FIN, FECHA_INICIO, PROCESSED_DATA_DIR

    _, resumen = build_p2_features(
        PROCESSED_DATA_DIR,
        cfg.FEATURES_PATH,
        FECHA_INICIO,
        FECHA_FIN,
        cfg.SERVICE_DAY_START_HOUR,
        cfg.INITIAL_OCCUPANCY,
        cfg.EMPTY_THRESHOLD,
        cfg.FULL_THRESHOLD,
        cfg.HORIZON_HOURS,
        cfg.EMISSION_HOURS,
    )
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
