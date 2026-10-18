"""Ingeniería de características de MiBici.

Las funciones vienen del notebook ``1.0-JMRR-EDA``: carga, limpieza, variables
temporales, unión con estaciones y partición temporal. Las variables de cada
pregunta se agregan sobre esta base.

Regla contra fuga de información: ninguna variable de una fila con hora ``t``
puede usar viajes posteriores a ``t``; las estadísticas se calculan solo con el
periodo de entrenamiento.

Uso desde la terminal:

    python -m tc5061_proyecto_equipo19.features
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from tc5061_proyecto_equipo19.config import (
    FECHA_FIN,
    FECHA_INICIO,
    FERIADOS,
    FIN_ENTRENAMIENTO,
    FIN_VALIDACION,
    INICIO_PRUEBA,
    INICIO_VALIDACION,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)

DURACION_MAXIMA_VIAJE = pd.Timedelta(days=1)

app = typer.Typer()


# ---------------------------------------------------------------------------
# Comunes
# ---------------------------------------------------------------------------
def cargar_datos_viajes_y_estaciones(
    processed_dir: Path = PROCESSED_DATA_DIR,
    fecha_inicio: str = FECHA_INICIO,
    fecha_fin: str = FECHA_FIN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los Parquet mensuales de viajes dentro del rango de fechas y el catálogo.

    Usar el rango de ``config`` garantiza que todos los runs trabajen sobre el
    mismo periodo que se registra en MLflow.
    """
    processed_dir = Path(processed_dir)
    parquet_files = sorted(processed_dir.glob("trips_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No hay trips_*.parquet en {processed_dir}. Corre la ingesta.")

    df_trips = pd.concat((pd.read_parquet(p) for p in parquet_files), ignore_index=True)
    inicio = pd.Timestamp(fecha_inicio)
    fin = pd.Timestamp(fecha_fin) + pd.Timedelta(days=1)
    df_trips = df_trips[(df_trips["start_time"] >= inicio) & (df_trips["start_time"] < fin)]
    df_stations = pd.read_parquet(processed_dir / "stations.parquet")

    logger.info(f"Viajes cargados: {len(df_trips):,} | Estaciones: {len(df_stations)}")
    return df_trips.reset_index(drop=True), df_stations


def agregar_variables_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega año, mes, día, día de la semana, hora y duración del viaje en minutos.

    El EDA mostró estacionalidad semanal (martes a jueves con más uso, caída en
    fin de semana) y anual (baja desde agosto). Estas columnas permiten medir esos
    patrones; la duración solo se usa para limpieza, no como predictor, porque se
    conoce hasta que el viaje termina.
    """
    df = df.copy()
    df["Año"] = df["start_time"].dt.year
    df["Mes"] = df["start_time"].dt.month
    df["Dia"] = df["start_time"].dt.day
    df["Dia_semana"] = df["start_time"].dt.weekday
    df["Hora"] = df["start_time"].dt.hour
    df["duracion_viaje"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60.0
    return df


def agregar_calendario(df: pd.DataFrame, columna_tiempo: str = "timestamp") -> pd.DataFrame:
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
    df["feriado"] = ts.dt.normalize().isin(pd.to_datetime(FERIADOS)).astype("int8")
    df["mes"] = ts.dt.month.astype("int8")
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    return df


def limpiar_datos_viajes(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina viajes con fechas inválidas, con fin anterior al inicio y duplicados.

    El EDA encontró 115 viajes (0.0012 %) con duración negativa; se descartan por
    ser errores de registro. Los duplicados se quitan por ``trip_id`` para no
    contar dos veces la demanda de una estación.
    """
    antes = len(df)
    df_limpio = df.dropna(subset=["start_time", "end_time", "origin_station_id"])
    df_limpio = df_limpio[df_limpio["start_time"] <= df_limpio["end_time"]]
    df_limpio = df_limpio.drop_duplicates(subset="trip_id").reset_index(drop=True)
    logger.info(
        f"Limpieza: {antes:,} -> {len(df_limpio):,} viajes ({antes - len(df_limpio):,} fuera)"
    )
    return df_limpio


def mascara_viajes_consistentes(
    df: pd.DataFrame, anio: int, mes: int, duracion_maxima: pd.Timedelta = DURACION_MAXIMA_VIAJE
) -> pd.Series:
    """Marca los viajes consistentes con el archivo mensual del que provienen.

    Un viaje es consistente si su inicio cae en el mes del archivo y su duración
    está entre cero y ``duracion_maxima``. Los ``NaT`` quedan fuera de forma
    implícita porque las comparaciones con ellos son falsas.
    """

    duracion = df["end_time"] - df["start_time"]
    return (
        (df["start_time"].dt.year == anio)
        & (df["start_time"].dt.month == mes)
        & (duracion >= pd.Timedelta(0))
        & (duracion <= duracion_maxima)
    )


def merge_viajes_con_estaciones(df_trips: pd.DataFrame, df_stations: pd.DataFrame) -> pd.DataFrame:
    """Agrega coordenadas y capacidad de las estaciones de origen y destino.

    La ubicación y la capacidad explican diferencias de uso entre estaciones; la
    capacidad además define cuándo una estación se queda sin bicis o sin anclajes.
    """
    cols = ["station_id", "latitude", "longitude", "capacity"]
    resultado = df_trips
    for lado in ("origin", "destination"):
        renombre = {c: f"{lado}_{c}" for c in cols[1:]}
        resultado = resultado.merge(
            df_stations[cols].rename(columns=renombre),
            left_on=f"{lado}_station_id",
            right_on="station_id",
            how="left",
        ).drop(columns="station_id")
    return resultado


def particion_temporal(df: pd.DataFrame, columna_tiempo: str) -> pd.Series:
    """Etiqueta cada fila como ``train``, ``valid`` o ``test`` según su fecha.

    La partición es cronológica (fechas en ``config``): el modelo nunca ve meses
    posteriores a los que se evalúan, igual que en la operación real.
    """
    t = df[columna_tiempo]
    return pd.Series(
        np.select(
            [
                t <= pd.Timestamp(FIN_ENTRENAMIENTO),
                (t >= pd.Timestamp(INICIO_VALIDACION)) & (t <= pd.Timestamp(FIN_VALIDACION)),
                t >= pd.Timestamp(INICIO_PRUEBA),
            ],
            ["train", "valid", "test"],
            default="fuera",
        ),
        index=df.index,
        name="particion",
    )


def dividir_particiones(
    df: pd.DataFrame, columna_tiempo: str = "timestamp"
) -> dict[str, pd.DataFrame]:
    """Divide ``df`` en ``train``, ``valid`` y ``test`` con las fechas de ``config.py``.

    Las filas que no caen en ninguna partición (etiqueta ``fuera``) se descartan.
    """

    etiquetas = particion_temporal(df, columna_tiempo)
    return {
        nombre: df[etiquetas == nombre].reset_index(drop=True)
        for nombre in ("train", "valid", "test")
    }


@app.command()
def main(
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR / "viajes_limpios_enriquecidos.parquet",
) -> None:
    """Genera el dataset común de viajes limpios y enriquecidos en data/interim."""
    df_trips, df_stations = cargar_datos_viajes_y_estaciones(processed_dir)
    df = limpiar_datos_viajes(agregar_variables_temporales(df_trips))
    df = merge_viajes_con_estaciones(df, df_stations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(f"Guardado {output_path}")


if __name__ == "__main__":
    app()
