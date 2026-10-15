"""
Módulo de Ingeniería de Características (features.py)
------------------------------------------------------
Este módulo contiene las funciones de transformación para construir el dataset común sobre el 
que se trabajarán los modelos de Machine Learning. Incluye funciones para cargar, limpiar, enriquecer y transformar los datos de viajes y estaciones.
"""
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_datos_viajes_y_estaciones(processed_dir_path: str):
    """
    Carga todos los archivos Parquet de viajes mensuales y el catálogo de estaciones,
    uniéndolos en un solo DataFrame para los viajes.
    """
    processed_dir = Path(processed_dir_path)
    parquet_files = sorted(processed_dir.glob("trips_*.parquet"))
    
    df_list = []
    for file_path in parquet_files:
        print(f"Cargando: {file_path.name}")
        temp_df = pd.read_parquet(file_path)
        df_list.append(temp_df)
    
    df_trips_all = pd.concat(df_list, ignore_index=True)
    df_stations = pd.read_parquet(processed_dir / "stations.parquet")
    
    print(f"Total de registros combinados: {df_trips_all.shape}")
    print(f"Total de estaciones: {df_stations.shape}")
    
    return df_trips_all, df_stations

def agregar_variables_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas derivadas del tiempo (Año, Mes, Día, Día de la semana) 
    y calcula la duración del viaje en minutos.
    """
    df = df.copy()
    df["Año"] = df["start_time"].dt.year
    df["Mes"] = df["start_time"].dt.month
    df["Dia"] = df["start_time"].dt.day
    df["Dia_semana"] = df["start_time"].dt.weekday
    
    df["duracion_viaje"] = (
        df["end_time"] - df["start_time"]
    ).dt.total_seconds() / 60.0
    
    return df

def limpiar_datos_viajes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra y elimina registros con anomalías temporales (duración negativa o start_time > end_time)
    y elimina filas duplicadas.
    """
    # Filtrar duraciones negativas (tiempos inconsistentes)
    df_limpio = df[df['start_time'] <= df['end_time']].copy()
    
    # Eliminar duplicados si los hay
    longitud_antes = len(df_limpio)
    df_limpio = df_limpio.drop_duplicates().reset_index(drop=True)
    duplicados_eliminados = longitud_antes - len(df_limpio)
    
    print(f"Registros antes de limpieza: {len(df):,}")
    print(f"Duplicados eliminados: {duplicados_eliminados:,}")
    print(f"Registros después de limpieza: {len(df_limpio):,}")
    
    return df_limpio

def merge_viajes_con_estaciones(df_trips: pd.DataFrame, df_stations: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza un merge para enriquecer el DataFrame de viajes con las coordenadas 
    y capacidades tanto de la estación de origen como de la de destino.
    """
    # Merge para estación de origen
    df_enriquecido = df_trips.merge(
        df_stations[['station_id', 'latitude', 'longitude', 'capacity']],
        left_on='origin_station_id',
        right_on='station_id',
        how='left'
    ).rename(columns={
        'latitude': 'origin_latitude',
        'longitude': 'origin_longitude',
        'capacity': 'origin_capacity'
    }).drop(columns=['station_id'])

    # Merge para estación de destino
    df_enriquecido = df_enriquecido.merge(
        df_stations[['station_id', 'latitude', 'longitude', 'capacity']],
        left_on='destination_station_id',
        right_on='station_id',
        how='left'
    ).rename(columns={
        'latitude': 'destination_latitude',
        'longitude': 'destination_longitude',
        'capacity': 'destination_capacity'
    }).drop(columns=['station_id'])

    return df_enriquecido


def run(processed_dir_path: str = "../data/processed", output_file_path: str = "../data/processed/viajes_limpios_enriquecidos.parquet") -> pd.DataFrame:
    """
    Función principal que ejecuta todo el pipeline de datos:
    1. Carga
    2. Adición de variables temporales
    3. Limpieza
    4. Enriquecimiento (Merge con estaciones)
    5. Guardado automático en Parquet
    """
    
    # 1. Cargar datos
    df_trips_all, df_stations = cargar_datos_viajes_y_estaciones(processed_dir_path)
    
    # 2. Agregar variables temporales
    df_trips_temp = agregar_variables_temporales(df_trips_all)
    
    # 3. Limpiar datos
    df_trips_limpio = limpiar_datos_viajes(df_trips_temp)
    
    # 4. Enriquecer con estaciones
    df_final = merge_viajes_con_estaciones(df_trips_limpio, df_stations)
    
    # 5. Guardar automáticamente el archivo Parquet
    if output_file_path:
        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(output_path, index=False)
        print(f"\n¡Archivo procesado guardado exitosamente en: {output_path}")
    
    return df_final


if __name__ == "__main__":
    df_viajes_procesados = run(
        processed_dir_path="../data/processed", 
        output_file_path="../data/processed/viajes_limpios_enriquecidos.parquet"
    )