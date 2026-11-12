"""Orquestador para descargar los meses recientes necesarios para el A/B testing."""

from tc5061_proyecto_equipo19.monitoring import config_monitoring as cfg_mon
from tc5061_proyecto_equipo19.dataset import run_ingestion

def ejecutar_descarga():
    print("Iniciando descarga aditiva para monitoreo...")
    
    # Extraemos el primer y último mes de la tupla (ej. "2026_07" -> "2026-07")
    mes_inicio = cfg_mon.CURRENT_MONTHS[0].replace("_", "-")
    mes_fin = cfg_mon.CURRENT_MONTHS[-1].replace("_", "-")
    
    print(f"Rango a procesar: {mes_inicio} hasta {mes_fin}")
    
    # run_ingestion descarga y procesa SOLO este rango sin borrar lo anterior
    manifiesto_parcial = run_ingestion(start=mes_inicio, end=mes_fin)
    
    print(f"¡Descarga aditiva completada! Manifiesto guardado en: {manifiesto_parcial}")

if __name__ == "__main__":
    ejecutar_descarga()