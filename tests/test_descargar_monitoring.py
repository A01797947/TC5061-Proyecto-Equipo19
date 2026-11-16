"""El orquestador de descarga usa CURRENT_MONTHS y acepta un rango explícito."""

from tc5061_proyecto_equipo19.monitoring import descargar_2026
from tc5061_proyecto_equipo19.monitoring.data import CURRENT_MONTHS


def test_descarga_usa_current_months_por_defecto(monkeypatch):
    llamadas = []
    monkeypatch.setattr(
        descargar_2026, "run_ingestion",
        lambda start, end: llamadas.append((start, end)) or "manifiesto.json",
    )
    descargar_2026.main([])
    assert llamadas == [
        (min(CURRENT_MONTHS).replace("_", "-"), max(CURRENT_MONTHS).replace("_", "-"))
    ]


def test_descarga_acepta_rango_explicito(monkeypatch):
    llamadas = []
    monkeypatch.setattr(
        descargar_2026, "run_ingestion",
        lambda start, end: llamadas.append((start, end)) or "manifiesto.json",
    )
    descargar_2026.main(["--inicio", "2026-08", "--fin", "2026-09"])
    assert llamadas == [("2026-08", "2026-09")]
