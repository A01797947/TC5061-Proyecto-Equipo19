"""Análisis por segmentos del A/B offline (sin red ni MLflow)."""

from _monitoring_helpers import predictions_frame, write_month_parquets
import pytest

from tc5061_proyecto_equipo19.monitoring.data import (
    CANDIDATE_RECENT_MONTHS,
    HISTORICAL_TRAIN_MONTHS,
)
from tc5061_proyecto_equipo19.monitoring.segments import (
    SEEN_BY_BOTH,
    SEEN_BY_NEITHER,
    SEEN_ONLY_BY_B,
    compute_segments,
    generate_segments,
    stations_in_months,
)


def test_segmentos_clasifican_estaciones_y_tipo_de_dia():
    frame = predictions_frame()
    segments, by_station = compute_segments(
        frame, stations_a={"1", "2"}, stations_b={"1", "2", "3"}
    )
    seen = segments[segments["segmento"] == "estacion_vista_al_entrenar"].set_index("grupo")
    assert seen.loc[SEEN_BY_BOTH, "n_estaciones"] == 2
    assert seen.loc[SEEN_ONLY_BY_B, "n_estaciones"] == 1
    assert seen.loc[SEEN_BY_NEITHER, "n_estaciones"] == 1
    assert seen["n_filas"].sum() == len(frame)
    days = segments[segments["segmento"] == "tipo_dia"].set_index("grupo")
    assert set(days.index) == {"entre semana", "fin de semana"}
    glob = segments[segments["segmento"] == "global"].iloc[0]
    assert glob["mae_a"] == pytest.approx(frame["abs_error_baseline"].mean())
    assert glob["mejora_mae"] == pytest.approx(glob["mae_a"] - glob["mae_b"])
    assert list(by_station["n_filas"]) == sorted(by_station["n_filas"], reverse=True)
    assert (by_station["mejora_mae"] > 0).all()


def test_generate_segments_escribe_resultados(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    write_month_parquets(processed, HISTORICAL_TRAIN_MONTHS, ["1", "2"])
    write_month_parquets(processed, CANDIDATE_RECENT_MONTHS, ["1", "2", "3"])
    out = tmp_path / "out"
    out.mkdir()
    predictions_frame().to_csv(out / "predicciones_ab.csv", index=False)
    summary = generate_segments(out, processed_dir=processed)
    assert summary["estaciones_nuevas_para_A"] == 2  # la 3 (solo B) y la 4 (ninguno)
    assert summary["estaciones_nuevas_para_A_y_B"] == 1
    for name in ("segmentos_ab.csv", "mae_por_estacion.csv", "segmentos_ab.png",
                 "resumen_segmentos.json"):
        assert (out / name).is_file()


def test_segmentos_falla_sin_predicciones(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_segments(tmp_path)


def test_parquets_faltantes_se_listan_todos(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        stations_in_months(("2026_07", "2026_08"), tmp_path)
    message = str(error.value)
    assert "trips_2026_07.parquet" in message and "trips_2026_08.parquet" in message
    assert "descargar_2026" in message
