"""Ablación del A/B offline: datos recientes vs cambio de modelo (sin red ni MLflow)."""

import json

import numpy as np
import pandas as pd
import pytest
from _monitoring_helpers import predictions_frame, write_month_parquets

from tc5061_proyecto_equipo19.monitoring.ablation import decompose, run_ablation
from tc5061_proyecto_equipo19.monitoring.data import HISTORICAL_TRAIN_MONTHS


def test_descomposicion_suma_el_total():
    frame = predictions_frame()
    dates = pd.to_datetime({"year": frame["Año"], "month": frame["Mes"], "day": frame["Dia"]})
    pred_x = (frame["pred_baseline"] + frame["pred_candidate"]) / 2
    result = decompose(frame["cantidad_viajes"], frame["pred_baseline"], pred_x,
                       frame["pred_candidate"], dates, reps=200)
    assert result["efecto_modelo_A_a_X"] + result["efecto_datos_recientes_X_a_B"] == (
        pytest.approx(result["mejora_mae_total_A_a_B"])
    )
    assert 0 < result["proporcion_atribuible_a_datos_recientes"] < 1


def test_run_ablation_escribe_resultados(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    write_month_parquets(processed, HISTORICAL_TRAIN_MONTHS, ["1", "2", "3", "4"])
    out = tmp_path / "out"
    out.mkdir()
    predictions_frame().to_csv(out / "predicciones_ab.csv", index=False)
    result = run_ablation(out, alpha=10.0, processed_dir=processed, reps=200)
    saved = json.loads((out / "ablacion_ab.json").read_text(encoding="utf-8"))
    assert saved["n_filas"] == len(predictions_frame())
    assert np.isfinite(result["metricas"]["X_ridge_solo_historico"]["mae"])
    assert (out / "ablacion_ab.png").is_file()
    assert "interpretacion" in saved


def test_ablacion_rechaza_alpha_invalido(tmp_path):
    with pytest.raises(ValueError):
        run_ablation(tmp_path, alpha=0)
