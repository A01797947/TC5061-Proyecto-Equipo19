"""Calidad, casos límite y reproducibilidad del código de P2/P3 sin MLflow ni CSV grandes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from quality.p23_validation import (
    demo_p2_tables, demo_p3_tables, validate_p2_table, validate_p3_table,
)
from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.features.p2 import (
    P2_FEATURE_COLUMNS, estimar_ocupacion, tasa_historica_riesgo,
)
from tc5061_proyecto_equipo19.features.p3 import construir_features_estaciones
from tc5061_proyecto_equipo19.modeling.p2_pipeline import build_estimator
from tc5061_proyecto_equipo19.modeling.p3_pipeline import build_pipeline


def test_p2_ocupacion_se_acota_y_reinicia_en_cada_dia():
    net = np.array([[[0.6, -1.4, 0.25], [-0.6, 1.4, -0.25]]])
    actual = estimar_ocupacion(net, initial_occupancy=0.5)
    np.testing.assert_allclose(actual[0, 0], [1.0, 0.0, 0.25])
    np.testing.assert_allclose(actual[0, 1], [0.0, 1.0, 0.75])


def test_p2_tasa_historica_no_usa_datos_futuros():
    train = pd.DataFrame({"station_id": ["1", "1", "1"],
                          "fin_de_semana": [0, 0, 0], "hora": [10, 10, 11],
                          "y_riesgo": [0, 1, 0]})
    rates = tasa_historica_riesgo(train)
    assert rates.loc[rates["hora"] == 10, "tasa_hist_riesgo"].iat[0] == 0.5
    assert rates.loc[rates["hora"] == 11, "tasa_hist_riesgo"].iat[0] == 0.0


def test_p3_perfil_cuenta_viajes_y_descarta_estacion_sin_coordenadas():
    times = pd.to_datetime(["2025-01-06 08:00", "2025-01-06 09:00", "2025-01-07 08:00"])
    trips = pd.DataFrame({"start_time": times, "end_time": times + pd.Timedelta(minutes=15),
                          "origin_station_id": ["1", "1", "2"],
                          "destination_station_id": ["2", "2", "1"]})
    stations = pd.DataFrame({"station_id": ["1", "2", "3"],
                             "latitude": [20.6, 20.7, None],
                             "longitude": [-103.3, -103.4, None],
                             "capacity": [20, 15, 10]})
    result = construir_features_estaciones(trips, stations).set_index("station_id")
    assert len(result) == 2
    assert result.loc["1", "viajes_iniciados"] == 2
    assert result.loc["1", "viajes_finalizados"] == 1
    assert result.loc["1", "total_viajes"] == 3
    assert result.loc["1", "pct_salidas_hora_08"] == 0.5


def test_p3_sin_estaciones_validas_rechaza_datos():
    times = pd.to_datetime(["2025-01-06 08:00"])
    trips = pd.DataFrame({"start_time": times, "end_time": times,
                          "origin_station_id": ["1"], "destination_station_id": ["1"]})
    stations = pd.DataFrame({"station_id": ["1"], "latitude": [None],
                             "longitude": [-103.3], "capacity": [10]})
    with pytest.raises(ValueError, match="No hay estaciones"):
        construir_features_estaciones(trips, stations)


def test_p2_validador_acepta_nulos_historicos_y_rechaza_duplicados():
    train, _ = demo_p2_tables()
    train.loc[0, "neto_ventana_semana_pasada"] = np.nan
    assert validate_p2_table(train, "train")["schema"] == "OK"
    duplicated = pd.concat([train, train.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicadas"):
        validate_p2_table(duplicated, "train")


@pytest.mark.parametrize("issue,error", [
    ("invalid_label", "binaria"),
    ("bad_occupancy", "ocupacion_est_h"),
    ("inconsistent_label", "inconsistente"),
    ("missing_column", "faltan columnas"),
])
def test_p2_validador_detecta_datos_invalidos(issue, error):
    train, _ = demo_p2_tables()
    if issue == "invalid_label":
        train.loc[0, "y_riesgo"] = 3
    elif issue == "bad_occupancy":
        train.loc[0, "ocupacion_est_h"] = 1.2
    elif issue == "inconsistent_label":
        train.loc[0, "y_riesgo"] = 1 - train.loc[0, "y_riesgo"]
    else:
        train = train.drop(columns=P2_FEATURE_COLUMNS[0])
    with pytest.raises(ValueError, match=error):
        validate_p2_table(train, "train")


def test_p3_validador_acepta_perfiles_y_rechaza_estaciones_duplicadas():
    train, _ = demo_p3_tables()
    assert validate_p3_table(train, "train")["schema"] == "OK"
    duplicated = pd.concat([train, train.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicadas"):
        validate_p3_table(duplicated, "train")


@pytest.mark.parametrize("issue,error", [
    ("missing", "faltantes"),
    ("inconsistent_total", "total_viajes"),
    ("pct", "proporciones"),
])
def test_p3_validador_detecta_inconsistencias(issue, error):
    train, _ = demo_p3_tables()
    if issue == "missing":
        train.loc[0, "capacity"] = None
    elif issue == "inconsistent_total":
        train.loc[0, "total_viajes"] *= 2
    else:
        train.loc[0, "pct_salidas_hora_08"] = 1.5
    with pytest.raises(ValueError, match=error):
        validate_p3_table(train, "train")


def test_p2_logreg_misma_semilla_mismas_predicciones_y_accuracy():
    """Se usa build_estimator importado por train_p2.py, sin iniciar MLflow."""
    train, _ = demo_p2_tables()
    X = train[P2_FEATURE_COLUMNS]
    y = train["y_riesgo"]
    predictions = []
    for _ in range(2):
        estimator = build_estimator("logreg", {"max_iter": 500})
        estimator.fit(X.iloc[:180], y.iloc[:180])
        predictions.append(estimator.predict(X.iloc[180:]))
    np.testing.assert_array_equal(*predictions)
    assert accuracy_score(y.iloc[180:], predictions[0]) == accuracy_score(y.iloc[180:], predictions[1])
    assert cfg.RANDOM_SEED == build_estimator("logreg")[-1].random_state


def test_p3_kmeans_misma_semilla_mismos_clusters_y_centros():
    """Se usa build_pipeline importado por train_P3.py, sin iniciar MLflow."""
    train, _ = demo_p3_tables()
    cols = ["viajes_iniciados", "viajes_finalizados", "total_viajes"]
    runs = []
    for _ in range(2):
        model = build_pipeline("baseline", cols, 3, 10, 100, cfg.RANDOM_SEED)
        labels = model.fit_predict(train[cols])
        runs.append((labels, model.named_steps["kmeans"].cluster_centers_.copy()))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    np.testing.assert_array_equal(runs[0][1], runs[1][1])
