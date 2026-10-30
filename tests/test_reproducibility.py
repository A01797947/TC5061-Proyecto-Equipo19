"""Ajusta dos veces el MISMO pipeline de P1 con la semilla oficial y compara salidas."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from tc5061_proyecto_equipo19.config import RANDOM_SEED
from tc5061_proyecto_equipo19.modeling.train_P1 import build_pipeline


def test_rf_p1_misma_semilla_mismas_predicciones_y_mae():
    rng = np.random.default_rng(RANDOM_SEED)
    n = 90
    X = pd.DataFrame({
        "origin_station_id": np.where(np.arange(n) % 2 == 0, "A", "B"),
        "Año": np.full(n, 2025),
        "Mes": np.repeat([1, 2, 3], n // 3),
        "Dia": 1 + np.arange(n) % 28,
        "Dia_semana": np.arange(n) % 7,
    })
    y = pd.Series(7 + 2 * (X.origin_station_id == "A").astype(int)
                  + X.Dia_semana + rng.integers(0, 5, size=n))
    train, test = np.arange(70), np.arange(70, n)
    
    def entrenar(seed):
        modelo = build_pipeline(RandomForestRegressor(
            n_estimators=35, max_depth=6, random_state=seed, n_jobs=1,
        ))
        modelo.fit(X.iloc[train], y.iloc[train])
        return modelo.predict(X.iloc[test])

    pred1 = entrenar(RANDOM_SEED)
    pred2 = entrenar(RANDOM_SEED)
    
    np.testing.assert_array_equal(pred1, pred2)
    assert mean_absolute_error(y.iloc[test], pred1) == mean_absolute_error(y.iloc[test], pred2)
    assert np.isfinite(pred1).all()