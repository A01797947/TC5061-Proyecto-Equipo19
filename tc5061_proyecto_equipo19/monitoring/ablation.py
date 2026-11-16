"""Ablación del A/B: ¿la mejora viene de los datos recientes o del cambio de modelo?

El candidato (B) difiere del champion (A) en dos cosas a la vez: se entrena también con
``CANDIDATE_RECENT_MONTHS`` y usa Ridge con ``alpha``. Para separar ambos efectos se
entrena un modelo intermedio X: **mismo Ridge que B, pero solo con los meses históricos**
(los mismos datos de A). Se evalúa sobre la misma muestra pareada que el A/B.

    A -> X : efecto de cambiar el modelo (datos iguales).
    X -> B : efecto de añadir los datos recientes (modelo igual).

X no se registra en MLflow; es un diagnóstico offline, no un candidato a producción.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from tc5061_proyecto_equipo19.features.p1 import P1_FEATURE_COLUMNS
from tc5061_proyecto_equipo19.modeling.p1_pipeline import build_pipeline
from tc5061_proyecto_equipo19.monitoring.ab import _regression_metrics, bootstrap_paired_days
from tc5061_proyecto_equipo19.monitoring.data import (
    HISTORICAL_TRAIN_MONTHS,
    dataframe_to_xy,
    load_p1_months,
)
from tc5061_proyecto_equipo19.monitoring.segments import load_predictions

ABLATION_COLUMNS = [
    "Año", "Mes", "Dia", "Dia_semana", "origin_station_id", "cantidad_viajes",
    "pred_baseline", "pred_candidate",
]


def decompose(y, pred_a, pred_x, pred_b, dates: pd.Series, reps: int = 2000,
              seed: int = 42) -> dict:
    """Descompone la mejora de MAE entre A y B en efecto-modelo y efecto-datos."""
    y, pred_a, pred_x, pred_b = (
        np.asarray(v, dtype="float64") for v in (y, pred_a, pred_x, pred_b)
    )
    metrics = {
        "A_champion": _regression_metrics(y, pred_a),
        "X_ridge_solo_historico": _regression_metrics(y, pred_x),
        "B_candidato": _regression_metrics(y, pred_b),
    }
    total = metrics["A_champion"]["mae"] - metrics["B_candidato"]["mae"]
    model_effect = metrics["A_champion"]["mae"] - metrics["X_ridge_solo_historico"]["mae"]
    data_effect = metrics["X_ridge_solo_historico"]["mae"] - metrics["B_candidato"]["mae"]
    share_data = data_effect / total if total > 0 else None
    return {
        "metricas": metrics,
        "mejora_mae_total_A_a_B": total,
        "efecto_modelo_A_a_X": model_effect,
        "efecto_datos_recientes_X_a_B": data_effect,
        "proporcion_atribuible_a_datos_recientes": share_data,
        "bootstrap_A_vs_X": bootstrap_paired_days(y, pred_a, pred_x, dates, reps, seed),
        "bootstrap_X_vs_B": bootstrap_paired_days(y, pred_x, pred_b, dates, reps, seed),
    }


def interpretation(result: dict) -> str:
    share = result["proporcion_atribuible_a_datos_recientes"]
    if share is None:
        return "B no mejora a A en MAE; no hay mejora que atribuir."
    return (
        f"Mejora total de MAE (A a B): {result['mejora_mae_total_A_a_B']:.3f}. "
        f"Cambiar de modelo con los mismos datos (A a X): "
        f"{result['efecto_modelo_A_a_X']:+.3f}. "
        f"Añadir datos recientes (X a B): {result['efecto_datos_recientes_X_a_B']:+.3f}, "
        f"es decir {share:.0%} de la mejora total."
    )


def _plot(result: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["A: champion", "X: Ridge solo\nhistórico", "B: candidato"]
    maes = [m["mae"] for m in result["metricas"].values()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, maes, color=["#4C72B0", "#8C8C8C", "#55A868"])
    for position, value in enumerate(maes):
        ax.text(position, value, f"{value:.2f}", ha="center", va="bottom")
    ax.set_ylabel("MAE (menor es mejor)")
    ax.set_title("Ablación: datos recientes vs cambio de modelo")
    ax.set_ylim(0, max(maes) * 1.2)
    fig.tight_layout()
    fig.savefig(Path(out) / "ablacion_ab.png", dpi=170)
    plt.close(fig)


def run_ablation(out: Path, alpha: float = 10.0, processed_dir: Path | None = None,
                 reps: int = 2000, seed: int = 42) -> dict:
    """Entrena X (Ridge solo histórico), lo evalúa con predicciones_ab.csv y escribe resultados."""
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha debe ser positivo y finito")
    out = Path(out)
    predictions = load_predictions(out)
    missing = sorted(set(ABLATION_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError(f"predicciones_ab.csv no tiene las columnas esperadas: {missing}")
    X_train, y_train = dataframe_to_xy(load_p1_months(HISTORICAL_TRAIN_MONTHS, processed_dir))
    pipeline = build_pipeline(Ridge(alpha=alpha))
    pipeline.fit(X_train, y_train)
    X_eval = predictions[P1_FEATURE_COLUMNS].copy()
    X_eval["origin_station_id"] = X_eval["origin_station_id"].astype(str)
    pred_x = np.asarray(pipeline.predict(X_eval), dtype="float64").reshape(-1)
    if not np.isfinite(pred_x).all():
        raise ValueError("El modelo intermedio produjo predicciones no finitas")
    dates = pd.to_datetime(
        {"year": predictions["Año"], "month": predictions["Mes"], "day": predictions["Dia"]}
    )
    result = decompose(
        predictions["cantidad_viajes"], predictions["pred_baseline"], pred_x,
        predictions["pred_candidate"], dates, reps=reps, seed=seed,
    )
    result.update({
        "diseno": "Ridge solo histórico (X) como modelo intermedio; mismos datos de A",
        "ridge_alpha": alpha,
        "meses_entrenamiento_X": list(HISTORICAL_TRAIN_MONTHS),
        "n_filas": len(predictions),
        "nota": "La familia del champion no se verifica aquí; consúltala en el tag "
                "model_family de su corrida en MLflow.",
    })
    result["interpretacion"] = interpretation(result)
    (out / "ablacion_ab.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame([{"modelo": name, **values} for name, values in result["metricas"].items()]
                 ).to_csv(out / "ablacion_ab.csv", index=False)
    _plot(result, out)
    return result
