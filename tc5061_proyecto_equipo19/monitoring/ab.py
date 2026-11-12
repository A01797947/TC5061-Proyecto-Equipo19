"""Reentrenamiento P1 y comparación A/B offline por parejas en MLflow Registry.

Importante: ambos modelos ven EXACTAMENTE las mismas filas de 2026-09.
La prueba es histórica, NO un despliegue A/B online aleatorizado.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.modeling.p1_pipeline import build_pipeline
from tc5061_proyecto_equipo19.monitoring.data import (
    CANDIDATE_RECENT_MONTHS, EVALUATION_MONTHS, HISTORICAL_TRAIN_MONTHS,
    dataframe_to_xy, data_fingerprint, load_p1_months,
)

MODEL_NAME = "mibici-equipo19-p1-demanda"  # Mismo nombre que register_models.MODEL_NAMES["p1"]
CANDIDATE_ALIAS = "candidate_e8"


def _client():
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    return MlflowClient(tracking_uri=cfg.MLFLOW_TRACKING_URI)


def train_candidate(out: Path, alpha: float = 10.0, processed_dir: Path | None = None) -> dict:
    """Entrena Ridge histórico + Jul/Ago 2026; registra *nueva* versión P1.

    Una nueva versión sólo se crea al concluir el fit y validar la carga.
    No se mueve el alias champion bajo ninguna circunstancia.
    """
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha debe ser positivo y finito")
    import mlflow

    from tc5061_proyecto_equipo19.modeling.mlflow_utils import (
        configurar_experimento, iniciar_run, log_modelo_sklearn,
    )

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    months = HISTORICAL_TRAIN_MONTHS + CANDIDATE_RECENT_MONTHS
    if set(months) & set(EVALUATION_MONTHS):
        raise ValueError("Fuga de información: entrenamiento incluye meses de prueba")
    client = _client()
    baseline = client.get_model_version_by_alias(MODEL_NAME, "champion")
    original_champion = str(baseline.version)
    # Falla antes de entrenar si el Registry existente no tiene artefactos accesibles.
    mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{original_champion}")
    frame = load_p1_months(months, processed_dir)
    X, y = dataframe_to_xy(frame)
    if len(X) < 100:
        raise ValueError("Se requieren al menos 100 registros para la versión alternativa")
    pipeline = build_pipeline(Ridge(alpha=alpha))
    pipeline.fit(X, y)
    assert np.isfinite(pipeline.predict(X.head(5))).all(), "Predicciones no finitas"
    fingerprint = data_fingerprint(months, processed_dir)
    metadata = {
        "question": "p1", "model_name": MODEL_NAME, "candidate_model_family": "ridge",
        "ridge_alpha": alpha, "train_months": list(months),
        "excluded_holdout_months": list(EVALUATION_MONTHS),
        "reference_champion_version": original_champion,
        "training_rows": len(X), "new_data_version": fingerprint["version"],
        "files": fingerprint["sources"], "seed": cfg.RANDOM_SEED,
    }
    meta_path = out / "entrenamiento_candidato.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    configurar_experimento()
    with iniciar_run(
        "P1-entregable8-ridge-retraining", pregunta="P1", tipo="candidate_e8",
        params={"e8_alpha": alpha, "e8_train_rows": len(X),
                "e8_new_data_version": fingerprint["version"],
                "e8_holdout": ",".join(EVALUATION_MONTHS)},
    ) as run:
        mlflow.set_tags({
            "question": "p1", "model_family": "ridge", "stage_of_work": "candidate_e8",
            "e8_holdout": ",".join(EVALUATION_MONTHS),
            "e8_training_data_version": fingerprint["version"],
        })
        mlflow.log_metric("e8_train_mae", float(mean_absolute_error(y, pipeline.predict(X))))
        mlflow.log_artifact(str(meta_path), artifact_path="entregable8")
        info = log_modelo_sklearn(pipeline, X.head(5))
        candidate_uri = info.model_uri
        run_id = run.info.run_id
    version = mlflow.register_model(model_uri=candidate_uri, name=MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, str(version.version), "entregable", "8")
    client.set_model_version_tag(MODEL_NAME, str(version.version), "question", "p1")
    client.set_model_version_tag(MODEL_NAME, str(version.version), "data_version_e8", fingerprint["version"])
    client.set_model_version_tag(MODEL_NAME, str(version.version), "holdout", ",".join(EVALUATION_MONTHS))
    client.update_model_version(
        MODEL_NAME, version.version,
        description=(f"Entregable 8: Ridge alpha={alpha}, histórico 2024-01..2025-06 "
                     "y 2026-07..08. Septiembre 2026 reservado para A/B offline. "
                     f"Run {run_id}."),
    )
    # Verificar que el candidato se puede recuperar *desde Registry*.
    mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version.version}")
    client.set_registered_model_alias(MODEL_NAME, CANDIDATE_ALIAS, str(version.version))
    current_champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    if str(current_champion.version) != original_champion:
        raise RuntimeError("El alias champion cambió concurrentemente: revisar Registry")
    result = {
        "model_name": MODEL_NAME, "base_version": original_champion,
        "candidate_version": str(version.version), "candidate_alias": CANDIDATE_ALIAS,
        "run_id": run_id, "train_rows": len(X), "train_data_version": fingerprint["version"],
        "holdout": list(EVALUATION_MONTHS), "champion_unchanged": True,
    }
    (out / "registro_candidato.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def _regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def bootstrap_paired_days(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                          dates: pd.Series, reps: int = 2000, seed: int = 42) -> dict:
    """Bootstrap *pareado por día*, unidad de cluster para errores por estación.

    Delta = MAE(B)-MAE(A); CI95<0 apoya a B. Evita tratar estaciones de un
    mismo día como independientes y nunca reentrena con el conjunto evaluado.
    """
    if reps < 100:
        raise ValueError("Se recomiendan >=100 réplicas bootstrap")
    unique_days, inverse = np.unique(np.asarray(dates, dtype="datetime64[D]"), return_inverse=True)
    if len(unique_days) < 5:
        raise ValueError("A/B requiere al menos cinco días distintos")
    diff = np.abs(y - pred_b) - np.abs(y - pred_a)
    sums = np.bincount(inverse, weights=diff, minlength=len(unique_days))
    counts = np.bincount(inverse, minlength=len(unique_days))
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, len(unique_days), size=len(unique_days))
        diffs[i] = sums[idx].sum() / counts[idx].sum()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "delta_mae": float(diff.mean()), "ci95_lower": float(lo),
        "ci95_upper": float(hi), "n_days": len(unique_days),
        "replicas": reps, "bootstrap_seed": seed,
    }


def compare_versions(holdout: pd.DataFrame, base_version: str, candidate_version: str,
                     out: Path, reps: int = 2000, seed: int = 42,
                     min_relative_improvement: float = 0.02) -> dict:
    """Evalúa versiones del MISMO nombre de Registry; escribe predicciones y gráficos."""
    import mlflow

    if not 0 <= min_relative_improvement < 1:
        raise ValueError("min_relative_improvement debe estar en [0,1)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    client = _client()
    if base_version == candidate_version:
        raise ValueError("Versiones A y B deben ser diferentes")
    for version in (base_version, candidate_version):
        client.get_model_version(MODEL_NAME, version)
    X, y_series = dataframe_to_xy(holdout)
    y = y_series.to_numpy(dtype="float64")
    a = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{base_version}")
    b = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{candidate_version}")
    pa = np.asarray(a.predict(X), dtype="float64").reshape(-1)
    pb = np.asarray(b.predict(X), dtype="float64").reshape(-1)
    if len(pa) != len(y) or len(pb) != len(y) or not np.isfinite(pa).all() or not np.isfinite(pb).all():
        raise ValueError("Modelo devolvió predicciones inválidas o con distinta longitud")
    a_m = _regression_metrics(y, pa)
    b_m = _regression_metrics(y, pb)
    dates = pd.to_datetime({"year": X["Año"], "month": X["Mes"], "day": X["Dia"]})
    bootstrap = bootstrap_paired_days(y, pa, pb, dates, reps=reps, seed=seed)
    improvement = (a_m["mae"] - b_m["mae"]) / a_m["mae"] if a_m["mae"] > 0 else 0.0
    sufficient_evidence = (
        bootstrap["ci95_upper"] < 0
        and improvement >= min_relative_improvement
        and b_m["rmse"] <= a_m["rmse"]
    )
    recommendation = "candidate" if sufficient_evidence else "champion"
    result = {
        "test_design": "A/B offline retrospectivo, misma muestra pareada, sin asignación online",
        "model_name": MODEL_NAME, "base_version": str(base_version),
        "candidate_version": str(candidate_version), "n_rows": len(y),
        "period": sorted(set(dates.dt.strftime("%Y_%m"))),
        "baseline": a_m, "candidate": b_m, "bootstrap": bootstrap,
        "relative_mae_improvement": improvement,
        "min_relative_improvement": min_relative_improvement,
        "recommendation": recommendation,
        "reason": (
            "Mejora en MAE >= umbral, IC95% de diferencia de MAE por día < 0 "
            "y RMSE sin degradación." if sufficient_evidence else
            "No se demostró mejora robusta suficiente. Mantener champion actual."
        ),
        "champion_modified": False,
    }
    predictions = holdout.copy()
    predictions["pred_baseline"] = pa
    predictions["pred_candidate"] = pb
    predictions["abs_error_baseline"] = abs(y - pa)
    predictions["abs_error_candidate"] = abs(y - pb)
    predictions.to_csv(out / "predicciones_ab.csv", index=False)
    (out / "resultado_ab.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {"version": f"A: v{base_version}", **a_m},
        {"version": f"B: v{candidate_version}", **b_m},
    ]).to_csv(out / "metricas_ab.csv", index=False)
    _plots(a_m, b_m, bootstrap, out, base_version, candidate_version)
    return result


def _plots(a: dict, b: dict, boot: dict, out: Path, base: str, candidate: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in zip(axes, ("mae", "rmse"), ("MAE (menor mejor)", "RMSE (menor mejor)")):
        ax.bar([f"v{base} (A)", f"v{candidate} (B)"], [a[key], b[key]])
        ax.set_ylabel(label)
        ax.set_title(f"A/B offline — {key.upper()}")
        for i, value in enumerate((a[key], b[key])):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")
        ax.set_ylim(0, max(a[key], b[key], 0.01) * 1.23)
    fig.tight_layout()
    fig.savefig(out / "comparacion_ab.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.errorbar([boot["delta_mae"]], [0],
                xerr=[[boot["delta_mae"] - boot["ci95_lower"]],
                      [boot["ci95_upper"] - boot["delta_mae"]]],
                fmt="o", capsize=5)
    ax.axvline(0, linestyle="--", color="black")
    ax.set_yticks([])
    ax.set_xlabel("Delta MAE (B - A), IC95% por día; valores negativos favorecen B")
    ax.set_title("Incertidumbre de la diferencia de error")
    fig.tight_layout()
    fig.savefig(out / "bootstrap_ab.png", dpi=170)
    plt.close(fig)


def compare_registered_holdout(out: Path, base_version: str | None = None,
                               candidate_version: str | None = None,
                               processed_dir: Path | None = None) -> dict:
    client = _client()
    base = base_version or str(client.get_model_version_by_alias(MODEL_NAME, "champion").version)
    cand = candidate_version or str(client.get_model_version_by_alias(MODEL_NAME, CANDIDATE_ALIAS).version)
    evaluation = load_p1_months(EVALUATION_MONTHS, processed_dir)
    result = compare_versions(evaluation, base, cand, out)
    result["holdout_data_version"] = data_fingerprint(EVALUATION_MONTHS, processed_dir)["version"]
    # Guardar comparación como corrida MLflow sin subir observaciones individuales.
    import mlflow
    from tc5061_proyecto_equipo19.modeling.mlflow_utils import iniciar_run
    metrics = {
        "e8_a_mae": result["baseline"]["mae"],
        "e8_a_rmse": result["baseline"]["rmse"],
        "e8_a_r2": result["baseline"]["r2"],
        "e8_b_mae": result["candidate"]["mae"],
        "e8_b_rmse": result["candidate"]["rmse"],
        "e8_b_r2": result["candidate"]["r2"],
        "e8_delta_mae": result["bootstrap"]["delta_mae"],
        "e8_ci95_lower": result["bootstrap"]["ci95_lower"],
        "e8_ci95_upper": result["bootstrap"]["ci95_upper"],
    }
    with iniciar_run(
        "P1-entregable8-AB-offline", pregunta="P1", tipo="ab_e8",
        params={"version_a": base, "version_b": cand,
                "holdout": ",".join(EVALUATION_MONTHS),
                "holdout_data_version": result["holdout_data_version"],
                "ab_n_rows": result["n_rows"]},
    ) as run:
        mlflow.set_tags({"question": "p1", "evaluation": "ab_offline_e8",
                         "recommendation": result["recommendation"]})
        mlflow.log_metrics(metrics)
        for filename in ("metricas_ab.csv", "comparacion_ab.png", "bootstrap_ab.png"):
            mlflow.log_artifact(str(Path(out) / filename), artifact_path="entregable8/ab")
        result["mlflow_ab_run_id"] = run.info.run_id
    (Path(out) / "resultado_ab.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def promote_with_approval(out: Path, *, approve: bool) -> str:
    """Promoción consciente: exige resultado A/B verificable y aprobación explícita."""
    if not approve:
        raise ValueError("Se requiere --approve para cambiar el alias champion")
    outcome = json.loads((Path(out) / "resultado_ab.json").read_text(encoding="utf-8"))
    if outcome.get("recommendation") != "candidate":
        raise ValueError("El resultado A/B NO recomienda al candidato; no se promocionará")
    client = _client()
    current = str(client.get_model_version_by_alias(MODEL_NAME, "champion").version)
    if current != str(outcome["base_version"]):
        raise RuntimeError("El champion cambió después del A/B; reevalúa antes de promover")
    candidate = str(outcome["candidate_version"])
    version = client.get_model_version(MODEL_NAME, candidate)
    if version.tags.get("entregable") != "8":
        raise ValueError("Versión candidata sin linaje Entregable 8")
    client.set_registered_model_alias(MODEL_NAME, "champion", candidate)
    return candidate
