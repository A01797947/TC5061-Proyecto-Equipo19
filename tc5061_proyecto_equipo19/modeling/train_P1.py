"""P1: predicción de demanda diaria por estación con MLflow."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.features.p1 import construir_demanda_diaria
from tc5061_proyecto_equipo19.modeling.mlflow_utils import configurar_experimento, iniciar_run, log_modelo_sklearn
from tc5061_proyecto_equipo19.modeling.artifacts import log_run_config


def load_data(file_path: Path | None = None) -> pd.DataFrame:
    """Carga el dataset interim que ya contiene las variables temporales."""
    ruta = file_path if file_path else cfg.INTERIM_DATA_DIR / 'viajes_limpios_enriquecidos.parquet'
    print(f"Cargando datos desde: {ruta}")
    return pd.read_parquet(ruta)


def build_pipeline(model) -> Pipeline:
    """Preprocesamiento categórico/numérico y regresor."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), ["origin_station_id"]),
            ("num", "passthrough", ["Año", "Mes", "Dia", "Dia_semana"]),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])


def split_temporal(X: pd.DataFrame, y: pd.Series) -> dict:
    """Crea train/valid/test con fechas de config.py, sin mezclar el futuro."""
    dates = pd.to_datetime({
        "year": X["Año"].astype(int),
        "month": X["Mes"].astype(int),
        "day": X["Dia"].astype(int),
    })
    masks = {
        "train": dates <= pd.Timestamp(cfg.FIN_ENTRENAMIENTO),
        "valid": (dates >= pd.Timestamp(cfg.INICIO_VALIDACION)) & (dates <= pd.Timestamp(cfg.FIN_VALIDACION)),
        "test": dates >= pd.Timestamp(cfg.INICIO_PRUEBA),
    }
    result = {}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(
                f"La partición temporal '{name}' quedó vacía. Verifica el rango de fechas y los datos disponibles."
            )
        result[name] = (X.loc[mask].copy(), y.loc[mask].copy())
    
    return result


def generate_artifact(y_true, y_pred, model_name: str, artifact_path: Path) -> None:
    """Guarda la gráfica de valores reales frente a predichos."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, edgecolors="none")
    min_val = min(float(y_true.min()), float(y_pred.min()))
    max_val = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Predicción ideal")
    ax.set(title=f"Valores reales vs. predichos ({model_name})", xlabel="Viajes reales", ylabel="Viajes predichos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifact_path, dpi=150)
    plt.close(fig)


def preparar_datos(file_path: Path | None = None):
    """Prepara una vez los mismos cortes temporales para comparar hiperparámetros."""
    df = load_data(file_path)
    X, y, _ = construir_demanda_diaria(df)
    del df
    return split_temporal(X, y)


def train_model(model: str = "linear", params: dict | None = None, nested: bool = False,
                stage_of_work: str | None = None, prepared_splits=None):
    """Entrena P1, evalúa validación/prueba y registra modelo y artefactos en MLflow.

    `nested=True` permite usar esta función como corrida hija de una búsqueda.
    """
    params = params or {}
    alpha = float(params.get("alpha", cfg.P1_RIDGE_ALPHA))
    data_path = params.get("filepath")
    splits = prepared_splits if prepared_splits is not None else preparar_datos(data_path)
    X_train, y_train = splits["train"]
    X_valid, y_valid = splits["valid"]
    X_test, y_test = splits["test"]

    if model == "linear":
        estimator = LinearRegression()
        default_stage = "baseline"
    elif model == "ridge":
        estimator = Ridge(alpha=alpha)
        default_stage = "candidate"
    elif model == "rf":
        estimator = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", cfg.P1_RF_N_ESTIMATORS)),
            max_depth=int(params.get("max_depth", cfg.P1_RF_MAX_DEPTH)),
            random_state=cfg.RANDOM_SEED, n_jobs=-1,
        )
        default_stage = "candidate"
    else:
        raise ValueError(f"Modelo P1 no admitido: {model}. Usa 'linear', 'ridge' o 'rf'.")

    pipeline = build_pipeline(estimator)
    pipeline.fit(X_train, y_train)
    metrics = {}
    
    for split_name, x_part, y_part in [("valid", X_valid, y_valid), ("test", X_test, y_test)]:
        pred = pipeline.predict(x_part)
        metrics[f"{split_name}_mse"] = float(mean_squared_error(y_part, pred))
        metrics[f"{split_name}_mae"] = float(mean_absolute_error(y_part, pred))
        metrics[f"{split_name}_r2"] = float(r2_score(y_part, pred))

    configurar_experimento()
    stage = stage_of_work or default_stage
    run_name = f"P1-{model}" + (f"-alpha{alpha:g}" if model == "ridge" else "")
    if model == "rf":
        run_name += f"-trees{estimator.n_estimators}-depth{estimator.max_depth}"
    run_params = {
        "model": model,
        "alpha": alpha if model == "ridge" else "not_applicable",
        "n_estimators": estimator.n_estimators if model == "rf" else "not_applicable",
        "max_depth": estimator.max_depth if model == "rf" else "not_applicable",
        "n_train": len(X_train), "n_valid": len(X_valid), "n_test": len(X_test),
        "random_state": cfg.RANDOM_SEED, "split_strategy": "temporal",
        "train_end": cfg.FIN_ENTRENAMIENTO, "validation_start": cfg.INICIO_VALIDACION,
        "validation_end": cfg.FIN_VALIDACION, "test_start": cfg.INICIO_PRUEBA,
        "source_file": str(data_path),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        plot_path = tmp / "real_vs_predicho.png"
        y_pred_test = pipeline.predict(X_test)
        generate_artifact(y_test, y_pred_test, model, plot_path)
        config_path = tmp / "run_config.json"
        config_path.write_text(json.dumps({"question": "p1", **run_params}, indent=2, ensure_ascii=False), encoding="utf-8")

        with iniciar_run(run_name, pregunta="P1", tipo=stage, params=run_params, nested=nested) as run:
            mlflow.set_tags({"question": "p1", "model_family": model, "stage_of_work": stage, "model": model})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(plot_path), artifact_path="plots")
            mlflow.log_artifact(str(config_path), artifact_path="config")
            # Guardar solo metadatos de entrenamiento, nunca viajes individuales.
            mlflow.log_input(
                mlflow.data.from_pandas(
                    X_train.head(1000).reset_index(drop=True),
                    name="mibici-p1-training-sample",
                ), context="training",
            )
            example = X_train.head(5)
            log_modelo_sklearn(pipeline, example)
            print(f"Run P1: {run.info.run_id}")
            print(f"Métricas: {metrics}")
            return {"run_id": run.info.run_id, "run_name": run_name, **metrics}


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento P1 con MLflow")
    parser.add_argument("--model", choices=["linear", "ridge", "rf"], default=cfg.P1_DEFAULT_MODEL)
    parser.add_argument("--alpha", type=float, default=cfg.P1_RIDGE_ALPHA)
    parser.add_argument("--n-estimators", type=int, default=cfg.P1_RF_N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=cfg.P1_RF_MAX_DEPTH)
    parser.add_argument(
        "--filepath",
        type=Path,
        default=None,
        help=(
            "Ruta opcional a un archivo Parquet individual. "
            "Si se omite, se leen los archivos mensuales de data/processed."
        ),
    )
    args = parser.parse_args()
    return train_model(args.model, {"alpha": args.alpha, "filepath": args.filepath,
                                    "n_estimators": args.n_estimators, "max_depth": args.max_depth})

if __name__ == "__main__":
    main()