"""
Entrenamiento y evaluación de P2 con registro en MLflow.

Pregunta: ¿una estación estará en riesgo de quedarse sin bicicletas o sin anclajes en las
próximas tres horas? Cada modelo es un run del experimento ``mibici-equipo19``.

Uso desde la raíz del repositorio::

    python -m tc5061_proyecto_equipo19.modeling.train_p2            # todos los modelos
    python -m tc5061_proyecto_equipo19.modeling.train_p2 --models hgb
"""

import json
from pathlib import Path
import tempfile
from typing import Annotated

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import typer

from tc5061_proyecto_equipo19 import config_p2 as cfg
from tc5061_proyecto_equipo19.config import (
    FECHA_FIN,
    FECHA_INICIO,
    MAX_FILAS_ENTRENAMIENTO,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from tc5061_proyecto_equipo19.features import dividir_particiones
from tc5061_proyecto_equipo19.features.p2 import (
    P2_FEATURE_COLUMNS,
    build_p2_features,
    tasa_historica_riesgo,
)
from tc5061_proyecto_equipo19.modeling.mlflow_utils import configurar_experimento, iniciar_run

app = typer.Typer()

TARGET = "y_riesgo"

SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]

MODEL_INFO = {
    "tasa_historica": {
        "run_name": "P2-baseline-tasa-historica",
        "tipo": "baseline",
        "descripcion": "Proporción de horas en riesgo de la estación a esa hora y tipo de día "
        "en entrenamiento",
    },
    "regla_ocupacion": {
        "run_name": "P2-baseline-regla-ocupacion",
        "tipo": "baseline",
        "descripcion": "Distancia de la ocupación estimada actual al umbral más cercano "
        "(persistencia del estado)",
    },
    "logreg": {
        "run_name": "P2-candidato-logreg-balanceada",
        "tipo": "candidato",
        "descripcion": "Regresión logística con class_weight='balanced' y variables escaladas",
    },
    "hgb": {
        "run_name": "P2-candidato-hist-gradient-boosting",
        "tipo": "candidato",
        "descripcion": "HistGradientBoostingClassifier sobre todas las variables de P2",
    },
}


# -----------------------------------------------------------------------------
# Datos y partición
# -----------------------------------------------------------------------------

def add_baseline_scores(splits: dict[str, pd.DataFrame]) -> None:
    """Agrega la tasa histórica (ajustada solo con entrenamiento) a cada conjunto."""
    rates = tasa_historica_riesgo(splits["train"], TARGET)
    global_rate = splits["train"][TARGET].mean()
    for name, part in splits.items():
        merged = part[["station_id", "fin_de_semana", "hora"]].merge(
            rates, on=["station_id", "fin_de_semana", "hora"], how="left"
        )
        splits[name] = part.assign(
            tasa_hist_riesgo=merged["tasa_hist_riesgo"].fillna(global_rate).to_numpy()
        )


def occupancy_rule_score(part: pd.DataFrame) -> np.ndarray:
    """
    Puntaje de la regla de ocupación: qué tan cerca está la ocupación estimada actual de
    quedar vacía o llena. Vale 1 en el umbral o más allá y 0 a media capacidad.
    """
    occ = part["ocupacion_est_h"].to_numpy()
    gap_empty = (occ - cfg.EMPTY_THRESHOLD) / (cfg.INITIAL_OCCUPANCY - cfg.EMPTY_THRESHOLD)
    gap_full = (cfg.FULL_THRESHOLD - occ) / (cfg.FULL_THRESHOLD - cfg.INITIAL_OCCUPANCY)
    return np.clip(1 - np.minimum(gap_empty, gap_full), 0, 1)


# -----------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------
def choose_threshold(y: np.ndarray, score: np.ndarray, beta: float = 2.0) -> float:
    """
    Umbral que maximiza F-beta en validación.

    Con beta = 2 el recall pesa más que la precisión: no anticipar una estación vacía o llena
    cuesta más (viajes perdidos, usuarios que no pueden devolver) que mandar un camión de más.
    """
    precision, recall, thresholds = precision_recall_curve(y, score)
    precision, recall = precision[:-1], recall[:-1]
    fbeta = (1 + beta**2) * precision * recall / np.maximum(beta**2 * precision + recall, 1e-12)
    return float(thresholds[np.nanargmax(fbeta)])


def top_k_metrics(part: pd.DataFrame, score: np.ndarray, k: int) -> dict:
    """
    Precisión y recall en el top-k por hora de emisión.

    Refleja la decisión real: cada hora el supervisor manda camión a las k estaciones con
    mayor puntaje. La precisión dice cuántas de esas k estaban de verdad en riesgo y el
    recall cuántas estaciones en riesgo quedaron cubiertas.
    """
    ranked = pd.DataFrame({"ts": part["timestamp"].to_numpy(), "y": part[TARGET].to_numpy()})
    ranked["score"] = score
    ranked["rank"] = ranked.groupby("ts")["score"].rank(method="first", ascending=False)
    top = ranked[ranked["rank"] <= k]
    per_hour = top.groupby("ts")["y"].sum() / k
    return {
        f"precision_at_{k}": float(per_hour.mean()),
        f"recall_at_{k}": float(top["y"].sum() / max(ranked["y"].sum(), 1)),
    }


def evaluate(part: pd.DataFrame, score: np.ndarray, threshold: float, prefix: str) -> dict:
    """Métricas de clasificación de un conjunto con el umbral elegido en validación."""
    y = part[TARGET].to_numpy()
    pred = (score >= threshold).astype(int)
    prob = np.clip(score, 1e-6, 1 - 1e-6)
    metrics = {
        "average_precision": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "brier": brier_score_loss(y, prob),
        "log_loss": log_loss(y, prob, labels=[0, 1]),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "recall_vacia": recall_score(part["y_riesgo_vacia"], pred, zero_division=0),
        "recall_llena": recall_score(part["y_riesgo_llena"], pred, zero_division=0),
        "prevalencia": float(y.mean()),
        "ap_referencia_azar": float(y.mean()),
    }
    metrics.update(top_k_metrics(part, score, cfg.TOP_K))
    return {f"{prefix}_{name}": float(value) for name, value in metrics.items()}


# -----------------------------------------------------------------------------
# Gráficas
# -----------------------------------------------------------------------------
def plot_pr_curves(scores: dict, splits: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["valid", "test"]:
        PrecisionRecallDisplay.from_predictions(
            splits[name][TARGET], scores[name], ax=ax, name=name, plot_chance_level=name == "test"
        )
    ax.set_title(f"Curva precisión-recall · {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(part: pd.DataFrame, score: np.ndarray, threshold: float, path: Path) -> None:
    cm = confusion_matrix(part[TARGET], (score >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["sin riesgo", "en riesgo"]).plot(
        ax=ax, values_format=",", colorbar=False
    )
    ax.set_title(f"Matriz de confusión (prueba, umbral {threshold:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_calibration(part: pd.DataFrame, score: np.ndarray, path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(part[TARGET], score, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="calibración perfecta")
    ax.plot(mean_pred, frac_pos, marker="o", label="modelo")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Proporción observada en riesgo")
    ax.set_title("Calibración (prueba)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_topk_by_hour(part: pd.DataFrame, score: np.ndarray, path: Path) -> None:
    k = cfg.TOP_K
    ranked = pd.DataFrame({"ts": part["timestamp"], "hora": part["hora"], "y": part[TARGET]})
    ranked["score"] = score
    ranked["rank"] = ranked.groupby("ts")["score"].rank(method="first", ascending=False)
    top = ranked[ranked["rank"] <= k]
    by_hour = top.groupby(["hora", "ts"])["y"].sum().div(k).groupby("hora").mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_hour.index, by_hour.values, color="#3b6ea5")
    ax.set_xlabel("Hora de emisión")
    ax.set_ylabel(f"Precisión en top-{k}")
    ax.set_ylim(0, 1)
    ax.set_title(f"Precisión en las {k} estaciones de mayor riesgo, por hora (prueba)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(names: list[str], values: np.ndarray, xlabel: str, path: Path) -> None:
    order = np.argsort(np.abs(values))
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(names) + 1.5))
    ax.barh(np.array(names)[order], values[order], color="#3b6ea5")
    ax.set_xlabel(xlabel)
    ax.set_title("Importancia de variables")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Modelos
# -----------------------------------------------------------------------------
def build_estimator(name: str):
    """Estimadores candidatos con hiperparámetros fijos y semilla de config_p2."""
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000),
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=300,
            max_leaf_nodes=63,
            min_samples_leaf=200,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Modelo desconocido: {name}")


def estimator_params(name: str, estimator) -> dict:
    if name == "logreg":
        lr = estimator[-1]
        return {
            "C": lr.C,
            "class_weight": lr.class_weight,
            "max_iter": lr.max_iter,
            "imputer": "median",
            "scaler": "standard",
        }
    keys = [
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "early_stopping",
    ]
    return {key: getattr(estimator, key) for key in keys}


def run_model(name: str, splits: dict, common_params: dict, max_train_rows: int | None) -> dict:
    """Entrena (si aplica), evalúa y registra un modelo como run de MLflow."""
    info = MODEL_INFO[name]
    X = {part: splits[part][P2_FEATURE_COLUMNS] for part in splits}
    estimator = None

    if name == "tasa_historica":
        scores = {part: splits[part]["tasa_hist_riesgo"].to_numpy() for part in splits}
        model_params = {"agrupacion": "estacion x fin_de_semana x hora"}
    elif name == "regla_ocupacion":
        scores = {part: occupancy_rule_score(splits[part]) for part in splits}
        model_params = {"variable": "ocupacion_est_h"}
    else:
        estimator = build_estimator(name)
        train = splits["train"]
        if max_train_rows and len(train) > max_train_rows:
            train = train.sample(max_train_rows, random_state=RANDOM_SEED)
        estimator.fit(train[P2_FEATURE_COLUMNS], train[TARGET])
        scores = {part: estimator.predict_proba(X[part])[:, 1] for part in splits}
        model_params = estimator_params(name, estimator)
        model_params["train_rows_used"] = len(train)

    threshold = choose_threshold(splits["valid"][TARGET].to_numpy(), scores["valid"])
    metrics = {"umbral_elegido_valid": threshold}
    for part in ["train", "valid", "test"]:
        metrics.update(evaluate(splits[part], scores[part], threshold, part))

    params = {
        **common_params,
        **{f"modelo_{k}": v for k, v in model_params.items()},
        "modelo": name,
        "umbral_criterio": "max F2 en validacion",
    }
    # iniciar_run agrega fechas, particiones, semilla, data_version, manifiesto y las
    # etiquetas equipo, pregunta, tipo_modelo y git_commit.
    with iniciar_run(info["run_name"], pregunta="P2", tipo=info["tipo"], params=params) as run:
        mlflow.set_tags(
            {"tarea": "clasificacion", "modelo": name, "mlflow.note.content": info["descripcion"]}
        )
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_pr_curves(scores, splits, info["run_name"], tmp / "curva_precision_recall.png")
            plot_confusion(
                splits["test"], scores["test"], threshold, tmp / "matriz_confusion_test.png"
            )
            plot_calibration(splits["test"], scores["test"], tmp / "calibracion_test.png")
            plot_topk_by_hour(splits["test"], scores["test"], tmp / "precision_topk_por_hora.png")

            if name == "logreg":
                coefs = estimator[-1].coef_[0]
                plot_importance(
                    P2_FEATURE_COLUMNS,
                    coefs,
                    "Coeficiente (variables escaladas)",
                    tmp / "importancia_variables.png",
                )
            elif name == "hgb":
                sample = splits["valid"].sample(
                    min(50_000, len(splits["valid"])), random_state=RANDOM_SEED
                )
                result = permutation_importance(
                    estimator,
                    sample[P2_FEATURE_COLUMNS],
                    sample[TARGET],
                    scoring="average_precision",
                    n_repeats=3,
                    random_state=RANDOM_SEED,
                )
                plot_importance(
                    P2_FEATURE_COLUMNS,
                    result.importances_mean,
                    "Caída de average precision al permutar (validación)",
                    tmp / "importancia_variables.png",
                )

            (tmp / "metricas.json").write_text(json.dumps(metrics, indent=2))
            mlflow.log_artifacts(tmp, artifact_path="evaluacion")

        if estimator is not None:
            example = X["valid"].head(5)
            mlflow.sklearn.log_model(
                estimator,
                name="modelo",
                signature=infer_signature(example, estimator.predict_proba(example)[:, 1]),
                input_example=example,
                # Tipos que el propio equipo serializa; skops pide declararlos como confiables.
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
        logger.success("{}: AP test = {:.3f}", info["run_name"], metrics["test_average_precision"])
        return {"run_id": run.info.run_id, "run_name": info["run_name"], **metrics}


def plot_comparison(results: list[dict], path: Path) -> None:
    names = [r["run_name"].replace("P2-", "") for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    x = np.arange(len(results))
    ax.bar(
        x - width / 2, [r["valid_average_precision"] for r in results], width, label="validación"
    )
    ax.bar(x + width / 2, [r["test_average_precision"] for r in results], width, label="prueba")
    ax.axhline(results[0]["test_prevalencia"], ls="--", color="gray", label="azar (prevalencia)")
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylabel("Average precision")
    ax.set_title("P2 · comparación de modelos")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@app.command()
def main(
    features_path: Path = cfg.FEATURES_PATH,
    rebuild: Annotated[
        bool, typer.Option(help="Reconstruye la tabla de P2 desde los Parquet.")
    ] = False,
    models: Annotated[list[str] | None, typer.Option(help="Modelos a correr.")] = None,
    max_train_rows: Annotated[
        int, typer.Option(help="Muestra de entrenamiento para candidatos; 0 = todo.")
    ] = MAX_FILAS_ENTRENAMIENTO,
):
    """Construye (si hace falta) la tabla de P2, entrena los modelos y los registra en MLflow."""
    np.random.seed(RANDOM_SEED)
    models = models or list(MODEL_INFO)

    stats = {}
    if rebuild or not features_path.exists():
        logger.info("Construyendo la tabla de P2 en {}", features_path)
        df, stats = build_p2_features(
            PROCESSED_DATA_DIR,
            features_path,
            FECHA_INICIO,
            FECHA_FIN,
            cfg.SERVICE_DAY_START_HOUR,
            cfg.INITIAL_OCCUPANCY,
            cfg.EMPTY_THRESHOLD,
            cfg.FULL_THRESHOLD,
            cfg.HORIZON_HOURS,
            cfg.EMISSION_HOURS,
        )
        logger.info("Resumen de la tabla: {}", stats)
    else:
        df = pd.read_parquet(features_path)

    splits = dividir_particiones(df)
    add_baseline_scores(splits)

    common_params = {
        "horizonte_horas": cfg.HORIZON_HOURS,
        "horas_emision": f"{cfg.EMISSION_HOURS[0]}-{cfg.EMISSION_HOURS[-1]}",
        "inicio_dia_servicio": cfg.SERVICE_DAY_START_HOUR,
        "ocupacion_inicial": cfg.INITIAL_OCCUPANCY,
        "umbral_vacia": cfg.EMPTY_THRESHOLD,
        "umbral_llena": cfg.FULL_THRESHOLD,
        "top_k": cfg.TOP_K,
        "n_variables": len(P2_FEATURE_COLUMNS),
        "n_train": len(splits["train"]),
        "n_valid": len(splits["valid"]),
        "n_test": len(splits["test"]),
        "n_estaciones": int(df["station_id"].nunique()),
    }
    if stats:
        common_params.update(
            {
                f"limpieza_{k}": v
                for k, v in stats.items()
                if k in {"viajes_validos", "viajes_descartados"}
            }
        )
    configurar_experimento()
    results = [run_model(name, splits, common_params, max_train_rows or None) for name in models]

    cfg.P2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results).set_index("run_name")
    summary_cols = [
        "valid_average_precision",
        "test_average_precision",
        "test_roc_auc",
        f"test_precision_at_{cfg.TOP_K}",
        f"test_recall_at_{cfg.TOP_K}",
        "test_recall",
        "test_precision",
        "test_f2",
    ]
    table[summary_cols].to_csv(cfg.P2_FIGURES_DIR / "comparacion_modelos.csv")
    if len(results) > 1:
        plot_comparison(results, cfg.P2_FIGURES_DIR / "comparacion_modelos.png")
    logger.info("Resumen:\n{}", table[summary_cols].round(3).to_string())


if __name__ == "__main__":
    app()
