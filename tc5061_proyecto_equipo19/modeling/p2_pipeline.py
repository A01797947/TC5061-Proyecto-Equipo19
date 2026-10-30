"""Estimadores reutilizables de P2, sin efectos secundarios de MLflow.

Importado tanto por el entrenamiento real como por las pruebas PyTest.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tc5061_proyecto_equipo19 import config as cfg


def build_estimator(name: str, overrides: dict | None = None):
    """Construye exactamente el estimador empleado en el entrenamiento de P2."""
    overrides = overrides or {}
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                C=overrides.get("C", cfg.P2_LOGREG_C),
                max_iter=overrides.get("max_iter", cfg.P2_LOGREG_MAX_ITER),
                random_state=cfg.RANDOM_SEED,
            ),
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=overrides.get("learning_rate", cfg.P2_HGB_LEARNING_RATE),
            max_iter=overrides.get("max_iter", cfg.P2_HGB_MAX_ITER),
            max_leaf_nodes=overrides.get("max_leaf_nodes", cfg.P2_HGB_MAX_LEAF_NODES),
            min_samples_leaf=overrides.get("min_samples_leaf", cfg.P2_HGB_MIN_SAMPLES_LEAF),
            l2_regularization=overrides.get("l2_regularization", cfg.P2_HGB_L2_REGULARIZATION),
            early_stopping=False,
            random_state=cfg.RANDOM_SEED,
        )
    raise ValueError(f"Modelo desconocido: {name}")
