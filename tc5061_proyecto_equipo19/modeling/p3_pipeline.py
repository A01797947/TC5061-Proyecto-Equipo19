"""Pipeline K-Means P3 compartido por el entrenamiento y PyTest, sin MLflow."""
from __future__ import annotations

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(model_type, columns, n_clusters, n_init, max_iter, random_state,
                   n_components=5, volume_weight=2.0):
    """Conserva el preprocesamiento y parámetros usados por train_P3.py."""
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter,
                    random_state=random_state, algorithm="lloyd")
    if model_type != "hibrido":
        return Pipeline([("scaler", StandardScaler()), ("kmeans", kmeans)])

    volume_cols = ["log_viajes_dia"]
    profile_cols = [c for c in columns if c not in volume_cols]
    features = ColumnTransformer(
        [
            ("perfil", Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=random_state)),
            ]), profile_cols),
            ("volumen", StandardScaler(), volume_cols),
        ],
        transformer_weights={"perfil": 1.0, "volumen": volume_weight},
    )
    return Pipeline([("features", features), ("kmeans", kmeans)])
