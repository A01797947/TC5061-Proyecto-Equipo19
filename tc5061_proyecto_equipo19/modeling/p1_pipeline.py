"""Pipeline de entrenamiento real de P1, sin dependencias del servidor MLflow.

Se reutiliza en train_P1.py y se prueba sin conexión a PostgreSQL/Supabase.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(model) -> Pipeline:
    """Codifica la estación y conserva las variables temporales de demanda P1."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), ["origin_station_id"]),
            ("num", "passthrough", ["Año", "Mes", "Dia", "Dia_semana"]),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])
