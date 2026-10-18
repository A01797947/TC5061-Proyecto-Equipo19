import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tc5061_proyecto_equipo19.config import INTERIM_DATA_DIR
from tc5061_proyecto_equipo19.features.p1 import construir_demanda_diaria
from tc5061_proyecto_equipo19.modeling.mlflow_utils import configurar_experimento, registrar_run


# Función para carga de la información.
def load_data(file_path: Path) -> pd.DataFrame:
    """Carga el dataset procesado desde un archivo parquet."""
    print(f"Cargando datos desde: {file_path}")
    return pd.read_parquet(file_path)

# Construcción del pipeline genérico que acepta cualquier modelo
def build_pipeline(model) -> Pipeline:
    """Construye el ColumnTransformer y el Pipeline con el modelo proporcionado."""
    categorical_features = ['origin_station_id']
    numerical_features = ['Año', 'Mes', 'Dia', 'Dia_semana']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    return model_pipeline


# Función para obtener un artefacto de este análisis (Valores Reales vs Predichos).
def generate_artifact(y_test, y_pred, model_name: str, artifact_path: Path):
    """Genera una gráfica de dispersión de Valores Reales vs. Predichos como artefacto."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='cornflowerblue', edgecolors='none')
    
    # Línea diagonal de referencia ideal (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Predicción Perfecta')
    
    plt.title(f"Valores Reales vs. Predichos ({model_name})")
    plt.xlabel("Cantidad de Viajes Real")
    plt.ylabel("Cantidad de Viajes Predicha")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(artifact_path)
    plt.close()


# Ejecución de la serie de pasos
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Entrenamiento de modelos Baseline y Candidato (Ridge) con MLflow')
    parser.add_argument('--alpha', type=float, default=1.0, help='Parámetro de regularización L2 para Ridge')
    parser.add_argument('--solver', type=str, default='auto', help='Algoritmo de cálculo (ej. auto, svd, cholesky, lsqr, sag)')
    parser.add_argument('--fit_intercept', type=bool, default=True, help='Si se debe calcular la intercepción para el modelo')
    parser.add_argument('--max_iter', type=int, default=None, help='Número máximo de iteraciones para los solvers iterativos')
    parser.add_argument('--filepath', type=str, default=(INTERIM_DATA_DIR / 'viajes_limpios_enriquecidos.parquet'), help='Ruta del dataset procesado')

    args = parser.parse_args()

    # Configurar experimento, tracking URI, fechas, daat_dversion y coomit 
    configurar_experimento()
    
    # Cargar datos
    df = load_data(Path(args.filepath))
    
    # Ingeniería de características y agrupación
    X, y, df_grouped = construir_demanda_diaria(df)
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RUN 1: MODELO BASELINE (Regresión Lineal)

    print("\nEntrenando Regresión Lineal....")
    baseline_pipeline =  build_pipeline(LinearRegression())
    baseline_pipeline.fit(X_train, y_train)

    y_pred_base = baseline_pipeline.predict(X_test)
    mse_base = mean_squared_error(y_test, y_pred_base)
    r2_base = r2_score(y_test, y_pred_base)
    print(f"MSE: {mse_base:.4f} | R^2: {r2_base:.4f}")

    artifact_file_base = Path("real_vs_predicho_baseline.png")
    registrar_run(
        "P1-baseline-regresion-lineal",
        pregunta="P1",
        tipo="baseline",
        params={"total_samples": len(df_grouped), "model:type": "LinearRegression"},
        metricas={"mse": mse_base, "r2": r2_base},
        artefactos=[artifact_file_base],

    )
    artifact_file_base.unlink(missing_ok=True)


    # RUN 2: MODELO CANDIDATO (Ridge Regression)

    ridge_params = {
            "alpha": args.alpha,
            "solver": args.solver,
            "fit_intercept": args.fit_intercept,
            "max_iter": args.max_iter,
            "random_state": 42
    }
    print("\nEntrenando Ridge Regression...")
    candidato_model = Ridge(**ridge_params)
    candidato_pipeline = build_pipeline(candidato_model)
    
    candidato_pipeline.fit(X_train, y_train)
    
    y_pred_cand = candidato_pipeline.predict(X_test)
    mse_cand = mean_squared_error(y_test, y_pred_cand)
    r2_cand = r2_score(y_test, y_pred_cand)
    print(f"MSE: {mse_cand:.4f} | R^2: {r2_cand:.4f}")

    artifact_file_cand = Path("real_vs_predicho_candidato.png")
    generate_artifact(y_test, y_pred_cand, "Ridge Regression", artifact_file_cand)
    registrar_run(
        "P1-candidato-ridge",
        pregunta="P1",
        tipo="candidato",
        params= {**ridge_params, "total_samples": len(df_grouped), "model_type": "Ridge"},
        metricas={"mse": mse_cand, "r2": r2_cand},
        artefactos=[artifact_file_cand]
    )

    artifact_file_cand.unlink(missing_ok=True)
    
    print("\nRuns terminados")