import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import argparse


def get_git_commit() -> str:
    """Obtiene el hash del commit actual de Git de manera segura."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        return commit
    except Exception:
        return "unknown_commit"


# Función para carga de la información.
def load_data(file_path: Path) -> pd.DataFrame:
    """Carga el dataset procesado desde un archivo parquet."""
    print(f"Cargando datos desde: {file_path}")
    return pd.read_parquet(file_path)


# Función para generación del dataset como se requiere para este modelo.
def prepare_features(df: pd.DataFrame):
    """Realiza la agrupación por fecha y estación de origen, y separa X y y."""
    df_grouped = df.groupby([
        'Año', 
        'Mes', 
        'Dia', 
        'Dia_semana', 
        'origin_station_id', 
    ]).size().reset_index(name='cantidad_viajes')
    
    X = df_grouped[['Año', 'Mes', 'Dia', 'Dia_semana', 'origin_station_id']]
    y = df_grouped['cantidad_viajes']
    
    return X, y, df_grouped


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
    parser.add_argument('--filepath', type=str, default='../data/processed/viajes_limpios_enriquecidos.parquet', help='Ruta del dataset procesado')

    args = parser.parse_args()

    # Configurar el nombre del experimento basado en tu equipo
    experiment_name = "mibici-equipo19-P1"
    mlflow.set_experiment(experiment_name)
    
    # Parámetros generales y de versión
    data_path = Path(args.filepath)
    
    # Cargar datos
    df = load_data(data_path)
    
    # Ingeniería de características y agrupación
    X, y, df_grouped = prepare_features(df)
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Obtener el hash del commit actual de Git
    git_commit_hash = get_git_commit()

    # RUN 1: MODELO BASELINE (Regresión Lineal)
    print("\nIniciando Run: Baseline (Regresión Lineal)...")
    with mlflow.start_run(run_name="baseline-lineal-p1"):
        mlflow.log_param("total_samples", len(df_grouped))
        mlflow.log_param("model_type", "LinearRegression")
        
        mlflow.set_tag("team", "equipo19")
        mlflow.set_tag("question", "P1")
        mlflow.set_tag("git_commit", git_commit_hash)
        
        baseline_model = LinearRegression()
        baseline_pipeline = build_pipeline(baseline_model)
        
        print("Entrenando Regresión Lineal...")
        baseline_pipeline.fit(X_train, y_train)
        
        y_pred_base = baseline_pipeline.predict(X_test)
        mse_base = mean_squared_error(y_test, y_pred_base)
        r2_base = r2_score(y_test, y_pred_base)
        
        mlflow.log_metric("mse", mse_base)
        mlflow.log_metric("r2", r2_base)
        print(f"MSE: {mse_base:.4f} | R^2: {r2_base:.4f}")
        
        artifact_file_base = Path("real_vs_predicho_baseline.png")
        generate_artifact(y_test, y_pred_base, "Linear Regression", artifact_file_base)
        mlflow.log_artifact(str(artifact_file_base))
        if artifact_file_base.exists(): artifact_file_base.unlink()

    # RUN 2: MODELO CANDIDATO (Ridge Regression)
    print("\nIniciando Run: Candidato (Ridge Regression)...")
    with mlflow.start_run(run_name="candidato-ridge-p1"):
        ridge_params = {
            "alpha": args.alpha,
            "solver": args.solver,
            "fit_intercept": args.fit_intercept,
            "max_iter": args.max_iter,
            "random_state": 42
        }
        mlflow.log_params(ridge_params)
        mlflow.log_param("total_samples", len(df_grouped))
        mlflow.log_param("model_type", "Ridge")
        
        mlflow.set_tag("team", "equipo19")
        mlflow.set_tag("question", "P1")
        mlflow.set_tag("git_commit", git_commit_hash)
        
        candidato_model = Ridge(**ridge_params)
        candidato_pipeline = build_pipeline(candidato_model)
        
        print("Entrenando Ridge Regression...")
        candidato_pipeline.fit(X_train, y_train)
        
        y_pred_cand = candidato_pipeline.predict(X_test)
        mse_cand = mean_squared_error(y_test, y_pred_cand)
        r2_cand = r2_score(y_test, y_pred_cand)
        
        mlflow.log_metric("mse", mse_cand)
        mlflow.log_metric("r2", r2_cand)
        print(f"MSE: {mse_cand:.4f} | R^2: {r2_cand:.4f}")
        
        artifact_file_cand = Path("real_vs_predicho_candidato.png")
        generate_artifact(y_test, y_pred_cand, "Ridge Regression", artifact_file_cand)
        mlflow.log_artifact(str(artifact_file_cand))
        if artifact_file_cand.exists(): artifact_file_cand.unlink()

    print("\nRuns terminados")