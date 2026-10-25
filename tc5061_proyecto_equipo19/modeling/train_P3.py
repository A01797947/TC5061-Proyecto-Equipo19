"""P3: agrupamiento de estaciones MiBici con K-Means y registro en MLflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.features.p3 import construir_features_estaciones
from tc5061_proyecto_equipo19.modeling.mlflow_utils import (
    configurar_experimento,
    iniciar_run,
    log_modelo_sklearn,
)


def _read_table(path: str | Path, columns=None) -> pd.DataFrame:
    """Lee un archivo CSV o Parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga viajes mensuales y estaciones desde data/processed."""
    processed_dir = cfg.PROCESSED_DATA_DIR
    station_path = processed_dir / "stations.parquet"
    if not station_path.exists():
        raise FileNotFoundError(f"No existe {station_path}. Ejecuta primero la ingesta.")

    trip_paths = [processed_dir / f"trips_{month}.parquet" for month in cfg.MESES]
    missing = [str(path) for path in trip_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos de viajes procesados. Ejecuta primero la ingesta. "
            f"Primeros archivos ausentes: {missing[:3]}"
        )

    required = ["start_time", "end_time", "origin_station_id", "destination_station_id"]
    trips = pd.concat([_read_table(path, columns=required) for path in trip_paths], ignore_index=True)
    stations = _read_table(station_path)
    return trips, stations


def load_data(trips_path: str | Path, stations_path: str | Path):
    """Carga los archivos indicados por CLI; acepta CSV o Parquet."""
    trips = _read_table(trips_path, ["start_time", "end_time", "origin_station_id", "destination_station_id"])
    stations = _read_table(stations_path)
    return trips, stations


def get_feature_columns(features: pd.DataFrame, model_type: str) -> list[str]:
    basic = ["viajes_iniciados", "viajes_finalizados", "total_viajes"]
    if model_type == "baseline":
        return basic
    if model_type == "perfil":
        return [c for c in features.columns
                if c.startswith("pct_")] + ["ratio_finde"]
    if model_type == "hibrido" :
        return [c for c in features.columns
                        if c.startswith("pct_")] + ["ratio_finde", "log_viajes_dia"]
    return basic + [
        c for c in features.columns
        if c.startswith(("salidas_hora_", "salidas_dia_"))
    ]

def build_pipeline (model_type, columns, n_clusters, n_init, max_iter, random_state,
                    n_components = 5, volume_weight=2.0):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter,
                    random_state=random_state, algorithm="lloyd")
    if model_type != "hibrido":
        return Pipeline([("scaler", StandardScaler()), ("kmeans", kmeans)])

    volume_cols = ["log_viajes_dia"]
    profile_cols = [c for c in columns if c not in volume_cols]
    features = ColumnTransformer(
        [
            ("perfil", Pipeline([("scaler", StandardScaler()), 
                                 ("pca", PCA(n_components=n_components, random_state=random_state))]), profile_cols),
            ("volumen", StandardScaler(), volume_cols)
        ],
        transformer_weights={"perfil": 1.0, "volumen": volume_weight}
    )
    return Pipeline([("features", features), ("kmeans", kmeans)])

def train_evaluate(features: pd.DataFrame, columns: list[str], n_clusters: int = 3, n_init: int = 10, max_iter: int = 300, 
                   random_state: int = 42, output_dir: str | Path | None = None, model_type: str = "candidate", 
                   n_components: int = cfg.P3_N_COMPONENTS, volume_weight: float = cfg.P3_VOLUME_WEIGHT):
    """Entrena StandardScaler + KMeans y genera métricas/archivos de evaluación."""
    if n_clusters < 2 or len(features) <= n_clusters:
        raise ValueError("n_clusters debe ser >= 2 y menor que el número de estaciones.")
    X = features[columns].astype(float)
    pipeline = build_pipeline(model_type, columns, n_clusters, n_init, max_iter, random_state,
                    n_components, volume_weight)
    labels = pipeline.fit_predict(X)
    X_scaled = pipeline[:-1].transform(X)
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2 or len(unique_labels) >= len(features):
        raise ValueError("No se puede calcular silhouette: el resultado no tiene entre 2 y n-1 clústeres.")

    metrics = {
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
        "inertia": float(pipeline.named_steps["kmeans"].inertia_),
        "n_stations": len(features),
        "n_features": len(columns),
        "n_clusters_observed": len(unique_labels),
    }

    summary = None
    
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        assigned = features.copy()
        assigned["cluster"] = labels

        cols_sal_manana = [f"pct_salidas_hora_{h:02d}" for h in range(6, 10)]
        cols_sal_tarde = [f"pct_salidas_hora_{h:02d}" for h in range(16, 21)]
        cols_lle_manana = [f"pct_llegadas_hora_{h:02d}" for h in range(6, 10)]
        assigned["pct_salidas_manana"] = assigned[cols_sal_manana].sum(axis=1)
        assigned["pct_salidas_tarde"] = assigned[cols_sal_tarde].sum(axis=1)
        assigned["pct_llegadas_manana"] = assigned[cols_lle_manana].sum(axis=1)

        assigned.to_csv(outdir / "cluster_assignments.csv", index=False)
        (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = assigned.groupby("cluster").agg(
            numero_estaciones=("station_id", "nunique"),
            promedio_salidas=("viajes_iniciados", "mean"),
            promedio_llegadas=("viajes_finalizados", "mean"),
            promedio_total_viajes=("total_viajes", "mean"),
            mediana_viajes_dia=("total_viajes", "median"),
            pct_salidas_manana=("pct_salidas_manana", "mean"),
            pct_salidas_tarde=("pct_salidas_tarde", "mean"),
            pct_llegadas_manana=("pct_llegadas_manana", "mean"),
            ratio_finde=("ratio_finde", "mean"),
            capacidad_promedio=("capacity", "mean"),
            latitud_media=("latitude", "mean"),
            longitud_media=("longitude", "mean"),
        ).round(3)
        summary.to_csv(outdir / "cluster_summary.csv")

        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(assigned["longitude"], assigned["latitude"], c=assigned["cluster"],
                            cmap="tab10", s=35, alpha=0.8, edgecolors="black", linewidths=0.25,)
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.set_title(f"MiBici {model_type}: k={n_clusters}")
        fig.colorbar(scatter, ax=ax, label="Clúster")
        fig.tight_layout()
        fig.savefig(outdir / "clusters_map.png", dpi=150)
        plt.close(fig)
        
    return pipeline, metrics, summary


def _profile_by_period(trips: pd.DataFrame, stations: pd.DataFrame,
                       start: str, end: str) -> pd.DataFrame:
    """Perfiles por estación usando solamente los viajes del intervalo indicado.

    Se expresan los conteos por día natural del intervalo; así los cortes de
    18, 3 y 3 meses son comparables pese a tener distinta duración.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    times = pd.to_datetime(trips["start_time"], errors="coerce")
    selected = trips.loc[(times >= start_ts) & (times < end_ts + pd.Timedelta(days=1))]
    if selected.empty:
        raise ValueError(f"No hay viajes en el intervalo {start} - {end}")
    features = construir_features_estaciones(selected, stations)
    days = (end_ts - start_ts).days + 1
    count_cols = ["viajes_iniciados", "viajes_finalizados", "total_viajes"]
    count_cols += [col for col in features if col.startswith(("salidas_hora_", "salidas_dia_"))]
    features[count_cols] = features[count_cols].astype(float) / days
    features["log_viajes_dia"] = np.log1p(features["total_viajes"])
    return features


def _evaluate_period(pipeline, train_features, other_features, columns, period):
    """Silhouette temporal y estabilidad ARI sin reentrenar sobre el futuro."""
    common = train_features[["station_id"]].merge(
        other_features[["station_id"] + columns], on="station_id", how="inner",
    )
    if len(common) < 3:
        return {f"{period}_n_stations": len(common)}
    train_idx = train_features.set_index("station_id")
    X_train = train_idx.loc[common["station_id"], columns].astype(float)
    X_future = common[columns].astype(float)
    labels_train = pipeline.predict(X_train)
    labels_future = pipeline.predict(X_future)
    scaled_future = pipeline[:-1].transform(X_future)
    result = {
        f"{period}_n_stations": len(common),
        f"{period}_ari_vs_train": float(adjusted_rand_score(labels_train, labels_future)),
    }
    if 1 < len(np.unique(labels_future)) < len(X_future):
        result[f"{period}_silhouette"] = float(silhouette_score(scaled_future, labels_future))
    return result


def train_model(model: str = "kmeans", params: dict | None = None,
                nested: bool = False, stage_of_work: str | None = None):
    """Ajusta en train y mide estabilidad de grupos en valid/test posteriores."""
    params = params or {}
    if model != "kmeans":
        raise ValueError(f"Modelo P3 no admitido: {model}. Usa 'kmeans'.")
    n_clusters = int(params.get("n_clusters", cfg.P3_N_CLUSTERS))
    n_init = int(params.get("n_init", cfg.P3_N_INIT))
    max_iter = int(params.get("max_iter", cfg.P3_MAX_ITER))
    random_state = int(params.get("random_state", cfg.RANDOM_SEED))
    n_components = int(params.get("n_components", cfg.P3_N_COMPONENTS))
    volume_weight = float(params.get("volume_weight", cfg.P3_VOLUME_WEIGHT))
    model_type = params.get("model_type", "candidate")
    if model_type not in {"baseline", "candidate", "perfil", "hibrido"}:
        raise ValueError("model_type debe ser 'baseline' o 'candidate' o 'perfil' o 'hibrido'.")
    random.seed(random_state)
    np.random.seed(random_state)
    trips, stations = load_processed_data()
    train_features = _profile_by_period(trips, stations, cfg.FECHA_INICIO, cfg.FIN_ENTRENAMIENTO)
    valid_features = _profile_by_period(trips, stations, cfg.INICIO_VALIDACION, cfg.FIN_VALIDACION)
    test_features = _profile_by_period(trips, stations, cfg.INICIO_PRUEBA, cfg.FECHA_FIN)
    del trips, stations
    columns = get_feature_columns(train_features, model_type)
    stage = stage_of_work or ("baseline" if model_type == "baseline" else "candidate")
    run_name = f"P3-{model_type}-kmeans-k{n_clusters}-init{n_init}-seed{random_state}"
    if model_type == "hibrido":
        run_name += f"-pc{n_components}-w{volume_weight:g}"
    run_params = {
        "model": model, "model_type": model_type, "n_clusters": n_clusters,
        "n_init": n_init, "max_iter": max_iter, "random_state": random_state,
        "n_features": len(columns), "n_stations": len(train_features),
        "feature_columns": ",".join(columns),
        "split_strategy": "temporal_station_profiles_per_day",
        "train_end": cfg.FIN_ENTRENAMIENTO,
        "validation_start": cfg.INICIO_VALIDACION,
        "validation_end": cfg.FIN_VALIDACION,
        "test_start": cfg.INICIO_PRUEBA,
    }
    if model_type == "hibrido":
        run_params.update({"n_components": n_components, "volume_weight": volume_weight})
    with tempfile.TemporaryDirectory(prefix="mibici_p3_") as temp_dir:
        output_dir = Path(temp_dir)
        pipeline, metrics, summary = train_evaluate(
            train_features, columns, n_clusters, n_init, max_iter, random_state,
            output_dir=output_dir, model_type=model_type,
            n_components=n_components, volume_weight=volume_weight
        )
        metrics.update(_evaluate_period(pipeline, train_features, valid_features, columns, "valid"))
        metrics.update(_evaluate_period(pipeline, train_features, test_features, columns, "test"))
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        configurar_experimento()
        with iniciar_run(run_name, pregunta="P3", tipo=stage, params=run_params, nested=nested) as run:
            mlflow.set_tags({
                "question": "p3", "model_family": "clustering", "stage_of_work": stage,
                "model": model,
                "mlflow.note.content": "K-Means entrenado solo en train; valid/test miden estabilidad temporal.",
            })
            mlflow.log_metrics(metrics)
            config_path = output_dir / "run_config.json"
            config_path.write_text(json.dumps({
                "question": "p3", "model": model, "params": run_params,
                "features": columns, "random_seed": random_state,
                "normalization": "counts_per_calendar_day",
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            mlflow.log_artifact(str(config_path), artifact_path="config")
            for filename in ["metrics.json", "cluster_assignments.csv", "cluster_summary.csv"]:
                file = output_dir / filename
                if file.is_file():
                    mlflow.log_artifact(str(file), artifact_path="evaluation")
            plot_file = output_dir / "clusters_map.png"
            if plot_file.is_file():
                mlflow.log_artifact(str(plot_file), artifact_path="plots")
            example = train_features[columns].astype(float).head(5)
            mlflow.log_input(mlflow.data.from_pandas(
                train_features[columns].astype(float).head(1000),
                name="mibici-p3-training-station-profiles",
            ), context="training")
            log_modelo_sklearn(pipeline, example)
            print(f"Run P3: {run.info.run_id}")
            print(f"Métricas: {json.dumps(metrics, indent=2, ensure_ascii=False)}")
            if summary is not None:
                print(f"Resumen por clúster:\n{summary}")
            return {"run_id": run.info.run_id, "run_name": run_name, **metrics}


def main():
    parser = argparse.ArgumentParser(description="Clustering de estaciones MiBici con MLflow")
    parser.add_argument("--trips-csv", help="CSV/Parquet de viajes; si se omite, usa data/processed")
    parser.add_argument("--stations-csv", help="CSV/Parquet de estaciones; si se omite, usa data/processed")
    parser.add_argument("--model-type", choices=["baseline", "candidate", "perfil"], default="candidate")
    parser.add_argument("--n-clusters", type=int, default=cfg.P3_N_CLUSTERS)
    parser.add_argument("--n-init", type=int, default=cfg.P3_N_INIT)
    parser.add_argument("--max-iter", type=int, default=cfg.P3_MAX_ITER)
    parser.add_argument("--random-state", type=int, default=cfg.RANDOM_SEED)
    args = parser.parse_args()

    if args.trips_csv and args.stations_csv:
        trips, stations = load_data(args.trips_csv, args.stations_csv)
        features = construir_features_estaciones(trips, stations)
        columns = get_feature_columns(features, args.model_type)
        output_dir = cfg.REPORTS_DIR / "p3"
        pipeline, metrics, _ = train_evaluate(features, columns, args.n_clusters, args.n_init, args.max_iter,
                                            args.random_state, output_dir=output_dir, model_type=args.model_type,)
        
        configurar_experimento()
        stage = "baseline" if args.model_type == "baseline" else "candidate"
        
        with iniciar_run(f"P3-{args.model_type}-kmeans-k{args.n_clusters}", "P3", stage,
                    params={"n_clusters": args.n_clusters, "n_init": args.n_init,
                    "max_iter": args.max_iter, "random_state": args.random_state},) as run:
            mlflow.set_tags({"question": "p3", "model_family": "clustering", "stage_of_work": stage})
            mlflow.log_metrics(metrics)
            example = features[columns].head(5)
            
            mlflow.sklearn.log_model(sk_model=pipeline, name="modelo",
                signature=infer_signature(example, pipeline.predict(example)), input_example=example,)
            
            for name in ["metrics.json", "cluster_assignments.csv", "cluster_summary.csv", "clusters_map.png"]:
                file = output_dir / name
                if file.exists(): mlflow.log_artifact(str(file), artifact_path="evaluation")
                
            print(f"Run P3: {run.info.run_id}; métricas: {metrics}")
    else:
        train_model("kmeans",
            {"n_clusters": args.n_clusters, "n_init": args.n_init,
             "max_iter": args.max_iter, "random_state": args.random_state,
             "model_type": args.model_type},)


if __name__ == "__main__":
    main()