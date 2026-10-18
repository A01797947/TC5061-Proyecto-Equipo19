import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from tc5061_proyecto_equipo19.features.p3 import construir_features_estaciones
from tc5061_proyecto_equipo19.modeling.mlflow_utils import configurar_experimento, registrar_run


def parse_args():
    p = argparse.ArgumentParser(description='Clustering de estaciones MiBici')
    p.add_argument('--trips-csv', required=True, help='CSV de viajes limpios')
    p.add_argument('--stations-csv', required=True, help='CSV del catálogo de estaciones')
    p.add_argument('--model-type', choices=['baseline','candidate'], default='candidate')
    p.add_argument('--n-clusters', type=int, default=3)
    p.add_argument('--n-init', type=int, default=10)
    p.add_argument('--max-iter', type=int, default=300)
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--output-dir', default='outputs')
    return p.parse_args()


def _read_table(path, columns=None):
    """Lee Parquet o CSV; con `columns` carga solo las necesarias."""
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def load_data(trips_path, stations_path):
    trips = _read_table(trips_path, ['start_time', 'origin_station_id', 'destination_station_id'])
    stations = _read_table(stations_path)
    return trips, stations


def get_feature_columns(features, model_type):
    basic = ['viajes_iniciados','viajes_finalizados','total_viajes']
    if model_type == 'baseline': return basic
    return basic + [c for c in features.columns if c.startswith('salidas_hora_') or c.startswith('salidas_dia_')]


def train_evaluate(features, columns, args, output_dir):
    if args.n_clusters < 2 or len(features) <= args.n_clusters:
        raise ValueError('n-clusters debe ser >= 2 y menor que el número de estaciones.')
    
    scaler = StandardScaler()
    X = scaler.fit_transform(features[columns].astype(float))
    model = KMeans(n_clusters=args.n_clusters,n_init=args.n_init,max_iter=args.max_iter,random_state=args.random_state,algorithm='lloyd')
    labels = model.fit_predict(X)
        
    metrics = {'silhouette':float(silhouette_score(X,labels)), 'davies_bouldin':float(davies_bouldin_score(X,labels)),
               'calinski_harabasz':float(calinski_harabasz_score(X,labels)), 'inertia':float(model.inertia_),
               'n_stations':int(len(features)), 'n_features':int(len(columns))}
    
    outdir = Path(output_dir); outdir.mkdir(parents=True,exist_ok=True)
    assigned = features.copy(); assigned['cluster'] = labels
    assigned.to_csv(outdir/'cluster_assignments.csv',index=False)
    
    (outdir/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    
    summary = assigned.groupby('cluster').agg(numero_estaciones=('station_id','nunique'), promedio_salidas=('viajes_iniciados','mean'),
        promedio_llegadas=('viajes_finalizados','mean'), promedio_total_viajes=('total_viajes','mean'), capacidad_promedio=('capacity','mean'),
        latitud_media=('latitude','mean'), longitud_media=('longitude','mean')).round(2)
    summary.to_csv(outdir/'cluster_summary.csv')
    
    plt.figure(figsize=(9,6)); sc=plt.scatter(assigned.longitude,assigned.latitude,c=assigned.cluster,cmap='tab10',s=35,alpha=.8,edgecolors='black',linewidths=.25)
    plt.xlabel('Longitud'); plt.ylabel('Latitud'); plt.title(f'MiBici {args.model_type}: k={args.n_clusters}'); plt.colorbar(sc,label='Cluster'); plt.tight_layout()
    plt.savefig(outdir/'clusters_map.png',dpi=150); plt.close()
    
    return model, scaler, metrics, summary


def main():
    args = parse_args()
    random.seed(args.random_state); np.random.seed(args.random_state)
    
    # Configurar experimento, tracking URI, fechas, daat_dversion y coomit 
    configurar_experimento()
    
    trips, stations = load_data(args.trips_csv,args.stations_csv)
    features = construir_features_estaciones(trips,stations)
    columns = get_feature_columns(features,args.model_type)
    
    run_name = (f"P3-{args.model_type}_k{args.n_clusters}_init{args.n_init}_seed{args.random_state}")

    params = {
        "model_type" : args.model_type, 'n_clusters': args.n_clusters, 'n_init': args.n_init,
        "max_iter": args.max_iter, "random_state": args.random_state,
        'n_features': len(columns), 'n_stations': len(features),
        'feature_columns': ','.join(columns),
        'trips_path': str(Path(args.trips_csv).resolve()),
        'stations_path': str(Path(args.stations_csv).resolve()),
    }

    tipo = "baseline" if args.model_type == "baseline" else "candidato"

    model, scaler, metrics, summary = train_evaluate(features, columns, args, args.output_dir)

    outdir = Path(args.output_dir)
    registrar_run(
        run_name,
        pregunta="P3",
        tipo=tipo,
        params=params,
        metricas=metrics,
        artefactos=[outdir / n for n in ['metrics.json', 'cluster_assignments.csv',
                                         'cluster_summary.csv', 'clusters_map.png']],
        modelo= Pipeline([('scaler', scaler), ('kmeans', model)]),
    )
    print('Métricas:', json.dumps(metrics, indent=2, ensure_ascii=False)); print('\nResumen por cluster:\n', summary)
    print('Artefactos:', outdir.resolve())
    
if __name__ == '__main__': main()