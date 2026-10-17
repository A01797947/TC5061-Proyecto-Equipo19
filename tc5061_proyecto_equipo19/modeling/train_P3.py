import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow, mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def parse_args():
    p = argparse.ArgumentParser(description='Clustering de estaciones MiBici')
    p.add_argument('--trips-csv', required=True, help='CSV de viajes limpios')
    p.add_argument('--stations-csv', required=True, help='CSV del catálogo de estaciones')
    p.add_argument('--model-type', choices=['baseline','candidate'], default='candidate')
    p.add_argument('--n-clusters', type=int, default=3)
    p.add_argument('--n-init', type=int, default=10)
    p.add_argument('--max-iter', type=int, default=300)
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--experiment-name', default='MiBici - Clustering de estaciones')
    p.add_argument('--tracking-uri', default=None)
    p.add_argument('--output-dir', default='outputs')
    return p.parse_args()


def load_data(trips_path, stations_path):
    trips, stations = pd.read_csv(trips_path), pd.read_csv(stations_path)
    required_t = {'start_time','origin_station_id','destination_station_id'}
    required_s = {'station_id','latitude','longitude','capacity'}
    return trips, stations


def prepare_features(trips, stations):
    trips, stations = trips.copy(), stations.copy()
    
    for c in ['origin_station_id','destination_station_id']:
        trips[c] = trips[c].astype('string').str.strip()
    
    stations['station_id'] = stations['station_id'].astype('string').str.strip()
    trips['start_time'] = pd.to_datetime(trips['start_time'], errors='coerce')
    trips = trips.dropna(subset=['start_time','origin_station_id','destination_station_id']).copy()
    
    for c in ['latitude','longitude','capacity']:
        stations[c] = pd.to_numeric(stations[c], errors='coerce')
    
    stations = stations.dropna(subset=['station_id']).drop_duplicates('station_id')
    
    trips['hour'] = trips['start_time'].dt.hour
    trips['weekday'] = trips['start_time'].dt.dayofweek
    
    out = trips.groupby('origin_station_id').size().rename_axis('station_id').rename('viajes_iniciados')
    inc = trips.groupby('destination_station_id').size().rename_axis('station_id').rename('viajes_finalizados')
    
    usage = pd.concat([out,inc],axis=1).fillna(0)
    usage['total_viajes'] = usage['viajes_iniciados'] + usage['viajes_finalizados']
    
    hourly = pd.crosstab(trips['origin_station_id'],trips['hour']).reindex(columns=range(24),fill_value=0)
    hourly.columns = [f'salidas_hora_{h:02d}' for h in range(24)]; hourly.index.name='station_id'
    
    daily = pd.crosstab(trips['origin_station_id'],trips['weekday']).reindex(columns=range(7),fill_value=0)
    daily.columns = [f'salidas_dia_{d}' for d in range(7)]; daily.index.name='station_id'
    
    features = usage.join(hourly,how='outer').join(daily,how='outer').fillna(0).reset_index()
    features = features.merge(stations[['station_id','latitude','longitude','capacity']],on='station_id',how='left',validate='one_to_one')
    features = features.dropna(subset=['latitude','longitude'])
    features = features.loc[features['total_viajes'] > 0].reset_index(drop=True)
    
    if features.empty: raise ValueError('No hay estaciones con viajes y coordenadas válidas.')
    
    return features


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
    
    if args.tracking_uri: mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    
    trips, stations = load_data(args.trips_csv,args.stations_csv)
    features = prepare_features(trips,stations)
    columns = get_feature_columns(features,args.model_type)
    
    run_name = (f"{args.model_type}_k{args.n_clusters}_init{args.n_init}_seed{args.random_state}")
    
    with mlflow.start_run(run_name=run_name):
        for key,value in {'model_type':args.model_type,'n_clusters':args.n_clusters,'n_init':args.n_init,'max_iter':args.max_iter,
                          'random_state':args.random_state,'n_features':len(columns),'n_stations':len(features),
                          'feature_columns':','.join(columns),'trips_csv':str(Path(args.trips_csv).resolve()),
                          'stations_csv':str(Path(args.stations_csv).resolve())}.items(): mlflow.log_param(key,value)
        
        model, scaler, metrics, summary = train_evaluate(features,columns,args,args.output_dir)
        
        for key,value in metrics.items():
            if isinstance(value,(int,float)) and np.isfinite(value): mlflow.log_metric(key,value)
        
        for name in ['metrics.json','cluster_assignments.csv','cluster_summary.csv','clusters_map.png']:
            mlflow.log_artifact(str(Path(args.output_dir)/name))
        
        mlflow.sklearn.log_model(Pipeline([('scaler',scaler),('kmeans',model)]),artifact_path='model')
        
        print('Métricas:',json.dumps(metrics,indent=2,ensure_ascii=False)); print('\nResumen por cluster:\n',summary)
        print('Experimento MLflow:',args.experiment_name); print('Artefactos:',Path(args.output_dir).resolve())

if __name__ == '__main__': main()