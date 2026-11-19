# Entregable 7 — Resultados reales: alta50

**Ambiente:** motor podman; 50 usuarios, 180 s configurados; tres modelos desde snapshots.

## Desempeño de las inferencias

| Indicador | Medición |
|---|---:|
| Peticiones totales | 2,042 |
| Inferencias satisfactorias (HTTP 200 + JSON esperado) | 2,042 |
| Fallos | 0 (0.00%) |
| Throughput medio exitoso | 11.34 pred/s |
| Latencia promedio | 2798.0 ms |
| Latencia p50 | 2200.0 ms |
| Latencia p95 | 4800.0 ms |
| Latencia p99 | 7000.0 ms |

### Desglose por endpoint

| Modelo | Peticiones | Errores | p95 (ms) |
|---|---:|---:|---:|
| POST /v1/predict/p1 | 1034 | 0 | 2300.0 |
| POST /v1/predict/p2 | 584 | 0 | 5000.0 |
| POST /v1/predict/p3 | 424 | 0 | 5000.0 |

## Recursos de contenedores

| Servicio | CPU promedio | Memoria máxima |
|---|---:|---:|
| api | 4.7% | 0.0 MiB |
| mlflow | 1.8% | 0.0 MiB |

## FinOps (simulación de infraestructura; throughput medido localmente)

**Referencia:** [EC2 t3.large, Linux, us-east-1](https://aws.amazon.com/ec2/instance-types/t3/), USD 0.0835/h.
**Costo hipotético de 180 s:** USD 0.004175.
**Costo unitario estimado:** USD 0.00000204 por inferencia exitosa.
**Costo por 1,000 inferencias exitosas:** USD 0.002045.
**Escenario VM encendida 720 h/mes:** USD 60.12/mes (sin extras).

### Limitaciones del cálculo

- Proyección teórica para una única instancia: no es una factura real de AWS.
- Asume tasa constante durante toda la ventana configurada; incluye ramp-up y puede subestimar capacidad estable.
- El precio de la VM se asigna por completo a API + MLflow. No incluye impuestos, discos, tráfico, red, backups ni Supabase.
- La tasa medida en la laptop/Podman no garantiza el rendimiento de EC2 t3.large.
- T3 usa créditos de CPU: un uso sostenido por encima del baseline puede generar cargos adicionales.
- Las mediciones del proceso Locust se ejecutan fuera de la VM hipotética: su costo no se incluye.

### Ideas de optimización para validar

- Repetir las pruebas con 5, 20 y 50 usuarios y comparar p95, errores, RPS y costo unitario.
- Evaluar separar MLflow del host de inferencia y dimensionar CPU/RAM según picos reales.
- Con picos intermitentes, comparar autoscaling / horarios de apagado contra una VM 24/7.
- Si CPU está saturada, comparar otra familia de instancias no burstable o más réplicas.
- Cachear únicamente consultas deterministas y repetidas si la latencia y coherencia lo permiten.

## Gráficas de las mediciones

![alta50_grafico_latencia.png](alta50_grafico_latencia.png)
![alta50_grafico_rps_errores.png](alta50_grafico_rps_errores.png)
![alta50_grafico_recursos.png](alta50_grafico_recursos.png)

**Metodología:** Locust envía solicitudes HTTP con ejemplos P1/P2/P3 (50%/30%/20%).
Los archivos `*_locust*.csv` contienen las métricas capturadas por Locust y
`*_recursos.csv` contiene las muestras periódicas de `docker stats`/`podman stats`.
Las estimaciones no deben presentarse como gasto real ni como benchmark ejecutado en AWS.
