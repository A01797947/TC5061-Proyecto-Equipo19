# Model Card — P1: predicción de demanda diaria por estación

**Proyecto:** MiBici, Equipo 19 · **Estado revisado:** 24-11-2026 · **Tipo:** regresión supervisada.
**Modelo servido en el snapshot:** `mibici-equipo19-p1-demanda`, versión **1**, alias `champion` al momento de la exportación (04-11-2026). **Algoritmo:** Ridge (`alpha=10`) con preprocesamiento de variables temporales y estación.

## Objetivo y uso previsto

Estimar la cantidad de viajes iniciados en una estación durante un día, para planificación operativa o análisis académico. Las entradas incluyen `Año`, `Mes`, `Dia`, `Dia_semana` y `origin_station_id`. La API utiliza `POST /v1/predict/p1`. La salida es una predicción numérica; no se garantiza que sea un entero ni que esté limitada a valores positivos. No se debe interpretar como demanda real confirmada ni como una decisión autónoma de redistribución.

## Datos, partición y trazabilidad

Origen: datos abiertos MiBici (Guadalajara, México), procesados con `dataset.py`, `features/base.py` y `features/p1.py`. El manifiesto `references/data_manifest.json` documenta 24 meses de enero 2024 a diciembre 2025, versión de datos `a8ebbf8d8375`; no se adjuntan los viajes originales al ZIP. Entrenamiento: enero 2024–junio 2025; validación: julio–septiembre 2025; prueba: octubre–diciembre 2025. Semilla general 42. La preparación cuenta pares estación/día **con viajes observados**, no todas las combinaciones posibles con cero viajes, lo que limita su interpretación para estaciones sin movimiento.

**Linaje SQLite local:** run `ba2b1e2923bf4565951150bb0fc5f107`, experimento `mibici-equipo19`, commit indicado en tags `3cc78723b23b2cb3e030b5d45c23f3ed03b33395`. El snapshot OCI incorpora firma MLflow y SHA256 en `deployment/model_store/manifest.json`. Para confirmar el modelo registrado en un entorno distinto, ejecutar `python -m tc5061_proyecto_equipo19.modeling.verify_models` contra el Registry original; el snapshot no garantiza que ese servidor siga accesible.

## Evaluación observada en MLflow local

| Métrica | Validación 2025 | Prueba 2025 |
|---|---:|---:|
| MAE (viajes por estación/día) | 8.2883 | 9.6215 |
| R² | 0.8375 | 0.8111 |

Los valores provienen de métricas del run original en `mlflow.db`. La mejora de MAE sobre regresión lineal en validación fue modesta (aprox. 0.58 %), conforme a `reports/seleccion_modelos_plantilla.md`. El rendimiento en periodos futuros puede ser distinto.

**Prueba A/B histórica posterior:** el SQLite conserva una comparación sobre septiembre 2026 entre P1 v1 (`champion`) y P1 v2 (`candidate_e8`). MAE de A = **15.3792**, de B = **11.9050** (mejora relativa ≈ **22.6 %**), RMSE de A = **22.5561**, de B = **19.9095**, e IC95 % bootstrap de ΔMAE (B−A) = **[-3.7461, -3.1860]**. Es una evaluación **offline pareada**, no asignación aleatoria en producción. **El alias `champion` sigue en v1** según el SQLite recibido; no se promueve automáticamente v2. Para auditoría completa faltan los reportes `resultado_ab.json` y `evidently_drift_calidad.html`.

## Riesgos, limitaciones y salvaguardas

- Variación estacional, crecimiento/retiro de estaciones y cambios de patrones pueden deteriorar el error; el componente `Año` siempre cambia entre periodos, por lo que su drift no basta para declarar alarma.
- El preprocesador depende de categorías de estación vistas en el entrenamiento. Las estaciones nuevas deben analizarse antes de aceptar su desempeño.
- No incorpora clima, eventos masivos, bloqueos o mantenimiento. No estima demanda no observada en días sin viajes.
- Validar formato y rango de las peticiones; evaluar errores por estación y periodo antes de decisiones operativas.
- No usar la predicción para decisiones individuales, vigilancia ni identificación de usuarios.

## Monitoreo, actualización y responsable

Evidently compara referencia 2024-07/09 con actual 2026-07/09; ejecutar `python -m tc5061_proyecto_equipo19.monitoring drift`. Para reentrenar y contrastar: `monitoring train`, `monitoring ab`, `monitoring report` (se necesita restaurar Parquet y acceder al Registry). Promover únicamente con revisión humana y `monitoring promote --approve`, reexportar snapshots y redesplegar. Responsable del artefacto y decisión de promoción: **Equipo 19**, con aprobación manual documentada. Estado: modelo académico empaquetado; **no se acredita explotación real de MiBici**.
