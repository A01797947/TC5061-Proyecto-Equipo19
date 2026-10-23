# P2. Riesgo de que una estación se quede sin bicicletas o sin anclajes

**Pregunta.** ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en las próximas tres horas?

**Tipo de tarea.** Clasificación binaria por estación × hora de emisión. Código: `tc5061_proyecto_equipo19/features/p2.py` (datos y variables), `tc5061_proyecto_equipo19/modeling/train_p2.py` (modelos, métricas y MLflow) y `tc5061_proyecto_equipo19/config.py` (parámetros de la aproximación del riesgo). Fechas, particiones, semilla y MLflow vienen de `config.py`, y el registro de cada run usa `modeling/mlflow_utils.py`.

## 1. Formulación

| | |
|---|---|
| Qué se predice | Probabilidad de que la estación *s* quede vacía (≤ 10 % de su capacidad) o llena (≥ 90 %) en algún momento de la ventana *h+1* a *h+3*, con la información disponible al cierre de la hora *h* |
| Granularidad y horizonte | Estación × hora; se emite cada hora de 6:00 a 20:00; horizonte de 3 horas, el tiempo que necesita un camión de redistribución para llegar |
| Por qué importa | Una estación vacía es un viaje perdido; una estación llena obliga al usuario a buscar otra para devolver la bici y le genera cargos por tiempo. Ambos casos son las quejas principales de un sistema de bici pública |
| Quién lo usa | Supervisores de redistribución. Cada hora reciben la lista de estaciones ordenada por riesgo y mandan camión a las *k* primeras que su flota puede atender, llevando bicis a las que se vacían y retirándolas de las que se llenan |

## 2. Aproximación del riesgo (justificación)

Los datos abiertos de MiBici traen viajes, no el inventario de bicis por estación, así que la ocupación real no se observa. Se reconstruye una **ocupación estimada** con el balance de viajes:

```
ocupacion_est(s, 5:00) = 0.5
ocupacion_est(s, t)    = clip( ocupacion_est(s, t-1) + (llegadas(s, t) - salidas(s, t)) / capacidad(s), 0, 1 )

riesgo_vacia(s, h) = min_{k=1..3} ocupacion_est(s, h+k) ≤ 0.10
riesgo_llena(s, h) = max_{k=1..3} ocupacion_est(s, h+k) ≥ 0.90
y(s, h)            = riesgo_vacia o riesgo_llena
```

Cada supuesto y por qué se eligió:

1. **Reinicio diario a las 5:00 con 50 % de ocupación.** El servicio arranca a las 5:00 y el operador rebalancea durante la noche, cuando casi no hay viajes. Partir de la mitad de la capacidad es el supuesto neutral: no favorece ni el vaciado ni el llenado. Reiniciar cada día evita que el error de la estimación se acumule durante semanas.
2. **Balance relativo a la capacidad.** Diez salidas netas vacían una estación de 12 anclajes pero apenas mueven una de 30. Dividir entre la capacidad (`dpcapacity` del catálogo) hace comparables estaciones de distinto tamaño.
3. **Cota entre 0 y la capacidad.** Una estación no puede tener menos de cero bicis ni más que anclajes. Si los viajes observados la empujarían fuera de ese rango es porque el operador repuso o retiró bicis; la estimación se queda en el límite y sigue desde ahí. Sin la cota, una estación que se vació a las 8:00 quedaría marcada en riesgo el resto del día aunque reciba llegadas.
4. **Umbrales de 10 % y 90 %, no 0 % y 100 %.** El riesgo operativo empieza antes de que la estación se vacíe por completo: con una o dos bicis ya no alcanza para el siguiente pico. Además dejan margen al error de la estimación.
5. **Riesgo bidireccional en una sola etiqueta.** La decisión es la misma (mandar camión); lo que cambia es la carga. Se conservan `y_riesgo_vacia` y `y_riesgo_llena` para medir el *recall* de cada tipo y para que el supervisor sepa qué llevar.
6. **Ventana dentro del mismo día de servicio.** Se emite de 6:00 a 20:00 para que *h+3* no pase de las 23:00. Así ninguna etiqueta mezcla dos días y los cortes de la partición, a medianoche, no se contaminan.

**Limitaciones declaradas.** No se observa la redistribución durante el día ni las bicis fuera de servicio; la capacidad viene de un catálogo de 2026-09 posterior al periodo. Por eso la etiqueta mide *presión de demanda neta sobre la capacidad* más que inventario real. Los umbrales y la ocupación inicial viven en `config.py` y se registran como parámetros en cada run; si la prevalencia observada se aleja mucho del rango útil (5 % a 30 %), se ajustan ahí. Esta definición amplía la de la propuesta (§1 y §2.3), que solo cubría el vaciado por flujo neto de tres horas.

## 3. Pipeline

| Etapa | Función | Detalle |
|---|---|---|
| Carga | `cargar_viajes_p2` | Lee los Parquet mensuales de la ingesta, solo inicio, fin, origen y destino |
| Limpieza | `mascara_viajes_inconsistentes` (features/base.py) |Se leen los Parquet mes por mes y se descartan los viajes cuyo inicio cae fuera del mes de su archivo, con duración negativa o mayor a 24 h. Los conteos quedan como parámetros del run |
| Agregación | `construir_flujos_horarios` | Salidas por estación de origen y hora de inicio; llegadas por estación de destino y hora de fin; matriz estación × hora completa con ceros. Se usan las estaciones del catálogo con capacidad positiva |
| Etiqueta | `estimar_ocupacion`, `construir_dataset_p2` | Ocupación estimada y etiqueta de la sección 2 |
| Variables | `construir_dataset_p2` (features/p2.py), `agregar_calendario` (features/base.py)| Ver tabla siguiente. Se descartan las filas antes del primer viaje y después del último de cada estación, para no inflar los negativos con estaciones que aún no existían |
| Partición | `dividir_particiones`(features/base.py) | Entrenamiento 2024-01 a 2025-06, validación 2025-07 a 2025-09, prueba 2025-10 a 2025-12 |
| Baselines y modelos | `run_model` | Sección 4 |
| Umbral | `choose_threshold` | Maximiza F2 en validación y se aplica sin cambios a prueba |
| Registro | `main` | Un run de MLflow por modelo |

**Variables** (todas usan información hasta el cierre de *h*):

| Grupo | Variables | Por qué |
|---|---|---|
| Calendario | `hora`, `hora_sin`, `hora_cos`, `dia_semana`, `fin_de_semana`, `feriado`, `mes` | El vaciado y el llenado siguen los viajes al trabajo y a la escuela: picos de mañana y tarde entre semana, otro patrón en fin de semana y feriados |
| Estación | `capacidad`, `latitud`, `longitud` | Una estación chica se vacía con menos viajes; la ubicación separa zonas residenciales (se vacían en la mañana) de zonas de empleo (se llenan) |
| Estado actual | `ocupacion_est_h`, `neto_h`, `neto_h_1`, `neto_ult_3h`, `salidas_h`, `llegadas_h` | El riesgo es acumulativo: lo cerca que está hoy del umbral y la tendencia reciente son el mejor indicio de las siguientes horas |
| Patrón histórico | `neto_ventana_ayer`, `neto_ventana_semana_pasada`, `ocup_min_ventana_semana_pasada`, `ocup_max_ventana_semana_pasada`, `riesgo_semana_pasada` | Lo que pasó en la misma ventana ayer y hace una semana; esas horas ya ocurrieron al momento de predecir |
| Sistema | `salidas_sistema_h`, `actividad_sistema_vs_semana_pasada` | Detectan días atípicos (lluvia, feriados, eventos) que cambian a toda la red |

## 4. Modelos y métricas

| Run en MLflow | Tipo | Qué hace |
|---|---|---|
| `P2-baseline-tasa-historica` | Baseline | Proporción de horas en riesgo de esa estación, a esa hora y tipo de día (entre semana o fin de semana), calculada solo con entrenamiento |
| `P2-baseline-regla-ocupacion` | Baseline | Regla de persistencia: el puntaje crece conforme la ocupación estimada actual se acerca a 10 % o a 90 % |
| `P2-candidato-logreg-balanceada` | Candidato | Imputación por mediana, `StandardScaler` y `LogisticRegression(class_weight="balanced")` |
| `P2-candidato-hist-gradient-boosting` | Candidato | `HistGradientBoostingClassifier` (300 iteraciones, 63 hojas, `random_state=42`); capta interacciones como hora × ubicación sin construirlas a mano |

| Métrica | Función de scikit-learn | Por qué es coherente con la tarea y la decisión |
|---|---|---|
| **Average precision** (principal) | `average_precision_score` | Resume la curva precisión-recall sin fijar umbral. Con clase positiva minoritaria es más informativa que ROC-AUC, que se infla con la gran cantidad de horas sin riesgo. Se compara contra la prevalencia, que es la AP de un modelo al azar |
| Precisión y recall en top-*k* por hora (*k* = 10) | Implementada sobre `precision_score` | Es la decisión real: cuántas de las 10 estaciones a las que se manda camión estaban de verdad en riesgo, y qué parte de las estaciones en riesgo quedó cubierta |
| Recall, precisión, F1 y F2 con el umbral de validación | `recall_score`, `precision_score`, `f1_score`, `fbeta_score` | El umbral maximiza F2 porque no anticipar una estación vacía o llena cuesta más que un viaje de camión de más |
| Recall por tipo (`recall_vacia`, `recall_llena`) | `recall_score` | Verifica que el modelo no atienda solo un lado del problema |
| ROC-AUC | `roc_auc_score` | Capacidad de ordenamiento, como referencia |
| Brier y log loss | `brier_score_loss`, `log_loss` | El supervisor lee probabilidades; estas métricas y la curva de calibración dicen si "70 %" significa 70 % |

## 5. Registro en MLflow

Experimento `mibici-equipo19`, un run por modelo, ejecutado con `python -m tc5061_proyecto_equipo19.modeling.train_p2` desde la raíz del repositorio.

- **Parámetros:** rango de fechas, cortes de la partición, semilla, horizonte, horas de emisión, ocupación inicial, umbrales, *k*, número de filas por conjunto y de estaciones, `data_version` (SHA-256 de los `output_sha256` ordenados del manifiesto, estable entre ingestas), `data_schema_version`, archivos y registros de la ingesta, conteos de la limpieza e hiperparámetros del modelo.
- **Métricas:** las de la sección 4 para entrenamiento, validación y prueba (`train_`, `valid_`, `test_`) y el umbral elegido.
- **Etiquetas:** `equipo=equipo19`, `pregunta=P2`, `tarea=clasificacion`, `modelo`, `tipo_modelo` (baseline o candidato), `git_commit`, `git_branch`, `git_dirty` y `data_version`. La descripción del run explica el modelo.
- **Artefactos:** el manifiesto de datos (`datos/`) y, en `evaluacion/`, curva precisión-recall de validación y prueba, matriz de confusión, curva de calibración, precisión en top-*k* por hora, importancia de variables (coeficientes o permutación) y `metricas.json`. Los candidatos registran además el modelo serializado con firma y ejemplo de entrada.
- Al terminar se guardan `reports/figures/p2/comparacion_modelos.csv` y `.png`.

## 6. Cómo ejecutarlo

```bash
pip install -r requirements.txt
python -m tc5061_proyecto_equipo19.dataset             # ingesta: descarga, Parquet en data/processed y manifiesto
python -m tc5061_proyecto_equipo19.features.p2          # tabla de P2 (data/processed/p2_features.parquet)
python -m tc5061_proyecto_equipo19.modeling.train_p2    # 4 runs en MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db       # revisar y comparar runs
```

Opciones de `train_p2`: `--models hgb` para correr un solo modelo, `--rebuild` para reconstruir la tabla y `--max-train-rows N` para cambiar la muestra de entrenamiento de los candidatos (por omisión `MAX_FILAS_ENTRENAMIENTO` de `config.py`; `0` usa todo). Conviene correrlo con el código ya en un commit, para que `git_commit` identifique exactamente la versión ejecutada (`git_dirty=false`).
