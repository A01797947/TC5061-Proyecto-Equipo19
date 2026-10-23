# Selección y promoción de modelos MiBici — Equipo 19

**Estado:** selección técnica propuesta a partir de `model_comparison.json`; registro en Model Registry y comprobación de producción pendientes de ejecución local.

**Fuente de resultados:** comparación exportada de MLflow (12 corridas, cuatro por pregunta). La selección se realiza **por validación**; los resultados de prueba se consignan como evaluación complementaria, no como criterio de ajuste de hiperparámetros.

## 1. Entorno, datos y trazabilidad

| Elemento | Valor / verificación |
|---|---|
| Equipo y experimento | `equipo19` / `mibici-equipo19` (comprobar nombre exacto en MLflow) |
| Versión de datos | `a8ebbf8d8375`, idéntica en las 12 corridas del JSON |
| Entrenamiento | Enero de 2024 a junio de 2025 |
| Validación | Julio a septiembre de 2025 |
| Prueba | Octubre a diciembre de 2025 |
| Separación | Temporal, sin entrenar con observaciones de validación o prueba |
| Tracking y Registry | MLflow local: `http://127.0.0.1:5000`; SQLite `sqlite:///mlflow.db` según la configuración del proyecto |
| Commit Git y estado de trabajo | **Pendiente:** consultar `git_commit`, `git_dirty` y `equipo` en los tres runs seleccionados |
| Python / MLflow / dependencias exactas | **Pendiente:** incorporar versiones del entorno ejecutado |
| Datos personales | El informe no contiene viajes ni identificadores de personas; conservar solo métricas, agregados y artefactos permitidos |

**Limitación de la evidencia:** el archivo de comparación incluye métricas, tiempos, URIs y versiones de datos, pero no incluye estado de las corridas, commit Git, todas las métricas secundarias, parámetros detallados, firmas de modelos ni estabilidad ARI. Las celdas no disponibles quedan explícitamente indicadas como *pendiente de verificar*, sin asignar valores supuestos.

## 2. P1 — ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro?

**Problema:** regresión de demanda diaria por estación. **Métrica principal:** MAE de validación (menor es mejor). **Baseline:** regresión lineal. Todas las filas reportan `data_version=a8ebbf8d8375`.

| Modelo | Run ID | MAE validación ↓ | MAE prueba ↓ | Tiempo (s) | Decisión |
|---|---|---:|---:|---:|---|
| Linear (baseline) | `94a843314ade4418be1a18cb630f26a4` | 8.3364 | 9.6843 | 32.083 | Referencia |
| Ridge, α=0.1 | `788053f5e10349568009581ebf724589` | 8.3348 | 9.6818 | 25.223 | No elegido |
| Ridge, α=1 | `0a3e3bd09f684f558a9b1b2937eaa5db` | 8.3258 | 9.6672 | 15.780 | No elegido |
| **Ridge, α=10** | **`ba2b1e2923bf4565951150bb0fc5f107`** | **8.2883** | **9.6215** | **15.282** | **Ganador propuesto** |

**Justificación.** Ridge con α=10 consigue el MAE de validación más bajo (8.2883 frente a 8.3364 del baseline), una disminución relativa de aproximadamente **0.58 %**. También obtiene menor MAE de prueba (9.6215 frente a 9.6843) y su tiempo de corrida registrado (15.282 s) es inferior al del baseline (32.083 s). La mejora es **modesta**, por lo que no debe presentarse como un salto sustancial. El coste de inferencia, la distribución de residuales, el R² y la existencia de predicciones de conteos negativos aún requieren revisión. El MAE expresa errores medios en **viajes por estación/día** solo si esta es efectivamente la granularidad del objetivo generado en el script.

**URI de modelo de origen:** `models:/m-eb33489f58ac411c925ced18ade788fe`.

## 3. P2 — ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro?

**Problema:** clasificación de riesgo. **Métrica principal:** `Precision@10` de validación (mayor es mejor), coherente con priorizar un número limitado de estaciones para intervención. **Baseline:** tasa histórica. Todas las filas reportan `data_version=a8ebbf8d8375`.

| Modelo / corrida | Run ID | Precision@10 valid. ↑ | Precision@10 prueba ↑ | Tiempo (s) | Decisión |
|---|---|---:|---:|---:|---|
| Tasa histórica (baseline, sin URI de modelo) | `75a8a52720814b50b657dd79914f7ae4` | 84.58 % | 79.15 % | 2.054 | Referencia; no registrable con el script actual |
| Regresión logística balanceada | `c854e127b65d4ea4afbf2e86966cf347` | 63.76 % | 61.89 % | 30.533 | No elegida |
| HGB, corrida `ec30...` | `ec30d16490dc4b83bd10125d95ead602` | 95.36 % | 94.45 % | 47.418 | Finalista |
| **HGB, corrida `05af...`** | **`05afe261554d40e1a4ca960b341305cb`** | **95.54 %** | **94.76 %** | **46.613** | **Ganador propuesto** |

**Justificación.** La corrida HGB `05afe...` presenta la mayor Precision@10 de validación (**0.9554**) y conserva un valor alto en prueba (**0.9476**). En validación supera al baseline por **10.96 puntos porcentuales** y en prueba por **15.61 puntos porcentuales**. La diferencia entre validación y prueba para HGB es de aproximadamente **0.78 puntos porcentuales**, dato favorable pero insuficiente por sí solo para demostrar estabilidad en diferentes periodos. El mayor coste de entrenamiento (46.613 s frente a 2.054 s del baseline) puede estar justificado cuando la precisión para la priorización operativa es crítica. **Pendiente:** recuperar del MLflow los hiperparámetros de las dos corridas HGB —tienen el mismo nombre y el JSON no permite distinguir sus configuraciones— y añadir recall, average precision, F2 y curva precisión-recall antes de publicar.

**URI de modelo de origen:** `models:/m-52f68f879c024e8e8c5353f3d89cd2c7`.

## 4. P3 — ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad?

**Problema:** clustering de perfiles de estaciones. **Métrica principal reportada:** silhouette de validación (mayor es mejor). Las columnas del baseline (**tres volúmenes de viajes**) son diferentes de las columnas del candidato (**volúmenes y patrones horarios/semanales**): sus silhouette **no son directamente comparables como si usaran un mismo espacio de características**.

| Modelo | Run ID | Silhouette valid. ↑ | Silhouette prueba ↑ | Tiempo (s) | Decisión |
|---|---|---:|---:|---:|---|
| Baseline K-Means k=3 (3 variables de volumen) | `00299dcd3fde4b4f85fb959bd91e1a51` | **0.6087** | **0.6087** | 25.174 | Baseline con mejor silhouette numérico, pero describe sobre todo intensidad/volumen |
| **Candidato K-Means k=3 (perfil enriquecido)** | **`594403c4ced240a2863cde00cfcdd086`** | **0.4941** | **0.5034** | **25.025** | **Ganador propuesto por adecuación al negocio** |
| Candidato K-Means k=4 (perfil enriquecido) | `a0a009c43dc44171bd224a18b4adb134` | 0.3732 | 0.3891 | 27.494 | No elegido |
| Candidato K-Means k=5 (perfil enriquecido) | `dfb90991205449b28a470f25741e13fc` | 0.2871 | 0.3220 | 28.462 | No elegido |

**Justificación.** El baseline presenta el índice silhouette más elevado (0.6087), pero sus tres características son conteos agregados de viajes iniciados, finalizados y totales. Para responder a la pregunta de **patrones de uso** —por ejemplo, diferencias por hora y día de la semana— se propone el candidato enriquecido con **k=3**. Entre los candidatos con el mismo conjunto de variables, k=3 logra la mayor silhouette de validación (0.4941, frente a 0.3732 para k=4 y 0.2871 para k=5), es consistente en prueba (0.5034) y presenta un tiempo de corrida similar al baseline. La calidad de los clústeres no puede determinarse solo por silhouette. **Antes de promoverlo como `champion`, revisar los mapas (`plots/clusters_map.png`), los perfiles (`evaluation/cluster_summary.csv`), los tamaños de grupo y `valid_ari_vs_train` / `test_ari_vs_train`.** Si los grupos no son interpretables, esta selección deberá reconsiderarse.

**Justificación de cambio metodológico:** la implementación actual ajusta K-Means sobre el intervalo de entrenamiento y calcula métricas de validación y prueba sobre perfiles posteriores sin reajustar el modelo. Normaliza conteos por días del periodo; por ello, no mezclar los resultados con corridas históricas entrenadas utilizando los 24 meses completos.

**URI de modelo de origen:** `models:/m-0d136fe3c2fb4473b4555fadbc3316e9`.

## 5. Decisiones y Model Registry

| Pregunta | Modelo elegido | Registered model name | Run ID de origen | Versión en Registry | Alias | Validación de carga |
|---|---|---|---|---|---|---|
| P1 | Ridge α=10 | `mibici-equipo19-p1-demanda` | `ba2b1e2923bf4565951150bb0fc5f107` | Pendiente de registro | `champion` pendiente | Pendiente |
| P2 | HGB, corrida `05afe...` | `mibici-equipo19-p2-riesgo` | `05afe261554d40e1a4ca960b341305cb` | Pendiente de registro | `champion` pendiente | Pendiente |
| P3 | K-Means k=3, features enriquecidas | `mibici-equipo19-p3-grupos` | `594403c4ced240a2863cde00cfcdd086` | Pendiente de registro | `champion` pendiente | Pendiente |

**Estrategia de promoción:** Model Registry con **aliases** (`champion`) en lugar de etapas clásicas. El alias permite actualizar la versión seleccionada sin alterar el URI de consumo. La asignación significa "versión seleccionada para producción en el ejercicio académico"; no constituye prueba de despliegue real en la operación de MiBici. El código de registro documenta experimento, run, commit y versión de datos mediante tags y descripción.

**Comandos exactos:** ver `reports/COMANDOS_REGISTRO.md`, incluido en el paquete de este informe.

## 6. Evidencias y verificaciones pendientes

1. En la interfaz MLflow confirmar que **las tres corridas elegidas están `FINISHED`**, su firma e `input_example` son accesibles y el modelo de cada URI se puede cargar.
2. Confirmar tags de linaje `question`, `equipo`, `git_commit`, `data_version=a8ebbf8d8375`, `logged_model_uri`; anotar SHA completo y `git_dirty`.
3. Verificar la **búsqueda anidada P1**: una corrida padre y tres hijas Ridge vinculadas por `mlflow.parentRunId`.
4. Revisar los artefactos de cada candidata: `plots/`, `data/` (manifiesto), `config/`, modelo con firma, ejemplo de entrada y dataset de entrenamiento referenciado por `mlflow.log_input`.
5. Completar las métricas secundarias del informe: **P1 R²/RMSE y residuales; P2 AP, recall, F2 y matriz de confusión; P3 ARI, mapa, tamaños y perfiles de clúster**. Estas cifras no están en `model_comparison.json`.
6. Para P2, consultar los parámetros de ambas corridas HGB en MLflow y asignarles descripciones inequívocas.
7. Confirmar el criterio temporal y la lógica del objetivo de P1; revisar que la demanda y sus unidades correspondan a la pregunta de negocio.
8. Solo después de estas validaciones, registrar los tres modelos, asignar `champion` y ejecutar `verify_models.py`; anotar versión real y evidencia de carga/predicción.
9. Actualizar README con comandos `mlflow run`, búsqueda anidada, comparación, modelo elegido, registro, alias y forma de cargarlo desde un clon limpio.
10. Entregar evidencias de GitHub: cada integrante abre al menos un Pull Request y revisa otro mediante la rama principal protegida.

## 7. Control de entrega GitHub

| Integrante | PR abierto (enlace) | PR revisado (enlace) |
|---|---|---|
| Integrante 1 | Pendiente | Pendiente |
| Integrante 2 | Pendiente | Pendiente |
| Integrante 3 | Pendiente | Pendiente |
| Integrante 4 | Pendiente | Pendiente |

**Fuentes de la selección:** `model_comparison.json` proporcionado por el equipo, y código `train_P1.py`, `train_p2.py`, `train_P3.py`, `register_models.py` y `verify_models.py` de la actualización local compartida. No se ha conectado a la instancia MLflow del equipo para confirmar firmas, estados ni alias.
