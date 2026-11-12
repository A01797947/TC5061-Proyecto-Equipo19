# Entregable 8 — Observabilidad de MiBici (P1)

## Alcance y fundamentos

Se conserva íntegra la API de despliegue y los flujos P1/P2/P3 anteriores. Esta fase implementa **P1**: demanda diaria de viajes por estación de salida, cuya versión `champion` actualmente está registrada en MLflow bajo el nombre `mibici-equipo19-p1-demanda`. Evidently permite identificar cambios en entradas y etiqueta, mientras que el A/B retrospectivo identifica diferencias reales de error sobre un conjunto futuro común.

| Componente | Periodos | Justificación |
|---|---|---|
| Referencia de drift | 2024-07, 2024-08, 2024-09 | Subperiodo del entrenamiento original (2024-01 hasta 2025-06), con estaciones y datos históricos |
| Periodo actual | 2026-07, 2026-08, 2026-09 | Datos posteriores, comparados en los mismos meses para reducir sesgo estacional |
| Entrenamiento candidato | 2024-01 a 2025-06 **más** 2026-07 y 2026-08 | Reentrena Ridge con historia y dos meses recientes, excluye el holdout |
| A/B offline | 2026-09 | Mes cerrado independiente, se evalúan las versiones P1 sobre las **mismas filas** |

La referencia **no** incluye el 2025-07..12 (se reservaron como validación/test originales). El candidato reutiliza el entrenamiento histórico más datos nuevos. La producción se aproxima con datos abiertos posteriores; **no son logs de llamadas de la API**. Esto se documenta para no presentar la simulación como telemetría de tráfico real.

La variable `Año` cambiará necesariamente entre referencia y actual: se incluye en el reporte porque sí es entrada del modelo, pero se trata de un cambio de contexto y se excluye de la alerta accionable. `origin_station_id` se considera categórico, porque estaciones nuevas/desaparecidas pueden afectar la distribución. `cantidad_viajes` es la etiqueta observada; solo se conoce con posterioridad al día evaluado. Aun tras drift, no se puede afirmar una pérdida de precisión sin evaluar ground truth.

## Requisitos para ejecutar desde PowerShell

Usa el entorno `(mlops_tec)` y ubícate **en la raíz** del repositorio MiBici, donde se encuentran `MLproject` y `pyproject.toml`.

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-monitoring.txt
python -m pip check
python -m pytest tests/test_monitoring_e8.py -q
```

La instalación incorpora Evidently **0.7.23** y `python-docx`. Se utiliza la API Evidently 0.7 (`Report.run(current, reference)`, `snapshot.save_html`), no `evidently.legacy`. Si la resolución de dependencias falla con Python 3.14, conserva el mensaje completo de `pip` para evaluar una variante de entorno antes de modificar tus librerías fijadas. No se necesita configurar cuenta de Evidently Cloud.

**Datos restaurados:** el ZIP compartido omitió todos los `trips_*.parquet`. Debes volver a copiar, como mínimo, los archivos en `data/processed` correspondientes a **todos los meses entre 2024-01 y 2025-06** y a **2026-07, 2026-08, 2026-09**. También debe estar disponible `references/data_manifest.json`. No hace falta generar el Parquet intermedio masivo ni estaciones para P1. Los archivos de 2026 pueden faltar aunque el manifiesto original solo abarque 2024-2025: esta actividad genera su huella de datos independiente.

Si ya tienes los CSV descargados en `data/raw/` pero todavía no los Parquet 2026, usa:

```powershell
python -m tc5061_proyecto_equipo19.monitoring prepare-raw --months 2026_07,2026_08,2026_09
```

Para convertir también los históricos puedes ejecutar `prepare-raw` especificando los meses correspondientes (se leen CSV locales). Si has copiado los Parquet del ordenador original, no necesitas volver a ingerirlos. **No descarga datos, no reconstruye de forma artificial meses faltantes y no los reemplaza por otros meses sin avisar.**

**MLflow:** es necesario que el `MLFLOW_TRACKING_URI` de tu entorno acceda al Registry con `mibici-equipo19-p1-demanda` y que los artefactos de su versión `champion` se puedan cargar. La versión 1 y su alias aparecen en el SQLite entregado; sus artefactos originales pueden estar en tu instalación local, servidor o almacenamiento remoto según dónde entrenaste. La versión 1 que acompaña al ZIP es un *snapshot de despliegue*, y no se presenta como sustituto de la versión registrada. Si no puedes cargar `models:/mibici-equipo19-p1-demanda/1` debes restaurar la ruta de artefactos o usar el tracking server correcto; no se crea una falsa versión 1.

## Ejecución paso a paso

**Paso 1 — generar Evidently.**

```powershell
python -m tc5061_proyecto_equipo19.monitoring drift
```

Genera `reports/entregable8/evidently_drift_calidad.html`, `evidently_snapshot.json`, `drift_por_variable.csv`, `calidad_datos.csv`, `resumen_drift.json`. Evidently usa las columnas P1 y `cantidad_viajes`, distancia Jensen–Shannon con umbral **0.10**, ratio de drift agregado **0.5**. Para controlar tiempos con series extensas, calcula el drift sobre hasta 10 000 filas estación-día por periodo y semilla 42, mientras la tabla de nulos/cardinalidad se calcula sobre el conjunto completo. Un `0.10` no es un p-valor y no debe leerse como un 10% de diferencia.

Puedes incrementar el tamaño de muestra (`--max-rows 25000`) si el equipo desea comprobar estabilidad de las conclusiones. El muestreo es reproducible, sin reposición. Conservar los dos conteos originales y la muestra utilizada evita tergiversar el número de observaciones.

**Paso 2 — nueva versión en MLflow.**

```powershell
python -m tc5061_proyecto_equipo19.monitoring train --alpha 10
```

El pipeline Ridge reutiliza `build_pipeline()` de P1 (variables de fecha + OneHotEncoder de estación). Entrena solo con datos anteriores al holdout, hace `log_modelo_sklearn()` con firma/input_example, guarda la huella SHA256 de los 20 meses empleados y registra en **el mismo nombre** del modelo P1. Si ya existe v1, se crea v2; si vuelves a entrenar, la numeración puede ser v3 o superior. Se crea el alias **`candidate_e8`** apuntando a la versión recién creada, sin tocar `champion`. El resultado queda en `registro_candidato.json`.

**Paso 3 — A/B offline sobre septiembre 2026.**

```powershell
python -m tc5061_proyecto_equipo19.monitoring ab
```

Se utilizan `models:/mibici-equipo19-p1-demanda/@champion` y `@candidate_e8` resueltos por el Registry como versiones fijas en la ejecución. Si deseas contrastar versiones numéricas concretas:

```powershell
python -m tc5061_proyecto_equipo19.monitoring ab --base-version 1 --candidate-version 2
```

Se calculan MAE, RMSE y R², se registran las métricas y gráficos comparativos como nueva corrida de evaluación en MLflow (sin subir datos individuales), y se hace bootstrap **pareado por día** (2000 réplicas, semilla 42) de **ΔMAE = MAE(candidato) - MAE(champion)**, para reconocer la dependencia entre estaciones del mismo día. Criterio previo: recomendar al candidato solo si **mejora MAE al menos 2%**, **el límite superior del IC95% es negativo** y **el RMSE no empeora**. El script nunca promueve automáticamente. Esto es evaluación histórica comparada, no A/B online con asignación aleatoria, telemetría ni estudio causal.

**Paso 4 — informe con resultados y gráficos.**

```powershell
python -m tc5061_proyecto_equipo19.monitoring report
```

Genera `reports/entregable8/informe_entregable8.md` y `reports/entregable8/informe_entregable8.docx`. Incorpora tabla de scores *extraídos del snapshot de Evidently*, gráficos derivativos de sus métricas y gráficos del A/B. Para cumplir la exigencia de **gráficas originales de Evidently**, abre también `evidently_drift_calidad.html` en Chrome y adjunta capturas de la distribución de `origin_station_id` y `cantidad_viajes` al informe Word cuando prepares la versión final. El HTML original se entrega siempre como evidencia verificable.

**Atajo (opcional):**

```powershell
python -m tc5061_proyecto_equipo19.monitoring all
```

Ejecuta drift → entrenamiento (crea otra versión cada vez) → A/B → informe. Por control de versiones se recomienda el flujo paso a paso al hacer entregas definitivas.

**Promoción solo si fue recomendada y el equipo la aprueba:**

```powershell
python -m tc5061_proyecto_equipo19.monitoring promote --approve
```

Se verifica que la recomendación persistida sea `candidate`, que la versión exista, que provenga del entregable y que `champion` no haya cambiado desde el A/B. **No ejecutes este paso antes de revisar los resultados.** El Alias `champion` es utilizado por el despliegue exportado en su momento; para que el servicio en contenedor use la nueva versión, posteriormente se debe ejecutar el proceso de exportación/build de snapshots y desplegar nuevamente. Promover el alias en Registry no reemplaza por sí solo el snapshot empaquetado.

## Política de operación (Continuous Training)

1. **Periódicamente** (mensual): cuando cierre un nuevo mes, verificar integridad, distribución de entradas y demanda con Evidently; conservar los HTML/JSON en carpeta con fecha.
2. **Activador investigable**: drift persistente de variables operativas (no exclusivamente Año), caída observada de desempeño en el siguiente mes con etiquetas confiables o cambios anómalos de cardinalidad/nulos.
3. **Candidato**: reentrenar en histórico + meses cerrados; nunca con el siguiente holdout, documentar hiperparámetros y hashes en MLflow.
4. **Validación**: comparar con la versión champion en mismas fechas/filas; utilizar MAE/RMSE/R², incertidumbre, estabilidad de predicciones e indicadores operativos (latencia y errores de API).
5. **Decisión**: aprobar manualmente promoción solo con evidencia; retener champion hasta verificar Registry, pruebas y plan de rollback.

No se afirma que existe una alerta activa ni que hubo mejora hasta ejecutar los scripts con datos reales y comprobar la evidencia.

## Archivos nuevos/modificados

- `tc5061_proyecto_equipo19/monitoring/data.py`: carga mensual y agregación consistente con P1, SHA256 del dato.
- `.../monitoring/drift.py`: Evidently 0.7, calidad completa y captura de scores por variable.
- `.../monitoring/ab.py`: Ridge alternativo, nuevo registro MLflow, evaluación pareada, bootstrap y promoción protegida.
- `.../monitoring/report.py`: informe Markdown/Word con resultados del usuario.
- `.../monitoring/__main__.py`: CLI para PowerShell.
- `tests/test_monitoring_e8.py`: test temporal, parser Evidently, estadística, limpieza y calidad.
- `requirements-monitoring.txt`: dependencias extra de esta entrega.
- `Makefile`: atajos `monitoring-drift`, `monitoring-train`, `monitoring-ab`, `monitoring-report` y `monitoring-all`.
- `.gitignore`: conserva informe, HTML, gráficos y CSV resumidos; omite el snapshot JSON de gran tamaño y predicciones fila a fila (adjúntalos aparte si te los solicitan).

Fuentes: [Evidently quickstart](https://docs.evidentlyai.com/quickstart_ml), [Evidently Drift](https://docs.evidentlyai.com/metrics/preset_data_drift), [Evidently Customize Drift](https://docs.evidentlyai.com/metrics/customize_data_drift), [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry), [MiBici datos abiertos](https://mibici.net/es/datos-abiertos/).
