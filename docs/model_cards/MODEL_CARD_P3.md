# Model Card — P3: agrupación de perfiles de estaciones

**Proyecto:** MiBici, Equipo 19 · **Estado revisado:** 24-11-2026 · **Tipo:** clustering no supervisado.
**Snapshot:** `mibici-equipo19-p3-grupos`, versión **1**, alias `champion` al exportar. **Familia:** K-Means, **k=3**, variables enriquecidas de uso.

## Propósito y entradas

Agrupar estaciones según patrones de viajes iniciados/finalizados y distribución de uso por hora y día de la semana, para análisis de operación e hipótesis sobre perfiles de movilidad. El endpoint `POST /v1/predict/p3` recibe un objeto `features` con la firma exportada en MLflow y devuelve el ID numérico de cluster.

**Los números de clúster no son categorías semánticas ni jerarquías de calidad**, y pueden cambiar tras reentrenamiento. El servicio no determina acciones operativas automáticamente. Las coordenadas se usan en construcción/visualización de perfiles; no debe inferirse que el clustering sea necesariamente espacial.

## Datos y trazabilidad

Fuente: viajes y estaciones públicas MiBici 2024-2025, documento `references/data_manifest.json` (`data_version=a8ebbf8d8375`). Una fila de perfil por estación, con conteos y distribuciones horarios/semanales, procesada por `features/p3.py` y entrenada/validada por cortes temporales definidos en `config.py`. El entrenamiento se ajusta sin usar perfiles posteriores para `fit`. Código principal: `modeling/train_P3.py`.

**Linaje:** run `594403c4ced240a2863cde00cfcdd086`, commit `3cc78723b23b2cb3e030b5d45c23f3ed03b33395`, versión 1, firma de entrada y SHA256 en manifiesto del snapshot OCI.

## Evaluación registrada en MLflow local

| Métrica | Resultado |
|---|---:|
| Silhouette (entrenamiento) | 0.5175 |
| Silhouette (validación) | 0.4941 |
| Silhouette (prueba) | 0.5034 |
| ARI validación vs entrenamiento | 0.8632 |
| ARI prueba vs entrenamiento | 0.7819 |
| Estaciones de entrenamiento | 371 |

La comparación con el K-Means baseline (silhouette validación 0.6087) no es directa porque aquel usó **solamente tres variables de volumen**, frente al perfil enriquecido elegido. Entre las alternativas enriquecidas k=3, k=4 y k=5, k=3 obtuvo la mayor silhouette de validación, según la tabla histórica de selección.

## Riesgos, restricciones y gobernanza

- Segmentos sensibles a escalado, cambios en operación de estaciones, periodos temporales y a las variables seleccionadas; revisar estabilidad antes de consumir nuevas etiquetas de clúster.
- ARI evalúa concordancia de agrupaciones a través de periodos, **no** validez comercial ni causalidad de agrupaciones.
- Clústeres pequeños o atípicos pueden quedar mal caracterizados. Revisar los perfiles, proporciones, mapas y tamaños por cluster antes de interpretar su significado.
- No inferir características socioeconómicas, personales o individuales a partir de las estaciones o de su ubicación.
- El entregable de drift Evidently y A/B se concentra en P1; P3 requiere política específica de drift de perfiles y revisión de estabilidad para una operación real.

Responsable: **Equipo 19**, con evaluación y aprobación humana de reentrenamientos y cambios de alias `champion`. Estado: análisis/servicio académico; no se acredita uso operativo real de MiBici.
