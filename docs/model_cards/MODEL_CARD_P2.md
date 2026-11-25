# Model Card — P2: riesgo estimado de estación vacía o llena

**Proyecto:** MiBici, Equipo 19 · **Estado revisado:** 24-11-2026 · **Tipo:** clasificación binaria.
**Snapshot:** `mibici-equipo19-p2-riesgo`, versión **1**, alias `champion` al exportar. **Familia:** `HistGradientBoostingClassifier` (HGB).

## Objetivo y uso previsto

Estimar si, en la ventana de 1 a 3 horas posteriores a una observación, una estación podría presentar riesgo de quedarse sin bicicletas o sin anclajes. La API `POST /v1/predict/p2` devuelve la clase y, cuando está disponible, la probabilidad estimada. La finalidad es ayudar a **priorizar revisiones operativas**, no confirmar inventario ni ordenar automáticamente movimientos de flota.

## Origen y significado del objetivo

Se usan flujos de viajes públicos y capacidad/catálogos de estaciones para construir un **proxy de ocupación**, porque los datos abiertos **no incluyen el inventario observado**. La lógica supone **50 % de ocupación inicial** después del rebalanceo nocturno, acumula llegadas menos salidas, acota ocupación a [0,1] y define riesgo de vacío con umbral ≤0.10, de lleno con ≥0.90 (ventana siguiente de tres horas). Estos son **supuestos modelados**, no eventos operativos medidos. Ver `tc5061_proyecto_equipo19/features/p2.py`.

Las variables incluyen calendario, hora cíclica, capacidad, latitud/longitud, ocupación estimada, flujos e históricos de estación y sistema. Los datos son de MiBici Guadalajara para 2024-2025; se utiliza partición temporal, con cortes definidos en `config.py`. Manifiesto histórico: `a8ebbf8d8375`. No se almacenan identidades de usuarios en los artefactos del servicio.

**Linaje:** run `05afe261554d40e1a4ca960b341305cb`, commit `3cc78723b23b2cb3e030b5d45c23f3ed03b33395`, versión 1 en SQLite y firma/snapshot SHA256 en `deployment/model_store/manifest.json`.

## Evaluación observada en MLflow local

| Métrica | Validación | Prueba |
|---|---:|---:|
| Precision@10 | 0.9554 | 0.9476 |
| Average precision | 0.8366 | 0.8338 |
| Recall (umbral elegido en validación) | 0.9226 | 0.9255 |
| F2 | 0.8169 | 0.8142 |
| ROC-AUC | 0.9300 | 0.9345 |

El umbral de decisión registrado fue **0.1633**, escogido en validación para optimizar F2. Precision@10 se calcula con una selección top 10 % según la implementación (`train_p2.py`); no significa necesariamente las diez estaciones con mayor riesgo. El rendimiento estima fidelidad al **objetivo proxy**, no precisión contra inventario físico observado.

## Riesgos, límites y supervisión

- La principal incertidumbre es la **validez del proxy de ocupación** (50 % inicial, sin rebalanceos reales, límites y horizonte). Sin inventario observado no puede demostrarse utilidad operacional definitiva.
- Puede haber fuga de información si variables construidas con datos posteriores a la hora de decisión se incorporan en una aplicación real; revisar disponibilidad de cada variable **antes** del despliegue operativo.
- Diferencias entre estaciones, disponibilidad de capacidad actualizada, feriados no contemplados y cambios de infraestructura exigen monitoreo segmentado y reevaluación.
- Un falso negativo podría dejar una estación sin bicicletas/anclajes; un falso positivo genera movimientos innecesarios. No utilizar para decisiones que afecten individualmente a personas.
- Confirmar calibración, matriz de confusión, precisión/recall por segmento y latencia en cada nueva versión; registrar rollback en MLflow.

## Operación y gobernanza

El endpoint síncrono es `POST /v1/predict/p2`; también existe una cola asíncrona Redis/RQ a través de `POST /v1/jobs/predict/p2`. La entrega monitorea drift A/B principalmente en **P1**; no afirmar que P2 cuenta con detección de drift operacional activa. Mantener control humano de la interpretación y promoción del alias. Responsable: **Equipo 19**. Estado: prototipo académico, no integración de inventario real en producción.
