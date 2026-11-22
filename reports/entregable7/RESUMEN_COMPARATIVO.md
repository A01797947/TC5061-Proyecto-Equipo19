# Entregable 7 — Comparativo real de pruebas de carga y FinOps

Resultados generados exclusivamente a partir de los archivos `*_resumen.json`
y de las gráficas producidas por Locust y el monitor de recursos. Las cifras
**son mediciones locales**; la tarifa cloud es una hipótesis de costo.

| Métrica | antes_u100 | despues_u100 | antes_u200 | despues_1w_u200 | despues_u200 | despues_u300 | despues_u400 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Usuarios | 100 | 100 | 200 | 200 | 200 | 300 | 400 |
| Duración configurada (s) | 90 | 90 | 90 | 90 | 90 | 90 | 90 |
| Predicciones exitosas | 5,510 | 5,662 | 9,335 | 10,764 | 11,270 | 16,322 | 19,442 |
| RPS exitosos | 61.222 | 62.911 | 103.722 | 119.600 | 125.222 | 181.356 | 216.022 |
| Latencia media (ms) | 52.27 | 11.94 | 329.73 | 91.19 | 17.22 | 67.71 | 257.59 |
| Latencia p95 (ms) | 180.00 | 29.00 | 1300.00 | 490.00 | 57.00 | 320.00 | 1100.00 |
| Latencia p99 (ms) | 700.00 | 82.00 | 1700.00 | 920.00 | 210.00 | 840.00 | 1500.00 |
| Errores (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CPU API promedio (%) | 60.84 | 39.92 | 124.40 | 74.93 | 68.30 | 116.65 | 160.13 |
| RAM API máxima (MiB) | 252.06 | 484.94 | 260.54 | 258.73 | 501.92 | 519.75 | 533.39 |
| USD / inferencia exitosa | 0.000000379 | 0.000000369 | 0.000000224 | 0.000000194 | 0.000000185 | 0.000000128 | 0.000000107 |
| USD / 1000 inferencias | 0.000379 | 0.000369 | 0.000224 | 0.000194 | 0.000185 | 0.000128 | 0.000107 |

## Evidencias gráficas

### antes_u100 (100 usuarios)

![antes_u100_grafico_latencia.png](antes_u100_grafico_latencia.png)
![antes_u100_grafico_rps_errores.png](antes_u100_grafico_rps_errores.png)
![antes_u100_grafico_recursos.png](antes_u100_grafico_recursos.png)

[Reporte HTML Locust](antes_u100_locust.html) · [Detalle FinOps](antes_u100_resumen.md)

### despues_u100 (100 usuarios)

![despues_u100_grafico_latencia.png](despues_u100_grafico_latencia.png)
![despues_u100_grafico_rps_errores.png](despues_u100_grafico_rps_errores.png)
![despues_u100_grafico_recursos.png](despues_u100_grafico_recursos.png)

[Reporte HTML Locust](despues_u100_locust.html) · [Detalle FinOps](despues_u100_resumen.md)

### antes_u200 (200 usuarios)

![antes_u200_grafico_latencia.png](antes_u200_grafico_latencia.png)
![antes_u200_grafico_rps_errores.png](antes_u200_grafico_rps_errores.png)
![antes_u200_grafico_recursos.png](antes_u200_grafico_recursos.png)

[Reporte HTML Locust](antes_u200_locust.html) · [Detalle FinOps](antes_u200_resumen.md)

### despues_1w_u200 (200 usuarios)

![despues_1w_u200_grafico_latencia.png](despues_1w_u200_grafico_latencia.png)
![despues_1w_u200_grafico_rps_errores.png](despues_1w_u200_grafico_rps_errores.png)
![despues_1w_u200_grafico_recursos.png](despues_1w_u200_grafico_recursos.png)

[Reporte HTML Locust](despues_1w_u200_locust.html) · [Detalle FinOps](despues_1w_u200_resumen.md)

### despues_u200 (200 usuarios)

![despues_u200_grafico_latencia.png](despues_u200_grafico_latencia.png)
![despues_u200_grafico_rps_errores.png](despues_u200_grafico_rps_errores.png)
![despues_u200_grafico_recursos.png](despues_u200_grafico_recursos.png)

[Reporte HTML Locust](despues_u200_locust.html) · [Detalle FinOps](despues_u200_resumen.md)

### despues_u300 (300 usuarios)

![despues_u300_grafico_latencia.png](despues_u300_grafico_latencia.png)
![despues_u300_grafico_rps_errores.png](despues_u300_grafico_rps_errores.png)
![despues_u300_grafico_recursos.png](despues_u300_grafico_recursos.png)

[Reporte HTML Locust](despues_u300_locust.html) · [Detalle FinOps](despues_u300_resumen.md)

### despues_u400 (400 usuarios)

![despues_u400_grafico_latencia.png](despues_u400_grafico_latencia.png)
![despues_u400_grafico_rps_errores.png](despues_u400_grafico_rps_errores.png)
![despues_u400_grafico_recursos.png](despues_u400_grafico_recursos.png)

[Reporte HTML Locust](despues_u400_locust.html) · [Detalle FinOps](despues_u400_resumen.md)

## Discusión y limitaciones (completar con interpretación del equipo)

- **Desempeño:** describir tendencias en p95/p99, RPS y errores por concurrencia.
- **Recursos:** explicar relación de CPU y RAM con variaciones de latencia.
- **FinOps:** comparar USD/inferencia de los escenarios; distinguir costo marginal,
  utilización y costo de inactividad. El supuesto representa una VM completa
  para API + MLflow, pero excluye red, almacenamiento y costo del generador Locust.
- **Validez externa:** resultados en Podman/WSL2 no acreditan igual throughput
  en AWS; confirmar tarifa y tamaño antes de tomar decisiones de producción.

## Evidencia GitHub (completar manualmente)

- URL del repositorio: PENDIENTE
- URL de ejecución GitHub Actions con todos los jobs en verde: PENDIENTE
