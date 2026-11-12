"""Síntesis interpretativa reproducible (Markdown + DOCX) del Entregable 8."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

VARIABLE_EXPLANATIONS = {
    "Año": "Cambio esperado por diseño entre 2024 y 2026; por sí solo NO evidencia degradación.",
    "Mes": "Puede indicar variación de estacionalidad, incluso comparando julio–septiembre.",
    "Dia": "Cambios en días del mes representados y en la mezcla de fechas con viajes.",
    "Dia_semana": "Cambios en patrones de días laborales y fines de semana.",
    "origin_station_id": "Variación del uso y composición de estaciones; vigilar estaciones no vistas.",
    "cantidad_viajes": "Cambio de la distribución de la demanda observada (target), no prueba causal de error.",
}


def _figure_drift(drift: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = drift.sort_values("score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh(ordered["variable"], ordered["score"])
    ax.axvline(float(ordered["umbral"].iloc[0]), linestyle="--", color="black",
               label="Umbral Evidently")
    ax.set_xlabel("Distancia Jensen-Shannon (Evidently)")
    ax.set_title("Drift por variable · métricas exportadas de Evidently")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "drift_por_variable.png", dpi=175)
    plt.close(fig)


def generate_report(out: Path) -> dict:
    out = Path(out)
    summary = json.loads((out / "resumen_drift.json").read_text(encoding="utf-8"))
    ab = json.loads((out / "resultado_ab.json").read_text(encoding="utf-8"))
    drift = pd.read_csv(out / "drift_por_variable.csv")
    quality = pd.read_csv(out / "calidad_datos.csv")
    _figure_drift(drift, out)
    columns = []
    for _, row in drift.iterrows():
        name = str(row["variable"])
        flagged = bool(row["drift"])
        columns.append({"name": name, "score": float(row["score"]), "drift": flagged,
                        "impact": VARIABLE_EXPLANATIONS.get(name, "Revisar cambio de distribución.")})
    main_changed = [r["name"] for r in columns if r["drift"] and r["name"] != "Año"]
    recommendation = (
        "La evidencia favorece promover al candidato, tras validar condiciones operativas "
        "(latencia, integridad y aprobación humana). El alias champion no cambió durante el A/B."
        if ab["recommendation"] == "candidate" else
        "Mantener la versión champion; el A/B no permite defender una mejora robusta. "
        "Revisar drift y ensayar reentrenamiento adicional, sin promoción automática."
    )
    rows = "\n".join(
        f"| {r['name']} | {r['score']:.4f} | {'Sí' if r['drift'] else 'No'} | {r['impact']} |"
        for r in columns
    )
    reference_months = summary.get("reference_months", ["2024_07", "2024_08", "2024_09"])
    current_months = summary.get("current_months", ["2026_07", "2026_08", "2026_09"])
    # Porcentajes de nulos medidos en datos completos (no en muestras de Evidently).
    max_null = quality.groupby("periodo")["porcentaje_nulos"].max().to_dict()
    text = f"""# Entregable 8 — Drift, calidad y comparación A/B (MiBici, P1)

**Equipo 19 · Operaciones de aprendizaje automático**  
**Estado:** reporte generado automáticamente con los Parquet locales y el Registry real de MLflow.

## 1. Objetivo y diseño

Se supervisa **P1: demanda diaria de viajes por estación**, utilizando la misma definición de variables del entrenamiento original. El baseline es la versión **v{ab['base_version']}** del modelo `{ab['model_name']}`. El candidato es la versión **v{ab['candidate_version']}** del mismo modelo registrado.

- **Referencia:** {", ".join(reference_months)}; datos comprendidos en el entrenamiento histórico (enero 2024 a junio 2025).
- **Actual:** {", ".join(current_months)}; datos posteriores de MiBici, mismos meses del calendario para mitigar estacionalidad.
- **Reentrenamiento alternativo:** entrenamiento histórico y julio–agosto de 2026. No utiliza septiembre de 2026.
- **Holdout A/B:** septiembre de 2026. Ambos modelos predicen las mismas {ab['n_rows']:,} filas estación/día sin ajustar hiperparámetros con este mes.

**Limitaciones del diseño:** Es un A/B *offline, retrospectivo y pareado*, no un experimento online aleatorizado. El P1 histórico agrupa solo estaciones/días con viajes registrados; los días con cero viajes no aparecen, lo que limita su representatividad. La estación se codifica con OneHotEncoder(handle_unknown='ignore') y estaciones nuevas pueden perder información. No se asume que drift estadístico implica degradación.

## 2. Reporte de Evidently y calidad

Evidently **0.7.23** ejecuta `DataDriftPreset` y `DataSummaryPreset` con **Jensen–Shannon, umbral 0.10**, y calcula métricas por variable y tests de calidad. Se analizaron **{summary['reference_rows']:,}** filas de referencia y **{summary['current_rows']:,}** actuales; para el análisis de distribuciones se muestrearon **{summary['sample_reference']:,}** y **{summary['sample_current']:,}** respectivamente, con semilla {summary['seed']}.

**Resultado agregado Evidently:** {summary['drifted_columns']} variables con drift; proporción **{summary['drift_share']:.1%}** (drift de conjunto: **{'Sí' if summary['dataset_drift'] else 'No'}** según umbral {0.5:.0%}). El cambio de `Año` es inevitable y se reporta como variable de contexto, no como señal de reentrenamiento por sí sola. Variables alteradas distintas de `Año`: **{', '.join(main_changed) if main_changed else 'ninguna'}**.

![Resultados por columna obtenidos del snapshot Evidently](drift_por_variable.png)

| Variable | Score JS | Drift | Interpretación / posible impacto |
|---|---:|:---:|---|
{rows}

**Calidad:** porcentaje máximo de nulos por variable: referencia **{max_null.get('referencia', 0):.2f}%**, actual **{max_null.get('actual', 0):.2f}%**. Consultar `calidad_datos.csv` para cada columna, cardinalidad y duplicados. **Gráficas interactivas originales de Evidently y tablas de calidad:** `evidently_drift_calidad.html`. Los valores y la gráfica anterior derivan de `evidently_snapshot.json`, exportación de Evidently (no son métricas estimadas externamente). Para la entrega final se pueden incorporar capturas de las distribuciones del HTML, especialmente de estaciones y demanda.

## 3. Comparación A/B: P1 v{ab['base_version']} vs v{ab['candidate_version']}

| Métrica en septiembre 2026 | A: champion v{ab['base_version']} | B: candidato v{ab['candidate_version']} | Interpretación |
|---|---:|---:|---|
| MAE ↓ | {ab['baseline']['mae']:.4f} | {ab['candidate']['mae']:.4f} | Menor es mejor |
| RMSE ↓ | {ab['baseline']['rmse']:.4f} | {ab['candidate']['rmse']:.4f} | Penaliza errores grandes |
| R² ↑ | {ab['baseline']['r2']:.4f} | {ab['candidate']['r2']:.4f} | Mayor es mejor |

![Comparación de errores A/B](comparacion_ab.png)

El cambio relativo de MAE del candidato frente al champion es **{ab['relative_mae_improvement']:.2%}** (positivo significa mejora). Se aplicó bootstrap **pareado por día** con {ab['bootstrap']['replicas']:,} réplicas y semilla {ab['bootstrap']['bootstrap_seed']}; ΔMAE (B−A) = **{ab['bootstrap']['delta_mae']:.4f}**, IC del 95% **[{ab['bootstrap']['ci95_lower']:.4f}, {ab['bootstrap']['ci95_upper']:.4f}]** sobre {ab['bootstrap']['n_days']} días. El criterio predefinido para recomendar B exige reducción de MAE ≥ {ab['min_relative_improvement']:.0%}, límite superior del IC95% < 0 y RMSE no peor.

![Intervalo de confianza de la diferencia](bootstrap_ab.png)

## 4. Recomendación y Continuous Training

**Decisión técnica:** **{'promover candidato' if ab['recommendation']=='candidate' else 'mantener champion'}**. {recommendation}

Como política de **Continuous Training**, conviene ejecutar esta comparación mensualmente: monitorear deriva y nulos en cada nueva carga de datos; investigar anomalías de estación/demanda; si existen cambios accionables persistentes y/o degradación de MAE, reentrenar usando solo periodos ya cerrados; evaluar el nuevo candidato en un holdout posterior e independiente antes de aprobar una promoción. Monitorear también latencia, errores y cobertura en la API. El reporte **no demuestra** que un cambio distributivo sea la causa de un error.

## 5. Evidencias y reproducibilidad

- `evidently_drift_calidad.html`: panel interactivo original de Evidently (distribuciones y calidad).
- `evidently_snapshot.json`, `drift_por_variable.csv`, `resumen_drift.json`, `calidad_datos.csv`: trazabilidad estadística.
- `entrenamiento_candidato.json`, `registro_candidato.json`: meses, hashes y versión registrada en MLflow.
- `predicciones_ab.csv`, `metricas_ab.csv`, `resultado_ab.json`: evidencia sobre las mismas observaciones y procedimiento bootstrap; corrida MLflow A/B: **{ab.get("mlflow_ab_run_id", "no registrada")}**.
- `drift_por_variable.png`, `comparacion_ab.png`, `bootstrap_ab.png`: figuras de esta síntesis.

## 6. Referencias

- Evidently AI. (2026). *Data and ML checks*. https://docs.evidentlyai.com/quickstart_ml
- Evidently AI. (2026). *Data Drift*. https://docs.evidentlyai.com/metrics/preset_data_drift
- MLflow. (2026). *Model Registry*. https://mlflow.org/docs/latest/ml/model-registry
- MiBici. *Datos abiertos*. https://mibici.net/es/datos-abiertos/
- Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly Media.
- Treveil, M., et al. (2020). *Introducing MLOps*. O'Reilly Media.
"""
    path = out / "informe_entregable8.md"
    path.write_text(text, encoding="utf-8")
    docx_file = _generate_docx(out, summary, ab, columns, max_null, recommendation)
    return {"markdown": str(path), "docx": str(docx_file)}


def _generate_docx(out: Path, summary: dict, ab: dict, columns: list,
                   max_null: dict, recommendation: str) -> Path:
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
    except ImportError as exc:
        raise RuntimeError("Para generar Word instala python-docx (requirements-monitoring.txt)") from exc
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.75)
    styles = document.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(10)
    styles["Normal"].paragraph_format.space_after = Pt(6)
    styles["Title"].font.size = Pt(19)
    styles["Title"].font.color.rgb = RGBColor(22, 51, 83)
    for name in ("Heading 1", "Heading 2"):
        styles[name].font.color.rgb = RGBColor(22, 51, 83)
    document.add_heading("Entregable 8", 0)
    document.add_paragraph("Monitoreo de drift y comparación A/B — Proyecto MiBici (P1)", style="Subtitle")
    document.add_paragraph("Equipo 19 | Operaciones de aprendizaje automático | Reporte reproducible")
    document.add_heading("1. Diseño experimental", level=1)
    document.add_paragraph(
        "Referencia: julio–septiembre 2024 (entrenamiento histórico). Actual: julio–septiembre "
        "2026. Candidato: Ridge entrenado con enero 2024–junio 2025 y julio–agosto 2026. "
        "Holdout independiente: septiembre 2026. Es una prueba offline pareada, no un A/B "
        "online aleatorizado. Las estaciones/días con cero viajes no se representan en P1."
    )
    document.add_heading("2. Drift y calidad de datos — Evidently", level=1)
    document.add_paragraph(
        f"Referencia: {summary['reference_rows']:,} filas; actual: {summary['current_rows']:,} filas. "
        f"Muestras de Evidently: {summary['sample_reference']:,} y {summary['sample_current']:,} filas. "
        f"Método Jensen–Shannon, umbral 0.10. Variables con drift: "
        f"{summary['drifted_columns']} ({summary['drift_share']:.1%}); "
        f"drift agregado: {'sí' if summary['dataset_drift'] else 'no'}. "
        "El año cambia deliberadamente entre periodos y es contextual."
    )
    t = document.add_table(rows=1, cols=3)
    t.style = "Light Shading Accent 1"
    for cell, label in zip(t.rows[0].cells, ("Variable", "Score JS / drift", "Implicación")):
        cell.text = label
    for row in columns:
        cells = t.add_row().cells
        cells[0].text = row["name"]
        cells[1].text = f"{row['score']:.4f} / {'Sí' if row['drift'] else 'No'}"
        cells[2].text = row["impact"]
    document.add_picture(str(out / "drift_por_variable.png"), width=Inches(5.95))
    document.add_paragraph("Figura 1. Scores por variable extraídos del snapshot original de Evidently.")
    document.add_paragraph(
        f"Máximo porcentaje de nulos (datos completos): referencia "
        f"{max_null.get('referencia', 0):.2f}%; actual {max_null.get('actual', 0):.2f}%. "
        "Consultar evidently_drift_calidad.html para las visualizaciones interactivas de "
        "Evidently y calidad_datos.csv para el desglose detallado."
    )
    document.add_heading("3. A/B offline: comparación de versiones registradas", level=1)
    document.add_paragraph(
        f"Modelo: {ab['model_name']}. A: v{ab['base_version']}; B: v{ab['candidate_version']}. "
        f"Mismo holdout de septiembre de 2026, n={ab['n_rows']:,} filas estación/día."
    )
    metrics = document.add_table(rows=1, cols=3)
    metrics.style = "Light Shading Accent 1"
    for cell, label in zip(metrics.rows[0].cells, ("Métrica", "A: champion", "B: candidato")):
        cell.text = label
    for key in ("mae", "rmse", "r2"):
        cells = metrics.add_row().cells
        cells[0].text = key.upper()
        cells[1].text = f"{ab['baseline'][key]:.4f}"
        cells[2].text = f"{ab['candidate'][key]:.4f}"
    document.add_picture(str(out / "comparacion_ab.png"), width=Inches(5.95))
    document.add_paragraph("Figura 2. Errores MAE y RMSE sobre el mismo holdout.")
    document.add_picture(str(out / "bootstrap_ab.png"), width=Inches(5.95))
    document.add_paragraph("Figura 3. Intervalo de confianza de ΔMAE (B−A), bootstrap por día.")
    document.add_paragraph(
        f"Mejora relativa MAE: {ab['relative_mae_improvement']:.2%}. "
        f"IC95% ΔMAE [{ab['bootstrap']['ci95_lower']:.4f}, "
        f"{ab['bootstrap']['ci95_upper']:.4f}] con {ab['bootstrap']['replicas']} réplicas. "
        "B se recomienda solo si mejora MAE ≥2%, IC95% enteramente negativo y RMSE no empeora."
    )
    document.add_heading("4. Decisión y Continuous Training", level=1)
    document.add_paragraph(recommendation)
    document.add_paragraph(
        "Política recomendada: ejecutar drift y calidad mensualmente, analizar cambio en "
        "estaciones/demanda, reentrenar ante cambios persistentes o caída de desempeño "
        "y mantener siempre evaluación temporal fuera de muestra y aprobación antes de "
        "cambiar el alias champion. Drift de variables no prueba causalidad sobre error."
    )
    document.add_heading("5. Evidencias y fuentes", level=1)
    document.add_paragraph(
        "Evidently: evidently_drift_calidad.html y evidently_snapshot.json; métricas por "
        "columna: drift_por_variable.csv; Registry: registro_candidato.json; A/B: "
        "predicciones_ab.csv, metricas_ab.csv y resultado_ab.json. Consultar el README "
        "del Entregable 8 para la reproducción y añadir capturas de la UI de MLflow."
    )
    for source in (
        "Evidently AI (2026): https://docs.evidentlyai.com/quickstart_ml",
        "MLflow (2026): https://mlflow.org/docs/latest/ml/model-registry",
        "MiBici: https://mibici.net/es/datos-abiertos/",
        "Huyen, C. (2022). Designing Machine Learning Systems. O'Reilly Media.",
        "Treveil, M. et al. (2020). Introducing MLOps. O'Reilly Media.",
    ):
        document.add_paragraph(source)
    path = out / "informe_entregable8.docx"
    document.save(path)
    return path
