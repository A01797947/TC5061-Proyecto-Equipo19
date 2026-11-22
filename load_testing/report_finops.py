"""Genera un reporte FinOps con las mediciones REALES de Locust y stats.

No inventa latencias ni muestras: si faltan archivos, falla con instrucciones.
Dependencias: pandas, matplotlib (ya declaradas en requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_RATE = 0.0835  # AWS t3.large Linux on-demand, us-east-1, USD/h (referencia 2026-11-18)
SOURCE = "https://aws.amazon.com/ec2/instance-types/t3/"


def col(frame: pd.DataFrame, *candidates: str, default: float | str | None = None):
    names = {str(c).strip().casefold(): c for c in frame.columns}
    for name in candidates:
        actual = names.get(name.casefold())
        if actual is not None:
            return frame[actual]
    if default is not None:
        return pd.Series([default] * len(frame), index=frame.index)
    raise ValueError(f"Falta columna {candidates}; presentes: {list(frame.columns)}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def metrics_from_stats(path: Path):
    frame = pd.read_csv(path).fillna("")
    names = col(frame, "Name").astype(str)
    total = frame[names.str.lower().isin(["aggregated", "total"])].copy()
    if len(total) != 1:
        raise ValueError(f"{path}: se espera una fila Aggregated; encontradas={len(total)}")
    agg = total.iloc[[0]]
    requests = int(numeric(col(agg, "Request Count")).iloc[0])
    failures = int(numeric(col(agg, "Failure Count")).iloc[0])
    if requests <= 0 or failures > requests:
        raise ValueError("No hay peticiones válidas para estimar el costo por inferencia")
    success = requests - failures
    ms = {
        "promedio": float(numeric(col(agg, "Average Response Time")).iloc[0]),
        "p50": float(numeric(col(agg, "Median Response Time", "50%")).iloc[0]),
        "p95": float(numeric(col(agg, "95%", "95%ile")).iloc[0]),
        "p99": float(numeric(col(agg, "99%", "99%ile")).iloc[0]),
    }
    rows = []
    for _, r in frame[~names.str.lower().isin(["aggregated", "total"])].iterrows():
        if "predict" not in str(r.get("Name", "")):
            continue
        rows.append({
            "endpoint": str(r.get("Name", "")),
            "requests": int(float(r.get("Request Count", 0))),
            "failures": int(float(r.get("Failure Count", 0))),
            "p95_ms": float(r.get("95%", r.get("95%ile", 0)) or 0),
        })
    return {"requests": requests, "errors": failures, "success": success,
            "error_rate_pct": 100 * failures / requests, "latency_ms": ms, "endpoints": rows}


def history_from_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Falta historial Locust: {path}")
    data = pd.read_csv(path)
    # --csv-full-history normalmente incluye Name, tipo de petición y métricas por endpoint.
    if "Name" in data.columns:
        data = data[data["Name"].astype(str).str.casefold().isin(["aggregated", "total"])].copy()
    if data.empty:
        raise ValueError("El historial no contiene filas Aggregated")
    data["_t"] = numeric(col(data, "Timestamp"))
    data = data.sort_values("_t").reset_index(drop=True)
    data["_sec"] = data["_t"] - data["_t"].min()
    data["_rps"] = numeric(col(data, "Total RPS", "Requests/s"))
    data["_failps"] = numeric(col(data, "Total Failure/s", "Failures/s", default=0))
    data["_latency"] = numeric(col(data, "Total Average Response Time", "Average Response Time"))
    return data


def resource_from_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Faltan muestras de docker/podman stats: {path}")
    data = pd.read_csv(path)
    needed = {"elapsed_s", "service", "cpu_percent", "mem_mib"}
    if not needed.issubset(data.columns) or data.empty:
        raise ValueError("Las muestras de CPU/memoria están vacías o no tienen todas las columnas")
    for name in ("elapsed_s", "cpu_percent", "mem_mib"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    if not data["cpu_percent"].notna().any() or not data["mem_mib"].notna().any():
        raise ValueError("No se pudieron interpretar muestras de CPU y memoria")
    if data["mem_mib"].max() <= 0:
        raise ValueError("La memoria medida es 0 MiB: el muestreo es inválido; revisar podman stats")
    return data


def draw_charts(history: pd.DataFrame, resources: pd.DataFrame, folder: Path, run: str) -> list[str]:
    charts = []
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["_sec"], history["_latency"], marker=".")
    ax.set(xlabel="Tiempo de prueba (s)", ylabel="Latencia media (ms)", title="Latencia HTTP durante la prueba")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    filename = f"{run}_grafico_latencia.png"; fig.savefig(folder / filename, dpi=145); plt.close(fig); charts.append(filename)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["_sec"], history["_rps"], label="Solicitudes/s")
    ax.plot(history["_sec"], history["_failps"], label="Fallos/s")
    ax.set(xlabel="Tiempo de prueba (s)", ylabel="Solicitudes/s", title="Throughput y fallos")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    filename = f"{run}_grafico_rps_errores.png"; fig.savefig(folder / filename, dpi=145); plt.close(fig); charts.append(filename)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for service, g in resources.groupby("service"):
        axes[0].plot(g["elapsed_s"], g["cpu_percent"], label=service, marker=".")
        axes[1].plot(g["elapsed_s"], g["mem_mib"], label=service, marker=".")
    axes[0].set(ylabel="CPU (%)", title="Consumo de contenedores (100% ≈ 1 núcleo)")
    axes[1].set(xlabel="Tiempo de prueba (s)", ylabel="RAM (MiB)")
    for ax in axes: ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    filename = f"{run}_grafico_recursos.png"; fig.savefig(folder / filename, dpi=145); plt.close(fig); charts.append(filename)
    return charts


def analyze(folder: Path, run: str, seconds: float, hourly_usd: float, users: int, engine: str) -> dict:
    if seconds <= 0 or hourly_usd <= 0 or users < 1:
        raise ValueError("seconds, hourly_usd y users deben ser positivos")
    stats = metrics_from_stats(folder / f"{run}_locust_stats.csv")
    history = history_from_csv(folder / f"{run}_locust_stats_history.csv")
    resources = resource_from_csv(folder / f"{run}_recursos.csv")
    charts = draw_charts(history, resources, folder, run)
    rps_success = stats["success"] / seconds
    cost_test = hourly_usd * seconds / 3600
    cost_per_success = cost_test / stats["success"] if stats["success"] else None
    summary = {
        "run": run, "users": users, "engine": engine, "duration_seconds_configured": seconds,
        "cloud_reference": "AWS EC2 t3.large Linux on-demand, US East (N. Virginia), 2 vCPU, 8 GiB",
        "hourly_usd": hourly_usd, "rate_source": SOURCE,
        "requests": stats["requests"], "errors": stats["errors"], "success": stats["success"],
        "error_rate_pct": stats["error_rate_pct"], "latency_ms": stats["latency_ms"],
        "rps_success": rps_success, "rps_all": stats["requests"] / seconds,
        "simulated_test_usd": cost_test,
        "usd_per_successful_inference": cost_per_success,
        "usd_per_1000_successful_inferences": None if cost_per_success is None else cost_per_success * 1000,
        "hypothetical_24x7_month_usd_720h": hourly_usd * 720,
        "resource_peak_mib_by_service": {k: float(g["mem_mib"].max()) for k, g in resources.groupby("service")},
        "resource_avg_cpu_percent_by_service": {k: float(g["cpu_percent"].mean()) for k, g in resources.groupby("service")},
        "endpoints": stats["endpoints"], "graphs": charts,
        "limitations": [
            "Proyección teórica para una única instancia: no es una factura real de AWS.",
            "Asume tasa constante durante toda la ventana configurada; incluye ramp-up y puede subestimar capacidad estable.",
            "El precio de la VM se asigna por completo a API + MLflow. No incluye impuestos, discos, tráfico, red, backups ni Supabase.",
            "La tasa medida en la laptop/Podman no garantiza el rendimiento de EC2 t3.large.",
            "T3 usa créditos de CPU: un uso sostenido por encima del baseline puede generar cargos adicionales.",
            "Las mediciones del proceso Locust se ejecutan fuera de la VM hipotética: su costo no se incluye.",
        ],
    }
    (folder / f"{run}_resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Entregable 7 — Resultados reales: {run}", "",
        f"**Ambiente:** motor {engine}; {users} usuarios, {seconds:g} s configurados; tres modelos desde snapshots.",
        "", "## Desempeño de las inferencias", "",
        "| Indicador | Medición |", "|---|---:|",
        f"| Peticiones totales | {stats['requests']:,} |",
        f"| Inferencias satisfactorias (HTTP 200 + JSON esperado) | {stats['success']:,} |",
        f"| Fallos | {stats['errors']:,} ({stats['error_rate_pct']:.2f}%) |",
        f"| Throughput medio exitoso | {rps_success:.2f} pred/s |",
        *[f"| Latencia {k} | {v:.1f} ms |" for k, v in stats["latency_ms"].items()],
        "", "### Desglose por endpoint", "",
        "| Modelo | Peticiones | Errores | p95 (ms) |", "|---|---:|---:|---:|",
        *[f"| {r['endpoint']} | {r['requests']} | {r['failures']} | {r['p95_ms']:.1f} |" for r in stats["endpoints"]],
        "", "## Recursos de contenedores", "",
        "| Servicio | CPU promedio | Memoria máxima |", "|---|---:|---:|",
        *[f"| {k} | {summary['resource_avg_cpu_percent_by_service'][k]:.1f}% | {v:.1f} MiB |"
          for k, v in summary["resource_peak_mib_by_service"].items()],
        "", "## FinOps (simulación de infraestructura; throughput medido localmente)", "",
        f"**Referencia:** [EC2 t3.large, Linux, us-east-1]({SOURCE}), USD {hourly_usd:.4f}/h.",
        f"**Costo hipotético de {seconds:g} s:** USD {cost_test:.6f}.",
        f"**Costo unitario estimado:** USD {cost_per_success:.8f} por inferencia exitosa." if cost_per_success is not None else "Costo por inferencia: no calculable.",
        f"**Costo por 1,000 inferencias exitosas:** USD {cost_per_success * 1000:.6f}." if cost_per_success is not None else "", 
        f"**Escenario VM encendida 720 h/mes:** USD {hourly_usd * 720:.2f}/mes (sin extras).",
        "", "### Limitaciones del cálculo", "",
        *[f"- {item}" for item in summary["limitations"]],
        "", "### Ideas de optimización para validar", "",
        "- Repetir las pruebas con 5, 20 y 50 usuarios y comparar p95, errores, RPS y costo unitario.",
        "- Evaluar separar MLflow del host de inferencia y dimensionar CPU/RAM según picos reales.",
        "- Con picos intermitentes, comparar autoscaling / horarios de apagado contra una VM 24/7.",
        "- Si CPU está saturada, comparar otra familia de instancias no burstable o más réplicas.",
        "- Cachear únicamente consultas deterministas y repetidas si la latencia y coherencia lo permiten.",
        "", "## Gráficas de las mediciones", "",
        *[f"![{name}]({name})" for name in charts], "",
        "**Metodología:** Locust envía solicitudes HTTP con ejemplos P1/P2/P3 (50%/30%/20%).",
        "Los archivos `*_locust*.csv` contienen las métricas capturadas por Locust y",
        "`*_recursos.csv` contiene las muestras periódicas de `docker stats`/`podman stats`.",
        "Las estimaciones no deben presentarse como gasto real ni como benchmark ejecutado en AWS.",
    ]
    (folder / f"{run}_resumen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--users", type=int, required=True)
    p.add_argument("--engine", default="docker")
    p.add_argument("--hourly-usd", type=float, default=DEFAULT_RATE)
    p.add_argument("--folder", type=Path, default=Path("reports/entregable7"))
    args = p.parse_args()
    report = analyze(args.folder, args.run_id, args.seconds, args.hourly_usd, args.users, args.engine)
    print(f"Reporte: {args.folder / (args.run_id + '_resumen.md')}")
    print(f"Inferencias exitosas: {report['success']}; USD/inferencia: {report['usd_per_successful_inference']}")


if __name__ == "__main__":
    main()
