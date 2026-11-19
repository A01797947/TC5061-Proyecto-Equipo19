"""Ejecuta Locust vía Compose y registra CPU/RAM durante la misma ventana.

Ejemplo: python -m load_testing.benchmark --engine docker --users 20 --duration 120 --run-id carga20
Requiere haber ejecutado: docker compose up -d --build
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import threading
import time

from load_testing.report_finops import DEFAULT_RATE, analyze

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "entregable7"
UNIT = {"b": 1 / (1024 * 1024), "kib": 1 / 1024, "mib": 1, "gib": 1024,
        "tib": 1024 * 1024, "kb": 1000 / 1024**2, "mb": 1000**2 / 1024**2,
        "gb": 1000**3 / 1024**2, "tb": 1000**4 / 1024**2}


def ram_mib(value) -> float:
    if isinstance(value, (float, int)):
        return float(value) / 1024**2
    first = str(value).split("/", 1)[0].strip().replace(",", "")
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)", first)
    if not m:
        raise ValueError(f"Formato de memoria inesperado: {value!r}")
    return float(m.group(1)) * UNIT[m.group(2).lower()]


def cpu_percent(value) -> float:
    return float(str(value).strip().rstrip("%").replace(",", "."))


def compose_command(engine: str) -> list[str]:
    return {"docker": ["docker", "compose"], "podman": ["podman", "compose"],
            "podman-compose": ["podman-compose"]}[engine]


def run(args, *, check=True, timeout=None, env=None):
    return subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=check, timeout=timeout, env=env)


def sample(engine: str, containers: dict[str, str]) -> list[dict]:
    ids = list(containers.values())
    if engine == "docker":
        proc = run([engine, "stats", "--no-stream", "--format", "{{json .}}", *ids], timeout=20)
    else:
        proc = run(["podman", "stats", "--no-stream", "--format", "json", *ids], timeout=20)
    text = proc.stdout.strip()
    if not text:
        raise ValueError("El motor no devolvió estadísticas")
    rows = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines()]
    result = []
    for entry in rows:
        identity = str(entry.get("ID", entry.get("id", entry.get("Container", ""))))
        service = next((name for name, value in containers.items() if value.startswith(identity) or identity.startswith(value)), None)
        # Podman puede devolver el nombre, no el ID completo.
        if service is None and len(rows) == len(containers):
            # No asociamos datos por posición: preserva la trazabilidad.
            continue
        if service is None:
            continue
        cpu = entry.get("CPUPerc", entry.get("cpu_percent", entry.get("CPU", entry.get("cpu"))))
        mem = entry.get("MemUsage", entry.get("mem_usage", entry.get("Mem", entry.get("mem"))))
        result.append({"service": service, "cpu_percent": cpu_percent(cpu), "mem_mib": ram_mib(mem)})
    if not result:
        raise ValueError(f"Stats no contienen IDs reconocibles: {list(rows[0])}")
    return result


def monitor(engine: str, containers: dict[str, str], target: Path, errors: Path,
            interval: float, stop: threading.Event, started: float):
    with target.open("w", newline="", encoding="utf-8") as stream, errors.open("w", encoding="utf-8") as logfile:
        writer = csv.DictWriter(stream, fieldnames=["timestamp_utc", "elapsed_s", "service", "cpu_percent", "mem_mib"])
        writer.writeheader()
        while not stop.is_set():
            try:
                rows = sample("docker" if engine == "docker" else "podman", containers)
                for item in rows:
                    writer.writerow({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                    "elapsed_s": round(time.monotonic() - started, 2), **item})
                stream.flush()
            except (subprocess.SubprocessError, ValueError, KeyError, OSError) as ex:
                logfile.write(f"{datetime.now(timezone.utc).isoformat()} {type(ex).__name__}: {ex}\n")
                logfile.flush()
            stop.wait(interval)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=["docker", "podman", "podman-compose"], default="docker")
    p.add_argument("--run-id", required=True, help="Identificador único, p.ej. piloto_5")
    p.add_argument("--users", type=int, default=10)
    p.add_argument("--spawn-rate", type=float, default=2)
    p.add_argument("--duration", type=int, default=120, help="Segundos de carga")
    p.add_argument("--sample-interval", type=float, default=5)
    p.add_argument("--hourly-usd", type=float, default=DEFAULT_RATE)
    args = p.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", args.run_id):
        p.error("--run-id solo puede incluir letras, números, - y _ (máx. 40)")
    if min(args.users, args.duration, args.spawn_rate, args.sample_interval, args.hourly_usd) <= 0:
        p.error("Usuarios, duración, sample interval, spawn rate y tarifa deben ser mayores que cero")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    compose = compose_command(args.engine)
    containers = {}
    for service in ("api", "mlflow"):
        res = run(compose + ["ps", "-q", service])
        cid = res.stdout.strip().splitlines()
        if not cid:
            p.error(f"Servicio {service} no está levantado. Ejecuta: {' '.join(compose)} up -d --build")
        containers[service] = cid[0]
    stop = threading.Event()
    start = time.monotonic()
    thread = threading.Thread(target=monitor, args=("docker" if args.engine == "docker" else "podman", containers,
            OUTPUT / f"{args.run_id}_recursos.csv", OUTPUT / f"{args.run_id}_recursos_errores.log",
            args.sample_interval, stop, start), daemon=True)
    thread.start()
    import os
    env = os.environ.copy()
    env.update(LOCUST_USERS=str(args.users), LOCUST_SPAWN_RATE=str(args.spawn_rate),
               LOCUST_RUN_TIME=f"{args.duration}s", LOCUST_RUN_ID=args.run_id)
    command = compose + ["--profile", "load", "run", "--rm", "locust"]
    print(f"Iniciando: {' '.join(command)} | {args.users} usuarios durante {args.duration}s", flush=True)
    code = 1
    try:
        proc = subprocess.run(command, cwd=ROOT, env=env)
        code = proc.returncode
    finally:
        stop.set()
        thread.join(timeout=25)
    if code:
        print(f"Locust terminó con código {code}; revisar fallos HTTP y logs.", flush=True)
    try:
        result = analyze(OUTPUT, args.run_id, args.duration, args.hourly_usd, args.users, args.engine)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"No se pudo generar informe: {exc}; revisar CSV y log de recursos") from exc
    print(f"Informe: {OUTPUT / (args.run_id + '_resumen.md')}")
    print(f"RPS exitosos={result['rps_success']:.2f}, p95={result['latency_ms']['p95']:.1f}ms, "
          f"errores={result['error_rate_pct']:.2f}%, USD/inferencia={result['usd_per_successful_inference']}")
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
