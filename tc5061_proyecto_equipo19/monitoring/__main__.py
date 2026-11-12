"""CLI Entregable 8: python -m tc5061_proyecto_equipo19.monitoring --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tc5061_proyecto_equipo19 import config as cfg
from tc5061_proyecto_equipo19.monitoring.data import (
    CURRENT_MONTHS, REFERENCE_MONTHS, EVALUATION_MONTHS,
    load_p1_months, parse_months, validate_periods,
)

DEFAULT_OUTPUT = cfg.REPORTS_DIR / "entregable8"


def prepare_raw(months: tuple[str, ...]):
    from tc5061_proyecto_equipo19.dataset import process_trips

    for month in months:
        csv = cfg.RAW_DATA_DIR / f"datos_abiertos_{month}.csv"
        if not csv.is_file():
            raise FileNotFoundError(
                f"No existe {csv}; descarga el CSV de MiBici antes de preparar la ingesta"
            )
        result = process_trips(csv)
        print(f"Procesado {month}: {result['records']:,} viajes")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-raw", "drift", "train", "ab", "report", "all", "promote"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--processed-dir", type=Path, default=cfg.PROCESSED_DATA_DIR)
    parser.add_argument("--reference", default=",".join(REFERENCE_MONTHS))
    parser.add_argument("--current", default=",".join(CURRENT_MONTHS))
    parser.add_argument("--months", default="", help="Para prepare-raw: meses CSV a convertir")
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--base-version", default=None, help="Versión A; por defecto alias champion")
    parser.add_argument("--candidate-version", default=None, help="Versión B; alias candidate_e8")
    parser.add_argument("--approve", action="store_true", help="Promoción explícita autorizada")
    args = parser.parse_args(argv)
    reference, current = parse_months(args.reference), parse_months(args.current)
    validate_periods(reference, current, EVALUATION_MONTHS)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare-raw":
        if not args.months:
            parser.error("prepare-raw requiere --months 2026_07,2026_08,2026_09")
        prepare_raw(parse_months(args.months))
        return
    if args.command == "promote":
        from tc5061_proyecto_equipo19.monitoring.ab import promote_with_approval
        print("Champion promovido a versión:", promote_with_approval(args.out, approve=args.approve))
        return
    if args.command in ("drift", "all"):
        from tc5061_proyecto_equipo19.monitoring.drift import generate_drift
        summary = generate_drift(
            load_p1_months(reference, args.processed_dir),
            load_p1_months(current, args.processed_dir), args.out,
            max_rows=args.max_rows,
        )
        summary["reference_months"] = list(reference)
        summary["current_months"] = list(current)
        (args.out / "resumen_drift.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("Evidently:", args.out / "evidently_drift_calidad.html")
        print("Variables con drift:", summary["actionable_drift_variables"])
    if args.command in ("train", "all"):
        from tc5061_proyecto_equipo19.monitoring.ab import train_candidate
        result = train_candidate(args.out, alpha=args.alpha, processed_dir=args.processed_dir)
        print("Nueva versión registrada:", result["candidate_version"], "run:", result["run_id"])
    if args.command in ("ab", "all"):
        from tc5061_proyecto_equipo19.monitoring.ab import compare_registered_holdout
        result = compare_registered_holdout(
            args.out, base_version=args.base_version,
            candidate_version=args.candidate_version, processed_dir=args.processed_dir,
        )
        print("Decisión A/B:", result["recommendation"], "(no cambia champion)")
    if args.command in ("report", "all"):
        from tc5061_proyecto_equipo19.monitoring.report import generate_report
        generated = generate_report(args.out)
        print("Informe:", generated)


if __name__ == "__main__":
    main()
