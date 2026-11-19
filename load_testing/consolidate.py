"""Consolida resultados reales de Locust/FinOps de tres escenarios.

Uso: python -m load_testing.consolidate --runs baja5 media20 alta50
No crea valores sintéticos ni sobreescribe los archivos originales.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

DEFAULT_FOLDER = Path(__file__).resolve().parents[1] / 'reports' / 'entregable7'


def consolidate(folder: Path, runs: list[str]) -> Path:
    if len(runs) < 2 or len(set(runs)) != len(runs):
        raise ValueError('Se necesitan al menos dos IDs de corrida distintos')
    summaries = []
    for run in runs:
        f = folder / f'{run}_resumen.json'
        if not f.is_file():
            raise FileNotFoundError(f'No existe {f}. Ejecuta primero el benchmark de esa corrida.')
        data = json.loads(f.read_text(encoding='utf-8'))
        if data.get('run') != run or not isinstance(data.get('success'), int):
            raise ValueError(f'Resumen inválido o inconsistente: {f}')
        if data['success'] <= 0 or data.get('usd_per_successful_inference') is None:
            raise ValueError(f'No se puede calcular USD/inferencia en {run}: cero éxitos')
        required = ('latency_ms', 'rps_success', 'error_rate_pct',
                    'resource_avg_cpu_percent_by_service', 'resource_peak_mib_by_service', 'graphs')
        if any(k not in data for k in required):
            raise ValueError(f'Faltan métricas obligatorias en {f}')
        for chart in data['graphs']:
            if not (folder / chart).is_file():
                raise FileNotFoundError(f'Falta gráfica de la corrida {run}: {chart}')
        summaries.append(data)

    lines = [
        '# Entregable 7 — Comparativo real de pruebas de carga y FinOps', '',
        'Resultados generados exclusivamente a partir de los archivos `*_resumen.json`',
        'y de las gráficas producidas por Locust y el monitor de recursos. Las cifras',
        '**son mediciones locales**; la tarifa cloud es una hipótesis de costo.', '',
        '| Métrica | ' + ' | '.join(runs) + ' |',
        '|---|' + '|'.join(['---:'] * len(runs)) + '|',
    ]
    metrics = [
        ('Usuarios', lambda x: str(x['users'])),
        ('Duración configurada (s)', lambda x: f"{x['duration_seconds_configured']:g}"),
        ('Predicciones exitosas', lambda x: f"{x['success']:,}"),
        ('RPS exitosos', lambda x: f"{x['rps_success']:.3f}"),
        ('Latencia media (ms)', lambda x: f"{x['latency_ms']['promedio']:.2f}"),
        ('Latencia p95 (ms)', lambda x: f"{x['latency_ms']['p95']:.2f}"),
        ('Latencia p99 (ms)', lambda x: f"{x['latency_ms']['p99']:.2f}"),
        ('Errores (%)', lambda x: f"{x['error_rate_pct']:.2f}"),
        ('CPU API promedio (%)', lambda x: f"{x['resource_avg_cpu_percent_by_service'].get('api', float('nan')):.2f}"),
        ('RAM API máxima (MiB)', lambda x: f"{x['resource_peak_mib_by_service'].get('api', float('nan')):.2f}"),
        ('USD / inferencia exitosa', lambda x: f"{x['usd_per_successful_inference']:.9f}"),
        ('USD / 1000 inferencias', lambda x: f"{x['usd_per_1000_successful_inferences']:.6f}"),
    ]
    for label, fmt in metrics:
        lines.append('| ' + label + ' | ' + ' | '.join(fmt(s) for s in summaries) + ' |')
    lines.extend(['', '## Evidencias gráficas', ''])
    for s in summaries:
        lines.append(f"### {s['run']} ({s['users']} usuarios)")
        lines.append('')
        for name in s['graphs']:
            lines.append(f'![{name}]({name})')
        lines.append('')
        lines.append(f"[Reporte HTML Locust]({s['run']}_locust.html) · "
                     f"[Detalle FinOps]({s['run']}_resumen.md)")
        lines.append('')
    lines.extend([
        '## Discusión y limitaciones (completar con interpretación del equipo)', '',
        '- **Desempeño:** describir tendencias en p95/p99, RPS y errores por concurrencia.',
        '- **Recursos:** explicar relación de CPU y RAM con variaciones de latencia.',
        '- **FinOps:** comparar USD/inferencia de los escenarios; distinguir costo marginal,',
        '  utilización y costo de inactividad. El supuesto representa una VM completa',
        '  para API + MLflow, pero excluye red, almacenamiento y costo del generador Locust.',
        '- **Validez externa:** resultados en Podman/WSL2 no acreditan igual throughput',
        '  en AWS; confirmar tarifa y tamaño antes de tomar decisiones de producción.',
        '', '## Evidencia GitHub (completar manualmente)', '',
        '- URL del repositorio: PENDIENTE',
        '- URL de ejecución GitHub Actions con todos los jobs en verde: PENDIENTE', '',
    ])
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'RESUMEN_COMPARATIVO.md'
    output.write_text('\n'.join(lines), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--folder', type=Path, default=DEFAULT_FOLDER)
    args = parser.parse_args()
    print(f'Generado: {consolidate(args.folder, args.runs)}')


if __name__ == '__main__':
    main()
