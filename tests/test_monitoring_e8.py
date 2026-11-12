"""Calidad temporal, consistencia P1, bootstrap pareado y parser Evidently."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo19.monitoring.data import (
    CURRENT_MONTHS, EVALUATION_MONTHS, HISTORICAL_TRAIN_MONTHS,
    REFERENCE_MONTHS, dataframe_to_xy, load_p1_months, parse_months, validate_periods,
)
from tc5061_proyecto_equipo19.monitoring.drift import _parse_snapshot, quality_table, sample_equal
from tc5061_proyecto_equipo19.monitoring.ab import bootstrap_paired_days


def fake_trips(month):
    timestamp = pd.Timestamp(month.replace('_', '-') + '-05 10:00:00')
    return pd.DataFrame({
        'trip_id': ['1', '2', '2', '3', '4', 'bad'],
        'start_time': [timestamp, timestamp, timestamp, timestamp,
                       timestamp + pd.Timedelta(days=1), timestamp],
        'end_time': [timestamp + pd.Timedelta(minutes=12),
                     timestamp + pd.Timedelta(minutes=10),
                     timestamp + pd.Timedelta(minutes=10),
                     timestamp - pd.Timedelta(minutes=1),
                     timestamp + pd.Timedelta(days=1, minutes=2),
                     timestamp + pd.Timedelta(minutes=10)],
        'origin_station_id': ['A', 'A', 'A', 'A', 'B', None],
    })


def test_periods_cannot_leak_future():
    assert not (set(HISTORICAL_TRAIN_MONTHS) & set(EVALUATION_MONTHS))
    assert max(REFERENCE_MONTHS) < min(CURRENT_MONTHS)
    validate_periods(REFERENCE_MONTHS, CURRENT_MONTHS, EVALUATION_MONTHS)
    with pytest.raises(ValueError):
        validate_periods(CURRENT_MONTHS, REFERENCE_MONTHS, EVALUATION_MONTHS)
    with pytest.raises(ValueError):
        validate_periods(REFERENCE_MONTHS, CURRENT_MONTHS, ('2026_10',))
    with pytest.raises(ValueError):
        parse_months('2026_13')


def test_month_loader_uses_original_aggregation_and_filters(monkeypatch, tmp_path):
    (tmp_path / 'trips_2026_09.parquet').touch()
    monkeypatch.setattr(pd, 'read_parquet', lambda p, columns: fake_trips('2026_09')[columns])
    df = load_p1_months(('2026_09',), tmp_path)
    assert len(df) == 2
    assert df['cantidad_viajes'].tolist() == [2, 1]
    assert df['origin_station_id'].tolist() == ['A', 'B']
    x, y = dataframe_to_xy(df)
    assert x.columns.tolist() == ['Año', 'Mes', 'Dia', 'Dia_semana', 'origin_station_id']
    assert y.sum() == 3


def test_missing_parquet_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match='trips_2026_09.parquet'):
        load_p1_months(('2026_09',), tmp_path)


def test_drift_parser_uses_evidently_scores_not_reconstructed():
    cols = ['Año', 'Mes', 'Dia', 'Dia_semana', 'origin_station_id', 'cantidad_viajes']
    metrics = [{'metric_name': 'ValueDrift', 'config': {'column': col, 'method': 'jensenshannon', 'threshold': .1}, 'value': .15 if col in ('Año', 'cantidad_viajes') else .01} for col in cols]
    metrics += [{'metric_name': 'DriftedColumnsCount', 'config': {'drift_share': .5}, 'value': {'count': 2, 'share': 2/6}}]
    overall, table = _parse_snapshot({'metrics': metrics})
    assert overall['drifted_columns'] == 2
    assert not overall['dataset_drift']
    assert table.loc[table['variable'] == 'cantidad_viajes', 'drift'].iloc[0]
    assert table.loc[table['variable'] == 'Año', 'es_temporal_contextual'].iloc[0]
    with pytest.raises(RuntimeError, match='cambió'):
        _parse_snapshot({'metrics': []})


def test_quality_summary_is_unweighted_full_frame():
    df = pd.DataFrame({
        'Año': [2024, 2024], 'Mes': [7, 7], 'Dia': [1, 2], 'Dia_semana': [0, 1],
        'origin_station_id': ['A', 'B'], 'cantidad_viajes': [1, 2],
    })
    df.loc[1, 'origin_station_id'] = None
    quality = quality_table(df, df)
    assert quality.loc[(quality['periodo']=='actual') & (quality['variable']=='origin_station_id'), 'porcentaje_nulos'].iloc[0] == 50
    assert len(sample_equal(df, 100)) == 2


def test_paired_bootstrap_detects_consistent_improvement():
    y = np.arange(50, dtype=float)
    a, b = y + 4, y + 1
    days = pd.Series(pd.date_range('2026-09-01', periods=10).repeat(5))
    result = bootstrap_paired_days(y, a, b, days, reps=200, seed=42)
    assert result['delta_mae'] == -3
    assert result['ci95_upper'] < 0
    assert result == bootstrap_paired_days(y, a, b, days, reps=200, seed=42)
    same = bootstrap_paired_days(y, a, a, days, reps=200)
    assert same['delta_mae'] == 0 and same['ci95_upper'] == 0


def test_generate_drift_runs_evidently_07_api_with_fake_snapshot(monkeypatch, tmp_path):
    """Verifica llamadas de Snapshot/HTML, sin instalar/alterar Evidently en CI."""
    import sys
    import types
    from tc5061_proyecto_equipo19.monitoring.drift import generate_drift

    cols = ['Año', 'Mes', 'Dia', 'Dia_semana', 'origin_station_id', 'cantidad_viajes']
    metrics = [{'metric_name': 'ValueDrift', 'config': {'column': col, 'method': 'jensenshannon', 'threshold': .1}, 'value': 0.12 if col in ('Año','origin_station_id') else .02} for col in cols]
    metrics += [{'metric_name': 'DriftedColumnsCount', 'value': {'count': 2, 'share': 2/6}}]
    captured = {}

    class FakeDataDefinition:
        def __init__(self, **kwargs):
            captured['schema'] = kwargs

    class FakeDataset:
        @classmethod
        def from_pandas(cls, data, data_definition):
            return data

    class FakePreset:
        def __init__(self, **kwargs):
            captured.setdefault('presets', []).append(kwargs)

    class FakeReport:
        def __init__(self, metrics, include_tests):
            captured['include_tests'] = include_tests

        def run(self, current, reference):
            captured['sizes'] = (len(current), len(reference))
            return self

        def save_html(self, path):
            from pathlib import Path
            Path(path).write_text('<html>Mock únicamente para CI</html>')

        def dict(self):
            return {'metrics': metrics}

    mod = types.ModuleType('evidently')
    mod.DataDefinition, mod.Dataset, mod.Report = FakeDataDefinition, FakeDataset, FakeReport
    presets = types.ModuleType('evidently.presets')
    presets.DataDriftPreset = presets.DataSummaryPreset = FakePreset
    monkeypatch.setitem(sys.modules, 'evidently', mod)
    monkeypatch.setitem(sys.modules, 'evidently.presets', presets)
    frame = pd.DataFrame({
        'Año': [2024,2024], 'Mes': [7,7], 'Dia': [1,2], 'Dia_semana': [0,1],
        'origin_station_id': ['A','B'], 'cantidad_viajes': [2,3],
    })
    result = generate_drift(frame, frame, tmp_path, max_rows=100)
    assert result['actionable_input_features'] == ['origin_station_id']
    assert result['target_drift'] is False
    assert captured['include_tests'] is True
    assert captured['sizes'] == (2,2)
    assert (tmp_path / 'evidently_drift_calidad.html').is_file()
    assert (tmp_path / 'drift_por_variable.csv').is_file()
