"""Pruebas de integración DeepChecks; requieren entorno Python 3.10 separado."""
import pytest
from quality.data_validation import validate_p1_table

pytestmark = pytest.mark.deepchecks


def test_deepchecks_analiza_integridad_y_drift(tmp_path):
    pytest.importorskip("deepchecks", reason="DeepChecks se instala en Python 3.10 aislado")
    from quality.deepchecks_validation import demo_p1_tables, run_validation
    train, test = demo_p1_tables()
    result = run_validation(train, test, tmp_path, mode="demo_sintetica")
    assert result["deepchecks"] == "ejecutado"
    assert "origin_station_id" in result["feature_drift_scores"]
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "drift_train_test.html").is_file()
    assert (tmp_path / "integridad_train.html").is_file()


def test_deepchecks_detecta_duplicados_antes_del_reporte():
    from quality.deepchecks_validation import demo_p1_tables
    train, _ = demo_p1_tables()
    dirty = train.copy()
    dirty.loc[len(dirty)] = train.iloc[0]
    with pytest.raises(ValueError, match="duplicadas"):
        validate_p1_table(dirty, "train")


def test_deepchecks_p2_clasificacion_integridad_y_drift(tmp_path):
    pytest.importorskip("deepchecks", reason="DeepChecks requiere el entorno Python 3.10")
    from quality.p23_validation import demo_p2_tables, run_p23_validation
    train, test = demo_p2_tables()
    result = run_p23_validation(train, test, tmp_path / "p2", problem="P2", mode="demo_sintetica")
    assert result["deepchecks"] == "ejecutado"
    assert result["target"] == "y_riesgo"
    assert (tmp_path / "p2" / "summary.json").is_file()
    assert (tmp_path / "p2" / "drift_train_test.html").is_file()
    assert (tmp_path / "p2" / "integridad_test.html").is_file()


def test_deepchecks_p3_clustering_sin_etiqueta(tmp_path):
    pytest.importorskip("deepchecks", reason="DeepChecks requiere el entorno Python 3.10")
    from quality.p23_validation import demo_p3_tables, run_p23_validation
    train, test = demo_p3_tables()
    result = run_p23_validation(train, test, tmp_path / "p3", problem="P3", mode="demo_sintetica")
    assert result["deepchecks"] == "ejecutado"
    assert result["target"] is None  # P3 es clustering, no tiene variable objetivo.
    assert (tmp_path / "p3" / "summary.json").is_file()
    assert (tmp_path / "p3" / "drift_train_test.html").is_file()
    assert (tmp_path / "p3" / "integridad_train.html").is_file()
