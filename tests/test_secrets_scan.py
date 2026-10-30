"""Comprueba que el detector no ignore accidentalmente claves comunes."""
from quality.secrets_scan import scan_paths


def test_detector_de_secretos_encuentra_clave_y_no_imprime_el_valor(tmp_path):
    secret = "AKIA" + "A" * 16
    (tmp_path / "config.py").write_text('AWS_ACCESS_KEY_ID="' + secret + '"', encoding="utf-8")
    issues, total = scan_paths(tmp_path)
    assert total == 1
    assert len(issues) >= 1
    assert secret not in " ".join(issues)


def test_detector_no_lee_entornos_ignorados(tmp_path):
    (tmp_path / ".env.example").write_text("DATABASE_URL=<host_placeholder>", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "secret.txt").write_text("AKIA" + "A" * 16)
    issues, total = scan_paths(tmp_path)
    assert not issues
    assert total == 1
