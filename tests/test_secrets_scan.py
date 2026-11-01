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


def test_detector_no_marca_variables_vacias_de_env_example(tmp_path):
    # Una variable vacía no debe "robarse" la línea siguiente como si fuera su valor.
    (tmp_path / ".env.example").write_text(
        "DATABASE_URL=\n\n# Supabase Storage (compatible con S3)\nAWS_ACCESS_KEY_ID=\n",
        encoding="utf-8",
    )
    issues, total = scan_paths(tmp_path)
    assert total == 1
    assert not issues


def test_detector_si_marca_valor_real_en_la_misma_linea(tmp_path):
    (tmp_path / "settings.py").write_text('PASSWORD = "contrasena-muy-real-123"\n', encoding="utf-8")
    issues, _ = scan_paths(tmp_path)
    assert any("literal de credencial" in issue for issue in issues)
