"""Detección ligera de secretos accidentales en archivos versionables.

No reemplaza gitleaks ni verifica commits pasados. Evita imprimir valores de
secretos incluso si detecta alguno.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {".git", ".venv", "venv", "data", "mlruns", "mlartifacts",
                "reports", "outputs", "notebooks", "__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".txt", ".json", ".md", ".toml", ".yml", ".yaml",
                 ".ini", ".cfg", ".sh", ".ps1", ".env", ".example"}
PATTERNS = [
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("GitHub personal token", re.compile(r"\bgh[pousr]_[a-zA-Z0-9]{30,}\b")),
]

ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID|PASSWORD|API_KEY|SECRET_KEY|DATABASE_URL)"
    r"[ \t]*=[ \t]*(.*?)[ \t]*$"
)


def scan_paths(root: Path) -> tuple[list[str], int]:
    root = Path(root)
    paths = []
    # En repos Git evaluamos index + archivos nuevos no ignorados.
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=root, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        if result.returncode == 0:
            paths = [root / p.decode("utf-8", "surrogateescape")
                     for p in result.stdout.split(b"\0") if p]
    if not paths:  # ZIP sin directorio .git
        paths = [p for p in root.rglob("*") if p.is_file()
                 and not set(p.relative_to(root).parts).intersection(IGNORED_DIRS)]
    problems = []
    visited = 0
    for path in paths:
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            problems.append(f"{rel}: archivo de credenciales versionable")
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in {"Makefile", ".gitignore"}:
            continue
        visited += 1
        # No leer archivos grandes, binarios ni secretos reales ignorados.
        if path.stat().st_size > 1_000_000:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        for category, regex in PATTERNS:
            if regex.search(content):
                problems.append(f"{rel}: posible {category}")
        for match in ASSIGNMENT.finditer(content):
            value = match.group(1).strip().strip("\"'")
            if len(value) >= 12 and not any(token in value.lower() for token in
                                            ("<", ">", "${", "***", "example", "your_", "change_me")):
                problems.append(f"{rel}: posible literal de credencial")
    return problems, visited


def main() -> int:
    problems, n = scan_paths(ROOT)
    print(f"Archivos de texto comprobados: {n}")
    if problems:
        for issue in problems:
            print(f"[RIESGO] {issue}")
        return 1
    print("OK: sin coincidencias de secretos en archivos versionables (análisis básico).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
