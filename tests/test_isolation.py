"""The generator runs with no third-party packages and no network access."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_imports_without_site_packages(project_root: Path) -> None:
    """``-S`` disables site-packages: only the standard library can satisfy the import."""
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import calib.generate_synthetic_health as m; print(m.GENERATOR_NAME)",
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "calib.generate_synthetic_health"


def test_runtime_sources_import_only_the_standard_library(project_root: Path) -> None:
    forbidden = ("import requests", "import numpy", "import pandas", "import yaml", "import sklearn")
    for path in sorted((project_root / "calib").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path.name} imports a third-party package"


def test_generator_makes_no_socket_calls(project_root: Path, tmp_path: Path) -> None:
    """Run a small generation with socket creation patched to fail."""
    script = """
import socket
class Blocked(socket.socket):
    def __init__(self, *args, **kwargs):
        raise AssertionError("network access attempted")
socket.socket = Blocked
from calib import generate_synthetic_health as generator
from calib.config import load_config
import dataclasses, sys
config = load_config("configs/default.yaml")
config = dataclasses.replace(config, dataset=dataclasses.replace(config.dataset, companies=20))
result = generator.generate(config)
generator.write_outputs(result, sys.argv[1])
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "run")],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
