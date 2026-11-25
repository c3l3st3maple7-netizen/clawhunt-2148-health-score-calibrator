"""Shared fixtures: one full default run, generated once per test session."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib import generate_synthetic_health as generator  # noqa: E402
from calib.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def default_config():
    return load_config(str(ROOT / "configs" / "default.yaml"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="session")
def default_run(tmp_path_factory, default_config) -> SimpleNamespace:
    directory = tmp_path_factory.mktemp("default-run")
    result = generator.generate(default_config)
    summary = generator.write_outputs(result, str(directory))
    return SimpleNamespace(
        directory=directory,
        result=result,
        summary=summary,
        rows=_read_csv(directory / "health_snapshot.csv"),
        churn=_read_csv(directory / "churn_label.csv"),
    )
