"""The command line entry point."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def run_cli(project_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "calib.generate_synthetic_health", *arguments],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_help(project_root: Path) -> None:
    completed = run_cli(project_root, "--help")
    assert completed.returncode == 0
    assert "generate_synthetic_health" in completed.stdout
    assert "--config" in completed.stdout


def test_run_with_overrides(project_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    completed = run_cli(
        project_root,
        "--config",
        "configs/default.yaml",
        "--out",
        str(out),
        "--companies",
        "40",
        "--months",
        "15",
        "--seed",
        "7",
    )
    assert completed.returncode == 0, completed.stderr
    assert "realised label rate" in completed.stdout
    assert (out / "health_snapshot.csv").exists()
    assert (out / "ground_truth.json").exists()
    with (out / "health_snapshot.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40 * 15


def test_quiet_writes_files_without_output(project_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "quiet"
    completed = run_cli(
        project_root,
        "--out",
        str(out),
        "--companies",
        "30",
        "--months",
        "15",
        "--quiet",
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert (out / "summary.json").exists()


def test_invalid_config_exits_with_usage_error(project_root: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text(
        "seed: 1\nout_dir: out\ndataset:\n  companies: 300\n  unknown: 1\n",
        encoding="utf-8",
    )
    completed = run_cli(project_root, "--config", str(broken))
    assert completed.returncode == 2
    assert "unknown key" in completed.stderr


def test_missing_config_exits_with_usage_error(project_root: Path, tmp_path: Path) -> None:
    completed = run_cli(project_root, "--config", str(tmp_path / "absent.yaml"))
    assert completed.returncode == 2
    assert "cannot read config file" in completed.stderr
