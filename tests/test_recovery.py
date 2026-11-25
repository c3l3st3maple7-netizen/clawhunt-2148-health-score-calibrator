"""The reference recovery method can find the buried weights (loosely, at n=300)."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from calib import generate_synthetic_health as generator


def load_recovery_module(project_root: Path):
    path = project_root / "examples" / "recover_weights.py"
    spec = importlib.util.spec_from_file_location("recover_weights", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_discrete_time_hazard_fit_recovers_the_true_weights(
    project_root: Path, default_config, tmp_path: Path
) -> None:
    recovery = load_recovery_module(project_root)
    config = dataclasses.replace(default_config, seed=20260119)
    result = generator.generate(config)
    generator.write_outputs(result, str(tmp_path / "run"))

    snapshot, churn, truth = recovery.load_run(str(tmp_path / "run"))
    table = recovery.z_table(snapshot, truth)
    features, labels = recovery.hazard_dataset(table, churn, truth["model"]["window_start"])
    weights, intercept, _ = recovery.fit_logistic(features, labels)
    report = recovery.compare(weights, truth["model"]["true_weights"])

    assert 30 <= sum(labels) <= 36
    assert report["mean_absolute_error"] < 0.45
    assert report["max_absolute_error"] < 0.90
    assert intercept == pytest.approx(float(truth["model"]["intercept_b0"]), abs=1.0)


def test_estimator_is_worse_when_the_months_are_averaged_away(
    project_root: Path, default_config, tmp_path: Path
) -> None:
    recovery = load_recovery_module(project_root)
    result = generator.generate(default_config)
    generator.write_outputs(result, str(tmp_path / "run"))
    snapshot, churn, truth = recovery.load_run(str(tmp_path / "run"))
    table = recovery.z_table(snapshot, truth)

    hazard_features, hazard_labels = recovery.hazard_dataset(
        table, churn, truth["model"]["window_start"]
    )
    hazard_weights, hazard_intercept, _ = recovery.fit_logistic(hazard_features, hazard_labels)
    annual_features, annual_labels = recovery.annual_dataset(
        table, churn, truth["model"]["window_start"]
    )
    annual_weights, annual_intercept, _ = recovery.fit_logistic(annual_features, annual_labels)

    true_intercept = float(truth["model"]["intercept_b0"])
    hazard_error = abs(hazard_intercept - true_intercept)
    annual_error = abs(annual_intercept - true_intercept)
    assert hazard_error < annual_error
    assert recovery.compare(hazard_weights, truth["model"]["true_weights"])[
        "mean_absolute_error"
    ] < recovery.compare(annual_weights, truth["model"]["true_weights"])["mean_absolute_error"]
