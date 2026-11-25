"""Recover the buried true weights from one generated run (reference method).

The generator writes ``ground_truth.json`` so a calibration method can be
checked: run this script on a run directory and compare the recovered weights
with the buried truth.

Two estimators are shown, on the same data:

1. ``annual``   — a logistic regression of the annual churn label on each
   company's mean standardised sub-item.  This is the naive approach; averaging
   the monthly z-scores attenuates the signal and the coefficients come out
   biased.
2. ``hazard``   — a discrete-time hazard regression on company-months: each
   company contributes one row per month it is still at risk, with ``y = 1`` in
   its churn month.  This matches how the data were generated
   (``p = logistic(b0 + sum w_k z_k)`` per month), so it recovers the weights.

Usage::

    python examples/recover_weights.py runs/default
    python examples/recover_weights.py runs/default --json reports/recovery.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from typing import Any, Sequence

ITEMS = ("event", "interaction", "resource", "opportunity", "space")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_run(run_dir: str) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, Any]]:
    with open(os.path.join(run_dir, "health_snapshot.csv"), encoding="utf-8", newline="") as handle:
        snapshot = list(csv.DictReader(handle))
    churn: dict[str, dict[str, str]] = {}
    with open(os.path.join(run_dir, "churn_label.csv"), encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            churn[row["company_code"]] = row
    with open(os.path.join(run_dir, "ground_truth.json"), encoding="utf-8") as handle:
        truth = json.load(handle)
    return snapshot, churn, truth


def z_table(
    snapshot: Sequence[dict[str, str]], truth: dict[str, Any]
) -> dict[tuple[str, str], dict[str, float]]:
    stats = truth["standardisation"]["stats"]
    clip = float(truth["standardisation"]["clip_z"])
    table: dict[tuple[str, str], dict[str, float]] = {}
    for row in snapshot:
        values: dict[str, float] = {}
        for item in ITEMS:
            raw = row[f"{item}_raw"]
            if raw == "":
                continue
            mean = float(stats[item]["mean"])
            deviation = float(stats[item]["pstdev"])
            values[item] = max(-clip, min(clip, (float(raw) - mean) / deviation))
        table[(row["company_code"], row["snapshot_month"])] = values
    return table


def hazard_dataset(
    z_table: dict[tuple[str, str], dict[str, float]],
    churn: dict[str, dict[str, str]],
    window_start: str,
) -> tuple[list[list[float]], list[int]]:
    """Company-month risk set: one row per at-risk month, y=1 in the churn month."""
    months = sorted({month for _, month in z_table if month >= window_start})
    features: list[list[float]] = []
    labels: list[int] = []
    for code in sorted(churn):
        record = churn[code]
        churned = record["label"] == "1"
        for month in months:
            if churned and month > record["churn_month"]:
                break
            values = z_table.get((code, month))
            if values is None or len(values) != len(ITEMS):
                continue  # missing sub-item: drop the month rather than impute
            features.append([values[item] for item in ITEMS])
            labels.append(1 if churned and month == record["churn_month"] else 0)
    return features, labels


def annual_dataset(
    z_table: dict[tuple[str, str], dict[str, float]],
    churn: dict[str, dict[str, str]],
    window_start: str,
) -> tuple[list[list[float]], list[int]]:
    """Company-level means of z over the label window, with the annual label."""
    months = sorted({month for _, month in z_table if month >= window_start})
    features: list[list[float]] = []
    labels: list[int] = []
    for code in sorted(churn):
        rows = [z_table[(code, month)] for month in months if (code, month) in z_table]
        rows = [row for row in rows if len(row) == len(ITEMS)]
        if not rows:
            continue
        features.append([statistics.fmean(row[item] for row in rows) for item in ITEMS])
        labels.append(int(churn[code]["label"]))
    return features, labels


# ---------------------------------------------------------------------------
# logistic regression (IRLS with ridge)
# ---------------------------------------------------------------------------
def fit_logistic(
    features: Sequence[Sequence[float]], labels: Sequence[int], penalty: float = 1.0
) -> tuple[list[float], list[float], list[float]]:
    """Fit ``logit(p) = b0 + sum b_k x_k``; returns (weights, intercept, feature sd)."""
    columns = list(zip(*features))
    means = [statistics.fmean(column) for column in columns]
    deviations = [statistics.pstdev(column) or 1.0 for column in columns]
    design = [
        [1.0] + [(value - mean) / deviation for value, mean, deviation in zip(row, means, deviations)]
        for row in features
    ]
    width = len(design[0])
    beta = [0.0] * width

    def negative_log_likelihood(candidate: Sequence[float]) -> float:
        total = 0.0
        for row, label in zip(design, labels):
            eta = sum(b * x for b, x in zip(candidate, row))
            total += math.log1p(math.exp(-abs(eta))) + max(eta, 0.0) - label * eta
        return total + 0.5 * penalty * sum(value * value for value in candidate[1:])

    for _ in range(100):
        hessian = [[0.0] * width for _ in range(width)]
        gradient = [0.0] * width
        for row, label in zip(design, labels):
            eta = sum(b * x for b, x in zip(beta, row))
            probability = 1.0 / (1.0 + math.exp(-eta))
            weight = max(probability * (1.0 - probability), 1e-10)
            for j in range(width):
                gradient[j] += row[j] * (label - probability)
                for k in range(width):
                    hessian[j][k] += weight * row[j] * row[k]
        for j in range(1, width):
            hessian[j][j] += penalty
        step = _solve(hessian, gradient)
        largest = max(abs(value) for value in step)
        scale = min(1.0, 0.5 / largest) if largest > 0.5 else 1.0
        while scale > 1e-6:
            candidate = [b + scale * d for b, d in zip(beta, step)]
            if negative_log_likelihood(candidate) <= negative_log_likelihood(beta) + 1e-12:
                break
            scale /= 2.0
        beta = candidate
        if max(abs(value) for value in gradient) < 1e-8:
            break

    weights = [beta[index + 1] / deviations[index] for index in range(len(deviations))]
    intercept = beta[0] - sum(w * mean for w, mean in zip(weights, means))
    return weights, intercept, deviations


def _solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    size = len(matrix)
    augmented = [list(row) + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column] / augmented[column][column]
            for offset in range(column, size + 1):
                augmented[row][offset] -= factor * augmented[column][offset]
    return [augmented[row][size] / augmented[row][row] for row in range(size)]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def compare(recovered: Sequence[float], truth: dict[str, float]) -> dict[str, Any]:
    true_values = [truth[item] for item in ITEMS]
    errors = [abs(a - b) for a, b in zip(recovered, true_values)]
    return {
        "recovered": {item: round(value, 4) for item, value in zip(ITEMS, recovered)},
        "true": {item: round(value, 4) for item, value in zip(ITEMS, true_values)},
        "absolute_error": {item: round(value, 4) for item, value in zip(ITEMS, errors)},
        "mean_absolute_error": round(statistics.fmean(errors), 4),
        "max_absolute_error": round(max(errors), 4),
        "pearson": round(_pearson(recovered, true_values), 4),
        "rank_spearman": round(
            _pearson(_ranks(recovered), _ranks(true_values)), 4
        ),
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right)
    )
    return numerator / denominator if denominator else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order, 1):
        ranks[index] = float(rank)
    return ranks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="a run directory written by the generator")
    parser.add_argument("--json", help="also write the report as JSON to this path")
    args = parser.parse_args(argv)

    snapshot, churn, truth = load_run(args.run_dir)
    table = z_table(snapshot, truth)
    truth_weights = truth["model"]["true_weights"]
    true_intercept = float(truth["model"]["intercept_b0"])
    window_start = truth["model"]["window_start"]

    hazard_features, hazard_labels = hazard_dataset(table, churn, window_start)
    hazard_weights, hazard_intercept, _ = fit_logistic(hazard_features, hazard_labels)
    annual_features, annual_labels = annual_dataset(table, churn, window_start)
    annual_weights, annual_intercept, _ = fit_logistic(annual_features, annual_labels)

    report = {
        "run_dir": args.run_dir,
        "seed": truth["seed"],
        "true_intercept": round(true_intercept, 4),
        "hazard": {
            **compare(hazard_weights, truth_weights),
            "intercept": round(hazard_intercept, 4),
            "rows": len(hazard_labels),
            "events": sum(hazard_labels),
        },
        "annual": {
            **compare(annual_weights, truth_weights),
            "intercept": round(annual_intercept, 4),
            "companies": len(annual_labels),
        },
    }

    print(f"run            {report['run_dir']}")
    print(f"true weights   {report['hazard']['true']}")
    print(f"true intercept {report['true_intercept']}  (b0 of the monthly logit)")
    print("")
    for name in ("hazard", "annual"):
        section = report[name]
        print(f"{name:<7} weights {section['recovered']}")
        print(
            f"        MAE {section['mean_absolute_error']:.3f}  "
            f"max {section['max_absolute_error']:.3f}  "
            f"pearson {section['pearson']:.2f}  "
            f"rank {section['rank_spearman']:.2f}  "
            f"intercept {section['intercept']:.2f}"
        )
    print("")
    print("The discrete-time hazard fit is the estimator that matches the generator;")
    print("the annual fit is shown to document how much averaging the months biases it.")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="") as handle:
            json.dump(report, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
