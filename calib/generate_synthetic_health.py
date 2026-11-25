"""Generate fictitious health-calibration data with buried true weights.

``python -m calib.generate_synthetic_health --config configs/default.yaml``

The generator simulates, for every company and month, five raw sub-item
measurements driven by a latent activity walk, standardises them into 0-100
scores, and derives a monthly churn probability from a logistic model whose
true weights are known and written to ``ground_truth.json``.  It emits two CSV
tables (``calib.health_snapshot`` and ``calib.churn_label``), the ground truth,
and a run summary that carries the acceptance-criteria statistics and file
hashes.

Only the Python standard library is used; nothing touches the network.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from . import miniyaml, schema
from .config import (
    Config,
    ConfigError,
    SUB_ITEMS,
    config_from_dict,
)

__all__ = [
    "generate",
    "write_outputs",
    "standardise",
    "solve_intercept",
    "main",
]

GENERATOR_NAME = "calib.generate_synthetic_health"
GENERATOR_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
UUID_NAMESPACE = uuid5(NAMESPACE_URL, "https://clawhunt.store/problems/2148/calib")
INTERCEPT_BOUNDS = (-40.0, 40.0)
INTERCEPT_TOLERANCE = 1e-12
INTERCEPT_MAX_ITER = 200


# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Period:
    index: int
    year: int
    month: int

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def first_day(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-01T00:00:00Z"

    @property
    def last_day(self) -> str:
        day = calendar.monthrange(self.year, self.month)[1]
        return f"{self.year:04d}-{self.month:02d}-{day:02d}"


@dataclass(frozen=True)
class Site:
    index: int
    code: str
    site_id: UUID


@dataclass
class Company:
    index: int
    code: str
    org_id: UUID
    site: Site
    industry: str
    stage: str
    tier: str
    latent: float
    biases: dict[str, float] = field(default_factory=dict)


@dataclass
class Row:
    company: Company
    period: Period
    raw: dict[str, float | None]
    score: dict[str, float | None]
    health: float | None
    data_flags: int
    effective_weights: dict[str, float]
    linear_predictor: float
    z: dict[str, float] = field(default_factory=dict)
    probability: float = 0.0


@dataclass
class GenerationResult:
    config: Config
    periods: list[Period]
    sites: list[Site]
    companies: list[Company]
    rows: list[Row]
    snapshot_row_text: list[dict[str, str]]
    churn_row_text: list[dict[str, str]]
    ground_truth: dict[str, Any]
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
def period_sequence(start_year: int, start_month: int, months: int) -> list[Period]:
    """Return ``months`` consecutive calendar months from ``start_year-start_month``."""
    periods: list[Period] = []
    year, month = start_year, start_month
    for index in range(months):
        periods.append(Period(index=index, year=year, month=month))
        month += 1
        if month == 13:
            month = 1
            year += 1
    return periods


def build_sites(config: Config) -> list[Site]:
    seed = config.seed
    return [
        Site(
            index=index,
            code=f"SYN-SITE-{index + 1:02d}",
            site_id=uuid5(UUID_NAMESPACE, f"{seed}/site/{index + 1}"),
        )
        for index in range(config.dataset.sites)
    ]


def _pick(rng: random.Random, pairs: Sequence[tuple[str, float]]) -> str:
    labels = [label for label, _ in pairs]
    weights = [weight for _, weight in pairs]
    return rng.choices(labels, weights=weights, k=1)[0]


def build_companies(config: Config, rng: random.Random, sites: Sequence[Site]) -> list[Company]:
    """Create companies in a fixed random order of draws.

    Draw order per company: industry category, stage, tier, initial latent
    activity, then one company-level bias per sub-item.  Keeping it fixed is
    what makes the same seed reproduce the same bytes.
    """
    companies: list[Company] = []
    for index in range(config.dataset.companies):
        code = f"SYN-{index + 1:04d}"
        industry = _pick(rng, config.attributes["industry"])
        stage = _pick(rng, config.attributes["stage"])
        tier = _pick(rng, config.attributes["tier"])
        latent = rng.gauss(0.0, config.latent.initial_sd)
        biases = {
            name: rng.gauss(0.0, config.sub_items[name].company_bias_sd)
            for name in SUB_ITEMS
        }
        companies.append(
            Company(
                index=index,
                code=code,
                org_id=uuid5(UUID_NAMESPACE, f"{config.seed}/org/{code}"),
                site=sites[index % len(sites)],
                industry=industry,
                stage=stage,
                tier=tier,
                latent=latent,
                biases=biases,
            )
        )
    return companies


def is_space_missing(config: Config, company: Company, period: Period) -> bool:
    missing = config.missing_data
    return (
        company.site.index == missing.site_index
        and missing.start_month_index <= period.index < missing.start_month_index + missing.months
    )


def simulate(
    config: Config, rng: random.Random, companies: Sequence[Company], periods: Sequence[Period]
) -> list[Row]:
    """Run the latent walk and produce raw sub-item values (no standardisation yet)."""
    seasonal_months = set(config.latent.seasonal_months)
    rows: list[Row] = []
    for company in companies:
        for period in periods:
            company.latent = (
                config.latent.phi * company.latent
                + rng.gauss(0.0, config.latent.innovation_sd)
            )
            activity = company.latent
            if period.month in seasonal_months:
                activity += config.latent.seasonal_effect

            raw: dict[str, float | None] = {}
            missing = is_space_missing(config, company, period)
            for name in SUB_ITEMS:
                if name == "space" and missing:
                    raw[name] = None
                    continue
                spec = config.sub_items[name]
                value = (
                    spec.base
                    + spec.loading * activity
                    + company.biases[name]
                    + rng.gauss(0.0, spec.noise_sd)
                )
                raw[name] = round(max(0.0, value), 3)
            rows.append(
                Row(
                    company=company,
                    period=period,
                    raw=raw,
                    score={},
                    health=None,
                    data_flags=schema.FLAG_SPACE_MISSING if missing else 0,
                    effective_weights={},
                    linear_predictor=0.0,
                )
            )
    # Output order is month-major so the snapshot reads like a table scan.
    rows.sort(key=lambda row: (row.period.index, row.company.code))
    return rows


def standardise(values: Sequence[float], clip_z: float) -> tuple[float, float, list[float]]:
    """Population z-score with a symmetric clip, one implementation for every item.

    Returns ``(mean, pstdev, clipped_z)``.  The same function produces the five
    ``*_score`` columns; a production scoring job can import it unchanged.
    """
    if len(values) < 2:
        raise ValueError("standardisation needs at least two observations")
    mean = statistics.fmean(values)
    deviation = statistics.pstdev(values, mu=mean)
    if deviation == 0.0:
        raise ValueError("standardisation needs a non-zero standard deviation")
    clipped = [max(-clip_z, min(clip_z, (value - mean) / deviation)) for value in values]
    return mean, deviation, clipped


def score_from_z(z: float, config: Config) -> float:
    raw = config.standardisation.score_mean + config.standardisation.score_sd * z
    return round(max(0.0, min(100.0, raw)), 2)


def effective_weights(config: Config, missing_space: bool) -> dict[str, float]:
    """Weights used in the logit; missing space renormalises the rest to the same total.

    Without this, a missing sub-item would silently lower every affected
    company's churn probability.  Renormalising keeps the average logit scale
    comparable across rows and is recorded in ``ground_truth.json``.
    """
    total = sum(config.true_weights.values())
    if not missing_space:
        return dict(config.true_weights)
    present = total - config.true_weights["space"]
    factor = total / present
    weights = {}
    for name in SUB_ITEMS:
        if name == "space":
            continue
        weights[name] = config.true_weights[name] * factor
    return weights


def apply_standardisation(config: Config, rows: Sequence[Row]) -> dict[str, dict[str, float]]:
    """Fill z, scores, health score, flags and the logit input on every row."""
    stats: dict[str, dict[str, float]] = {}
    for name in SUB_ITEMS:
        values = [row.raw[name] for row in rows if row.raw[name] is not None]
        mean, deviation, z_values = standardise(
            [float(value) for value in values], config.standardisation.clip_z
        )
        stats[name] = {"mean": mean, "pstdev": deviation, "observations": len(values)}
        cursor = 0
        for row in rows:
            if row.raw[name] is None:
                row.z.pop(name, None)
                continue
            z = z_values[cursor]
            cursor += 1
            row.z[name] = z
            row.score[name] = score_from_z(z, config)

    for row in rows:
        present = [name for name in SUB_ITEMS if row.raw[name] is not None]
        weights = effective_weights(config, "space" not in present)
        row.effective_weights = weights
        row.linear_predictor = math.fsum(weights[name] * row.z[name] for name in present)
        if len(present) == len(SUB_ITEMS):
            total_weight = sum(config.true_weights.values())
            row.health = round(
                math.fsum(
                    config.true_weights[name] * float(row.score[name]) for name in SUB_ITEMS
                )
                / total_weight,
                2,
            )
        else:
            row.health = None
    return stats


# ---------------------------------------------------------------------------
# churn model
# ---------------------------------------------------------------------------
def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def annual_probabilities(
    rows: Sequence[Row], intercept: float, window_start: int
) -> dict[str, float]:
    """1 - product(1 - monthly p) over the label window, per company code."""
    survival: dict[str, float] = {}
    for row in rows:
        if row.period.index < window_start:
            continue
        probability = sigmoid(intercept + row.linear_predictor)
        row.probability = probability
        survival[row.company.code] = survival.get(row.company.code, 1.0) * (1.0 - probability)
    return {code: 1.0 - value for code, value in survival.items()}


def mean_annual_churn(rows: Sequence[Row], intercept: float, window_start: int) -> float:
    probabilities = annual_probabilities(rows, intercept, window_start)
    if not probabilities:
        raise ValueError("label window contains no rows")
    return statistics.fmean(probabilities.values())


def solve_intercept(
    rows: Sequence[Row], window_start: int, target: float
) -> tuple[float, float]:
    """Bisection for b0 so the mean annual churn probability equals ``target``.

    The realised annual churn of the simulated probabilities (not the
    ``1-(1-monthly)^12`` approximation) is the quantity that is solved for, so
    the model expectation matches the configured target to floating precision.
    """
    low, high = INTERCEPT_BOUNDS
    low_value = mean_annual_churn(rows, low, window_start) - target
    high_value = mean_annual_churn(rows, high, window_start) - target
    if low_value > 0.0 or high_value < 0.0:
        raise ValueError(
            f"annual churn target {target} is not reachable inside {INTERCEPT_BOUNDS}"
        )
    for _ in range(INTERCEPT_MAX_ITER):
        middle = (low + high) / 2.0
        value = mean_annual_churn(rows, middle, window_start) - target
        if value > 0.0:
            high = middle
        else:
            low = middle
        if high - low <= INTERCEPT_TOLERANCE:
            break
    intercept = (low + high) / 2.0
    return intercept, mean_annual_churn(rows, intercept, window_start)


def draw_labels(
    config: Config,
    companies: Sequence[Company],
    rows: Sequence[Row],
    annual: Mapping[str, float],
    window_start: int,
    rng: random.Random,
) -> dict[str, str | None]:
    """Decide which companies churn in the window and in which month.

    ``exact_count`` (default) fixes the churner count at
    ``round(target * companies)`` and draws the churners proportionally to their
    annual probability, so the realised rate equals the configured target
    exactly; ``bernoulli`` draws each company independently at its own
    probability.  Both use the same latent model.
    """
    window_rows: dict[str, list[Row]] = {company.code: [] for company in companies}
    for row in rows:
        if row.period.index >= window_start:
            window_rows[row.company.code].append(row)

    if config.dataset.label_sampling == "exact_count":
        count = int(math.floor(config.dataset.annual_churn_target * len(companies) + 0.5))
        count = min(count, len(companies))
        ranked = sorted(
            companies,
            key=lambda company: (
                -(rng.random() ** (1.0 / max(annual[company.code], 1e-12))),
                company.code,
            ),
        )
        churned = {company.code for company in ranked[:count]}
    else:
        churned = {
            company.code
            for company in companies
            if rng.random() < annual[company.code]
        }

    churn_month: dict[str, str | None] = {}
    for company in companies:
        if company.code not in churned:
            churn_month[company.code] = None
            continue
        rows_for_company = window_rows[company.code]
        hazards = [row.probability for row in rows_for_company]
        total = math.fsum(hazards)
        if total <= 0.0:
            selected = rows_for_company[-1]
        else:
            selected = rng.choices(rows_for_company, weights=hazards, k=1)[0]
        churn_month[company.code] = selected.period.label
    return churn_month


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def generate(config: Config) -> GenerationResult:
    """Run the whole pipeline in memory and return every artefact."""
    rng = random.Random(config.seed)
    periods = period_sequence(
        config.dataset.start_year, config.dataset.start_month, config.dataset.months
    )
    sites = build_sites(config)
    companies = build_companies(config, rng, sites)
    rows = simulate(config, rng, companies, periods)
    stats = apply_standardisation(config, rows)

    window_start = config.dataset.months - config.window_months
    intercept, expected_annual = solve_intercept(
        rows, window_start, config.dataset.annual_churn_target
    )
    annual = annual_probabilities(rows, intercept, window_start)
    churn_month = draw_labels(config, companies, rows, annual, window_start, rng)

    snapshot_rows: list[dict[str, str]] = []
    for row in rows:
        snapshot_rows.append(
            {
                "id": str(uuid5(UUID_NAMESPACE, f"{config.seed}/snapshot/{row.company.code}/{row.period.label}")),
                "org_id": str(row.company.org_id),
                "site_id": str(row.company.site.site_id),
                "site_code": row.company.site.code,
                "company_code": row.company.code,
                "snapshot_month": row.period.label,
                "industry": row.company.industry,
                "stage": row.company.stage,
                "tier": row.company.tier,
                **{f"{name}_raw": schema.fmt(row.raw[name]) for name in SUB_ITEMS},
                **{f"{name}_score": schema.fmt(row.score.get(name), 2) for name in SUB_ITEMS},
                "health_score": schema.fmt(row.health, 2),
                "data_flags": str(row.data_flags),
                "created_at": row.period.first_day,
                "created_by": schema.NULL,
                "updated_at": row.period.first_day,
                "deleted_at": schema.NULL,
            }
        )

    window_periods = periods[window_start:]
    observed_until = window_periods[-1].last_day
    window_start_day = window_periods[0].first_day
    churn_rows: list[dict[str, str]] = []
    for company in companies:
        label = 1 if churn_month[company.code] else 0
        churn_rows.append(
            {
                "id": str(uuid5(UUID_NAMESPACE, f"{config.seed}/churn/{company.code}")),
                "org_id": str(company.org_id),
                "site_id": str(company.site.site_id),
                "site_code": company.site.code,
                "company_code": company.code,
                "label": str(label),
                "label_reason": "renewal_not_signed" if label else "renewed",
                "churn_month": churn_month[company.code] or schema.NULL,
                "observed_until": observed_until,
                "created_at": window_start_day,
                "created_by": schema.NULL,
                "updated_at": observed_until,
                "deleted_at": schema.NULL,
            }
        )

    churned_count = sum(1 for row in churn_rows if row["label"] == "1")
    missing_periods = [
        period
        for period in periods
        if config.missing_data.start_month_index
        <= period.index
        < config.missing_data.start_month_index + config.missing_data.months
    ]
    missing_rows = sum(1 for row in rows if row.data_flags & schema.FLAG_SPACE_MISSING)
    monthly = [row.probability for row in rows if row.period.index >= window_start]
    scores = [
        float(row.score[name])
        for row in rows
        for name in SUB_ITEMS
        if row.score.get(name) is not None
    ]
    health = [row.health for row in rows if row.health is not None]

    ground_truth: dict[str, Any] = {
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "schema_version": SCHEMA_VERSION,
        },
        "seed": config.seed,
        "config": config.as_dict(),
        "model": {
            "formula": "p(churn in month m) = logistic(b0 + sum_k w_k * z_k)",
            "true_weights": {name: round(config.true_weights[name], 8) for name in SUB_ITEMS},
            "intercept_b0": round(intercept, 10),
            "renormalised_weights_when_space_missing": {
                name: round(value, 8)
                for name, value in effective_weights(config, True).items()
            },
            "annual_churn_target": config.dataset.annual_churn_target,
            "model_expected_annual_churn": round(expected_annual, 10),
            "label_sampling": config.dataset.label_sampling,
            "realised_label_rate": round(churned_count / len(companies), 10),
            "window_months": config.window_months,
            "window_start": window_periods[0].label,
            "window_end": window_periods[-1].label,
        },
        "standardisation": {
            "method": (
                "population z-score per sub-item over all non-missing rows, "
                "clipped to +/-clip_z, then score = score_mean + score_sd * z "
                "clipped to [0, 100]"
            ),
            "clip_z": config.standardisation.clip_z,
            "score_mean": config.standardisation.score_mean,
            "score_sd": config.standardisation.score_sd,
            "stats": {
                name: {
                    "mean": round(stats[name]["mean"], 8),
                    "pstdev": round(stats[name]["pstdev"], 8),
                    "observations": int(stats[name]["observations"]),
                }
                for name in SUB_ITEMS
            },
        },
        "seasonality": {
            "months": list(config.latent.seasonal_months),
            "effect_on_activity": config.latent.seasonal_effect,
        },
        "missing_data": {
            "site_code": sites[config.missing_data.site_index].code,
            "site_index": config.missing_data.site_index,
            "periods": [period.label for period in missing_periods],
            "months": config.missing_data.months,
            "snapshot_rows": missing_rows,
            "handling": (
                "space_raw and space_score are NULL, health_score is NULL, "
                "data_flags bit 0 is set; the logit uses the remaining weights "
                "renormalised to the same total"
            ),
        },
        "companies": [
            {
                "company_code": company.code,
                "site_code": company.site.code,
                "industry": company.industry,
                "stage": company.stage,
                "tier": company.tier,
                "latent_activity_final": round(company.latent, 8),
                "annual_churn_probability": round(annual[company.code], 10),
                "label": 1 if churn_month[company.code] else 0,
                "churn_month": churn_month[company.code],
            }
            for company in companies
        ],
    }

    summary: dict[str, Any] = {
        "generator": GENERATOR_NAME,
        "version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "seed": config.seed,
        "dataset": {
            "companies": config.dataset.companies,
            "sites": config.dataset.sites,
            "months": config.dataset.months,
            "first_month": periods[0].label,
            "last_month": periods[-1].label,
        },
        "rows": {
            "health_snapshot": len(snapshot_rows),
            "churn_label": len(churn_rows),
        },
        "churn": {
            "annual_target": config.dataset.annual_churn_target,
            "model_expected_annual": round(expected_annual, 10),
            "realised_label_rate": round(churned_count / len(companies), 10),
            "churners": churned_count,
            "label_sampling": config.dataset.label_sampling,
            "intercept_b0": round(intercept, 10),
            "monthly_probability_mean": round(statistics.fmean(monthly), 10),
            "monthly_probability_min": round(min(monthly), 10),
            "monthly_probability_max": round(max(monthly), 10),
            "window_start": window_periods[0].label,
            "window_end": window_periods[-1].label,
        },
        "missing_data": {
            "site_code": sites[config.missing_data.site_index].code,
            "periods": [period.label for period in missing_periods],
            "consecutive_months": config.missing_data.months,
            "null_rows": missing_rows,
        },
        "scores": {
            "min": round(min(scores), 2),
            "max": round(max(scores), 2),
            "health_min": round(min(health), 2) if health else None,
            "health_max": round(max(health), 2) if health else None,
        },
        "files": {},
    }

    return GenerationResult(
        config=config,
        periods=periods,
        sites=sites,
        companies=companies,
        rows=rows,
        snapshot_row_text=snapshot_rows,
        churn_row_text=churn_rows,
        ground_truth=ground_truth,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def write_outputs(result: GenerationResult, out_dir: str) -> dict[str, Any]:
    """Write the two CSVs, ground truth and summary; return the summary with hashes."""
    os.makedirs(out_dir, exist_ok=True)
    snapshot_path = os.path.join(out_dir, "health_snapshot.csv")
    churn_path = os.path.join(out_dir, "churn_label.csv")
    truth_path = os.path.join(out_dir, "ground_truth.json")
    summary_path = os.path.join(out_dir, "summary.json")

    snapshot_rows = schema.write_csv(
        snapshot_path, schema.SNAPSHOT_COLUMNS, result.snapshot_row_text
    )
    churn_rows = schema.write_csv(churn_path, schema.CHURN_COLUMNS, result.churn_row_text)
    _write_json(truth_path, result.ground_truth)

    summary = dict(result.summary)
    summary["rows"] = {"health_snapshot": snapshot_rows, "churn_label": churn_rows}
    summary["files"] = {
        "health_snapshot.csv": {
            "rows": snapshot_rows,
            "sha256": schema.sha256_file(snapshot_path),
        },
        "churn_label.csv": {
            "rows": churn_rows,
            "sha256": schema.sha256_file(churn_path),
        },
        "ground_truth.json": {"sha256": schema.sha256_file(truth_path)},
    }
    _write_json(summary_path, summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m calib.generate_synthetic_health",
        description=(
            "Generate fictitious health-score calibration data with buried true weights."
        ),
    )
    parser.add_argument(
        "--config", default="configs/default.yaml", help="YAML configuration (default: %(default)s)"
    )
    parser.add_argument("--out", help="output directory (overrides out_dir in the config)")
    parser.add_argument("--seed", type=int, help="random seed override")
    parser.add_argument("--companies", type=int, help="company count override")
    parser.add_argument("--sites", type=int, help="club/site count override")
    parser.add_argument("--months", type=int, help="month count override")
    parser.add_argument("--annual-churn", type=float, help="annual churn target override")
    parser.add_argument(
        "--label-sampling",
        choices=("exact_count", "bernoulli"),
        help="churn label sampling override",
    )
    parser.add_argument("--quiet", action="store_true", help="write files but print nothing")
    return parser


def _apply_overrides(document: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dataset = dict(document.get("dataset") or {})
    if args.companies is not None:
        dataset["companies"] = args.companies
    if args.sites is not None:
        dataset["sites"] = args.sites
    if args.months is not None:
        dataset["months"] = args.months
    if args.annual_churn is not None:
        dataset["annual_churn_target"] = args.annual_churn
    if args.label_sampling is not None:
        dataset["label_sampling"] = args.label_sampling
    updated = dict(document)
    updated["dataset"] = dataset
    if args.seed is not None:
        updated["seed"] = args.seed
    if args.out:
        updated["out_dir"] = args.out
    return updated


def format_report(summary: Mapping[str, Any], out_dir: str, ground_truth_path: str) -> str:
    dataset = summary["dataset"]
    churn = summary["churn"]
    missing = summary["missing_data"]
    files = summary["files"]
    lines = [
        "calib synthetic health generator",
        f"  output            {out_dir}",
        f"  seed              {summary['seed']}",
        (
            f"  dataset           {dataset['companies']} companies x {dataset['sites']} sites "
            f"x {dataset['months']} months ({dataset['first_month']}..{dataset['last_month']})"
        ),
        "",
        "files",
        (
            f"  health_snapshot.csv  {files['health_snapshot.csv']['rows']:>6} rows  "
            f"sha256 {files['health_snapshot.csv']['sha256']}"
        ),
        (
            f"  churn_label.csv      {files['churn_label.csv']['rows']:>6} rows  "
            f"sha256 {files['churn_label.csv']['sha256']}"
        ),
        f"  ground_truth.json           sha256 {files['ground_truth.json']['sha256']}",
        "",
        "churn calibration",
        f"  annual target            {churn['annual_target'] * 100:8.2f}%   [AC2 window 10%-14%]",
        f"  model expected annual    {churn['model_expected_annual'] * 100:8.4f}%",
        (
            f"  realised label rate      {churn['realised_label_rate'] * 100:8.2f}%   "
            f"({churn['churners']}/{dataset['companies']} companies, {churn['label_sampling']})"
        ),
        (
            f"  monthly probability      {churn['monthly_probability_mean'] * 100:8.4f}% mean  "
            f"({churn['monthly_probability_min'] * 100:.4f}% .. "
            f"{churn['monthly_probability_max'] * 100:.4f}%)"
        ),
        f"  solved intercept b0      {churn['intercept_b0']:8.4f}",
        "",
        "missing data",
        (
            f"  {missing['site_code']} space utilisation missing for "
            f"{missing['consecutive_months']} consecutive months "
            f"({missing['periods'][0]}..{missing['periods'][-1]}): {missing['null_rows']} null rows"
        ),
        "",
        f"ground truth: {ground_truth_path}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        document = miniyaml.load(args.config)
    except OSError as exc:
        parser.error(f"cannot read config file {args.config!r}: {exc}")
    except miniyaml.YAMLError as exc:
        parser.error(f"{args.config}: {exc}")
    if not isinstance(document, dict):
        parser.error(f"{args.config}: the configuration root must be a mapping")
    try:
        config = config_from_dict(_apply_overrides(document, args))
    except ConfigError as exc:
        parser.error(f"{args.config}: {exc}")

    result = generate(config)
    out_dir = config.out_dir
    summary = write_outputs(result, out_dir)
    if not args.quiet:
        print(format_report(summary, out_dir, os.path.join(out_dir, "ground_truth.json")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI test
    sys.exit(main())
