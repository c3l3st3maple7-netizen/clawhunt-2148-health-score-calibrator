"""Acceptance criteria and data-quality properties of the default run."""

from __future__ import annotations

import dataclasses
import re
import statistics
from collections import defaultdict
from pathlib import Path
from uuid import UUID

import pytest

from calib import generate_synthetic_health as generator
from calib import schema
from calib.config import MissingDataConfig

SUB_ITEMS = schema.SUB_ITEMS


# ---------------------------------------------------------------------------
# AC1 — same seed, identical bytes
# ---------------------------------------------------------------------------
def test_same_seed_is_byte_identical(tmp_path: Path, default_config) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generator.write_outputs(generator.generate(default_config), str(first))
    generator.write_outputs(generator.generate(default_config), str(second))

    names = ["health_snapshot.csv", "churn_label.csv", "ground_truth.json", "summary.json"]
    for name in names:
        digest_first = schema.sha256_file(str(first / name))
        digest_second = schema.sha256_file(str(second / name))
        assert digest_first == digest_second, name
    assert first.joinpath("health_snapshot.csv").read_bytes() == second.joinpath(
        "health_snapshot.csv"
    ).read_bytes()


def test_different_seed_changes_the_data(tmp_path: Path, default_config) -> None:
    other = dataclasses.replace(default_config, seed=default_config.seed + 1)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generator.write_outputs(generator.generate(default_config), str(first))
    generator.write_outputs(generator.generate(other), str(second))
    assert schema.sha256_file(str(first / "health_snapshot.csv")) != schema.sha256_file(
        str(second / "health_snapshot.csv")
    )


def test_row_counts_and_summary(default_run) -> None:
    assert len(default_run.rows) == 300 * 24
    assert len(default_run.churn) == 300
    assert default_run.summary["rows"] == {"health_snapshot": 7200, "churn_label": 300}
    assert default_run.summary["files"]["health_snapshot.csv"]["sha256"] == schema.sha256_file(
        str(default_run.directory / "health_snapshot.csv")
    )


# ---------------------------------------------------------------------------
# AC2 — annual churn rate inside the 10%-14% window
# ---------------------------------------------------------------------------
def test_annual_churn_rate_is_inside_the_acceptance_window(default_run) -> None:
    churners = sum(1 for row in default_run.churn if row["label"] == "1")
    rate = churners / len(default_run.churn)
    assert 0.10 <= rate <= 0.14, f"realised annual churn {rate:.2%} outside 10%-14%"
    assert rate == pytest.approx(0.12, abs=1e-9)
    assert default_run.summary["churn"]["realised_label_rate"] == pytest.approx(rate)


def test_model_expectation_matches_the_target(default_run) -> None:
    truth = default_run.result.ground_truth
    assert truth["model"]["model_expected_annual_churn"] == pytest.approx(0.12, abs=1e-9)
    probabilities = generator.annual_probabilities(
        default_run.result.rows, truth["model"]["intercept_b0"], 12
    )
    assert statistics.fmean(probabilities.values()) == pytest.approx(0.12, abs=1e-8)


def test_monthly_probabilities_are_valid(default_run) -> None:
    window = [row.probability for row in default_run.result.rows if row.period.index >= 12]
    assert window
    assert all(0.0 < value < 1.0 for value in window)
    assert min(window) < 0.01 < max(window)


# ---------------------------------------------------------------------------
# AC3 — one club's space data missing for three consecutive months
# ---------------------------------------------------------------------------
def flagged_rows(default_run) -> list[dict[str, str]]:
    return [row for row in default_run.rows if row["data_flags"] == "1"]


def test_only_the_configured_site_has_missing_space_data(default_run) -> None:
    flagged = flagged_rows(default_run)
    assert flagged
    assert {row["site_code"] for row in flagged} == {"SYN-SITE-03"}
    flagged_ids = {row["id"] for row in flagged}
    assert all(
        row["data_flags"] == "0"
        for row in default_run.rows
        if row["id"] not in flagged_ids
    )


def test_missing_months_are_three_consecutive_months(default_run) -> None:
    periods = default_run.result.ground_truth["missing_data"]["periods"]
    assert periods == ["2025-01", "2025-02", "2025-03"]

    by_company: dict[str, list[str]] = defaultdict(list)
    for row in flagged_rows(default_run):
        by_company[row["company_code"]].append(row["snapshot_month"])
    assert len(by_company) == 100  # one third of the 300 companies
    for code, months in by_company.items():
        assert months == periods, code


def test_missing_rows_have_null_space_and_health(default_run) -> None:
    flagged = flagged_rows(default_run)
    assert len(flagged) == 3 * 100
    for row in flagged:
        assert row["space_raw"] == ""
        assert row["space_score"] == ""
        assert row["health_score"] == ""
        assert row["event_raw"] != ""
    complete = [row for row in default_run.rows if row["data_flags"] == "0"]
    assert len(complete) == 7200 - 300
    for row in complete[:1000]:
        assert row["space_raw"] != ""
        assert row["health_score"] != ""


def test_missing_window_is_configurable(tmp_path: Path, default_config) -> None:
    config = dataclasses.replace(
        default_config,
        dataset=dataclasses.replace(
            default_config.dataset, companies=80, sites=2, months=15
        ),
        missing_data=MissingDataConfig(site_index=0, start_month_index=2, months=3),
    )
    result = generator.generate(config)
    assert len(result.rows) == 80 * 15
    flagged = [row for row in result.rows if row.data_flags & schema.FLAG_SPACE_MISSING]
    assert len(flagged) == 40 * 3
    assert {row.company.site.code for row in flagged} == {"SYN-SITE-01"}
    assert {row.period.label for row in flagged} == {"2024-03", "2024-04", "2024-05"}


# ---------------------------------------------------------------------------
# table content
# ---------------------------------------------------------------------------
def test_snapshot_columns(default_run) -> None:
    with (default_run.directory / "health_snapshot.csv").open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
    assert tuple(header.split(",")) == schema.SNAPSHOT_COLUMNS


def test_churn_columns(default_run) -> None:
    with (default_run.directory / "churn_label.csv").open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
    assert tuple(header.split(",")) == schema.CHURN_COLUMNS


def test_company_codes_and_uuids(default_run) -> None:
    pattern = re.compile(r"^SYN-\d{4}$")
    codes = {row["company_code"] for row in default_run.rows}
    assert len(codes) == 300
    assert all(pattern.match(code) for code in codes)
    assert sorted(codes)[0] == "SYN-0001"
    assert sorted(codes)[-1] == "SYN-0300"

    for row in default_run.rows[:200]:
        UUID(row["id"])
        UUID(row["org_id"])
        UUID(row["site_id"])


def test_organisation_ids_are_consistent_across_tables(default_run) -> None:
    from_snapshot = {
        row["company_code"]: (row["org_id"], row["site_id"], row["site_code"])
        for row in default_run.rows
    }
    for row in default_run.churn:
        assert from_snapshot[row["company_code"]] == (
            row["org_id"],
            row["site_id"],
            row["site_code"],
        )
    assert len({value[0] for value in from_snapshot.values()}) == 300
    assert len({value[1] for value in from_snapshot.values()}) == 3


def test_values_are_in_range(default_run) -> None:
    for row in default_run.rows:
        for item in SUB_ITEMS:
            raw = row[f"{item}_raw"]
            if raw != "":
                assert float(raw) >= 0.0
            score = row[f"{item}_score"]
            if score != "":
                assert 0.0 <= float(score) <= 100.0
        if row["health_score"] != "":
            assert 0.0 <= float(row["health_score"]) <= 100.0


def test_health_score_is_the_weighted_mean_of_the_five_scores(default_run) -> None:
    weights = default_run.result.ground_truth["model"]["true_weights"]
    total = sum(weights.values())
    for row in default_run.rows[:300]:
        if row["health_score"] == "":
            continue
        expected = (
            sum(weights[item] * float(row[f"{item}_score"]) for item in SUB_ITEMS) / total
        )
        assert float(row["health_score"]) == pytest.approx(expected, abs=0.01)


def test_contract_has_no_personal_or_contact_columns(default_run) -> None:
    forbidden = {"name", "company_name", "email", "phone", "mobile", "contact"}
    assert forbidden.isdisjoint({column.lower() for column in schema.SNAPSHOT_COLUMNS})
    assert forbidden.isdisjoint({column.lower() for column in schema.CHURN_COLUMNS})
    for row in default_run.rows[:500]:
        assert "@" not in row["company_code"]


def test_january_february_seasonal_dip(default_run) -> None:
    early = [
        float(row["event_raw"])
        for row in default_run.rows
        if row["snapshot_month"][5:] in ("01", "02")
    ]
    other = [
        float(row["event_raw"])
        for row in default_run.rows
        if row["snapshot_month"][5:] not in ("01", "02")
    ]
    assert statistics.fmean(early) < statistics.fmean(other) - 0.5


def test_churn_labels_are_consistent(default_run) -> None:
    window = default_run.result.ground_truth["model"]
    labels = {row["company_code"]: row for row in default_run.churn}
    for row in default_run.churn:
        assert row["label"] in {"0", "1"}
        assert row["label_reason"] in {"renewed", "renewal_not_signed"}
        assert (row["label"] == "1") == (row["churn_month"] != "")
        if row["churn_month"]:
            assert window["window_start"] <= row["churn_month"] <= window["window_end"]
            assert row["label_reason"] == "renewal_not_signed"
        else:
            assert row["label_reason"] == "renewed"
        assert row["observed_until"] == "2025-12-31"
        assert row["created_at"] == "2025-01-01T00:00:00Z"
    assert len(labels) == 300


def test_ground_truth_carries_the_true_weights(default_run) -> None:
    truth = default_run.result.ground_truth
    assert truth["model"]["true_weights"] == pytest.approx(
        default_config_weights(default_run)
    )
    assert truth["standardisation"]["stats"]["space"]["observations"] == 6900
    assert truth["standardisation"]["stats"]["event"]["observations"] == 7200
    assert truth["missing_data"]["snapshot_rows"] == 300
    assert len(truth["companies"]) == 300
    probabilities = [company["annual_churn_probability"] for company in truth["companies"]]
    assert statistics.fmean(probabilities) == pytest.approx(0.12, abs=1e-8)
    assert all(0.0 < value < 1.0 for value in probabilities)


def default_config_weights(default_run) -> dict[str, float]:
    return dict(default_run.result.config.true_weights)


def test_exact_count_sampling_hits_the_target_exactly(default_config) -> None:
    config = dataclasses.replace(
        default_config,
        dataset=dataclasses.replace(default_config.dataset, label_sampling="exact_count"),
    )
    result = generator.generate(config)
    churners = sum(1 for row in result.churn_row_text if row["label"] == "1")
    assert churners == round(0.12 * 300)
    truth = result.ground_truth["model"]
    assert truth["realised_label_rate"] == pytest.approx(0.12)
    assert truth["model_expected_annual_churn"] == pytest.approx(0.12, abs=1e-9)
