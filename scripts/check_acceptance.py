"""Check the three customer acceptance criteria against two generated runs.

Usage::

    python scripts/check_acceptance.py --run-a deliver/run1 --run-b deliver/run2

Exit status is 0 only when all three criteria pass.  The printed report is the
"statistical output" evidence for AC2 and the assertion evidence for AC1/AC3.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calib import schema  # noqa: E402

FILES = ("health_snapshot.csv", "churn_label.csv", "ground_truth.json", "summary.json")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def longest_run(months: list[str]) -> int:
    """Longest run of consecutive YYYY-MM labels."""
    ordered = sorted(months)
    best = current = 1
    for previous, following in zip(ordered, ordered[1:]):
        year, month = (int(part) for part in previous.split("-"))
        expected = f"{year + 1:04d}-01" if month == 12 else f"{year:04d}-{month + 1:02d}"
        if following == expected:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def check_ac1(run_a: Path, run_b: Path) -> bool:
    print("AC1  same seed -> identical SHA-256 for every output file")
    passed = True
    for name in FILES:
        first = schema.sha256_file(str(run_a / name))
        second = schema.sha256_file(str(run_b / name))
        same = first == second
        passed = passed and same
        print(f"     {name:<22} {'identical' if same else 'DIFFERENT'}  {first}")
    print(f"     => {'PASS' if passed else 'FAIL'}\n")
    return passed


def check_ac2(run_a: Path) -> bool:
    churn = read_csv(run_a / "churn_label.csv")
    truth = json.loads((run_a / "ground_truth.json").read_text(encoding="utf-8"))
    summary = json.loads((run_a / "summary.json").read_text(encoding="utf-8"))
    churners = sum(1 for row in churn if row["label"] == "1")
    rate = churners / len(churn)
    window = truth["model"]
    monthly = summary["churn"]
    passed = 0.10 <= rate <= 0.14
    print("AC2  annual churn rate under the default configuration")
    print(
        f"     churners {churners} / companies {len(churn)} = {rate * 100:.2f}%"
        f"   window {window['window_start']}..{window['window_end']} ({window['window_months']} months)"
    )
    print(f"     model expected annual rate {window['model_expected_annual_churn'] * 100:.4f}%")
    print(
        f"     monthly probability mean {monthly['monthly_probability_mean'] * 100:.4f}%"
        f" (min {monthly['monthly_probability_min'] * 100:.4f}%,"
        f" max {monthly['monthly_probability_max'] * 100:.4f}%)"
    )
    print(f"     acceptance window 10%-14% -> {'PASS' if passed else 'FAIL'}\n")
    return passed


def check_ac3(run_a: Path) -> bool:
    rows = read_csv(run_a / "health_snapshot.csv")
    truth = json.loads((run_a / "ground_truth.json").read_text(encoding="utf-8"))
    missing = truth["missing_data"]
    flagged = [row for row in rows if int(row["data_flags"]) & schema.FLAG_SPACE_MISSING]
    by_company: dict[str, list[str]] = defaultdict(list)
    for row in flagged:
        by_company[row["company_code"]].append(row["snapshot_month"])
    clubs = {row["site_code"] for row in flagged}
    runs = {code: longest_run(months) for code, months in by_company.items()}
    passed = (
        clubs == {missing["site_code"]}
        and len(by_company) == 100
        and set(runs.values()) == {3}
        and all(row["space_raw"] == "" and row["health_score"] == "" for row in flagged)
    )
    print("AC3  one club's space utilisation missing for three consecutive months")
    print(f"     club {missing['site_code']} (index {missing['site_index']})")
    print(f"     months {', '.join(missing['periods'])}")
    print(
        f"     flagged rows {len(flagged)}; companies {len(by_company)};"
        f" per-company runs {sorted(set(runs.values()))}"
    )
    print(f"     space_raw and health_score are NULL on all flagged rows -> {'PASS' if passed else 'FAIL'}\n")
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-a", default="deliver/run1")
    parser.add_argument("--run-b", default="deliver/run2")
    args = parser.parse_args(argv)

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    for run in (run_a, run_b):
        if not (run / "health_snapshot.csv").exists():
            parser.error(f"no generated run at {run}")

    results = [check_ac1(run_a, run_b), check_ac2(run_a), check_ac3(run_a)]
    print("all acceptance criteria passed" if all(results) else "one or more criteria failed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
