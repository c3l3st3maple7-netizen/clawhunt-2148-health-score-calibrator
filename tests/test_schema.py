"""The column contract and the CSV writer."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from calib import schema


def test_snapshot_columns_follow_the_table_convention() -> None:
    columns = list(schema.SNAPSHOT_COLUMNS)
    assert columns[:5] == ["id", "org_id", "site_id", "site_code", "company_code"]
    assert columns[-4:] == ["created_at", "created_by", "updated_at", "deleted_at"]
    for name in schema.SUB_ITEMS:
        assert f"{name}_raw" in columns
        assert f"{name}_score" in columns
    assert "health_score" in columns
    assert "data_flags" in columns


def test_churn_columns_follow_the_table_convention() -> None:
    columns = list(schema.CHURN_COLUMNS)
    assert columns[:5] == ["id", "org_id", "site_id", "site_code", "company_code"]
    assert {"label", "label_reason", "churn_month", "observed_until"} <= set(columns)
    assert columns[-4:] == ["created_at", "created_by", "updated_at", "deleted_at"]


def test_fmt() -> None:
    assert schema.fmt(None) == ""
    assert schema.fmt(1.23456, 2) == "1.23"
    assert schema.fmt(1.0, 3) == "1.000"
    assert schema.fmt(7) == "7"


def test_write_csv_rejects_a_missing_column(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing column"):
        schema.write_csv(str(tmp_path / "out.csv"), ("a", "b"), [{"a": "1"}])


def test_write_csv_uses_lf_endings(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    rows = schema.write_csv(str(path), ("a", "b"), [{"a": "1", "b": "2"}])
    assert rows == 1
    assert path.read_bytes() == b"a,b\n1,2\n"
    with path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"a": "1", "b": "2"}]
