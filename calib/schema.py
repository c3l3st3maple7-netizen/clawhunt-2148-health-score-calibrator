"""The column contract of ``calib.health_snapshot`` and ``calib.churn_label``.

The task description names the two tables but does not ship their DDL, so this
module is the single place that defines the emitted columns.  The layout follows
the project-wide table convention (``id``, ``org_id``, ``site_id``,
``created_at``, ``created_by``, ``updated_at``, ``deleted_at``) plus the columns
the generator has to produce.  See ``docs/schema.md`` for the mapping and for
the one-line change needed if the client's DDL uses different names.
"""

from __future__ import annotations

import csv
import hashlib
import os
from typing import Iterable, Mapping, Sequence

SUB_ITEMS: tuple[str, ...] = (
    "event",
    "interaction",
    "resource",
    "opportunity",
    "space",
)

#: Bit 0 of ``data_flags`` marks the deliberately removed space utilisation.
FLAG_SPACE_MISSING = 1

SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "id",
    "org_id",
    "site_id",
    "site_code",
    "company_code",
    "snapshot_month",
    "industry",
    "stage",
    "tier",
    *tuple(f"{name}_raw" for name in SUB_ITEMS),
    *tuple(f"{name}_score" for name in SUB_ITEMS),
    "health_score",
    "data_flags",
    "created_at",
    "created_by",
    "updated_at",
    "deleted_at",
)

CHURN_COLUMNS: tuple[str, ...] = (
    "id",
    "org_id",
    "site_id",
    "site_code",
    "company_code",
    "label",
    "label_reason",
    "churn_month",
    "observed_until",
    "created_at",
    "created_by",
    "updated_at",
    "deleted_at",
)

#: Value written for a SQL NULL in the CSV files.
NULL = ""


def fmt(value: float | int | None, places: int = 3) -> str:
    """Format a cell deterministically: fixed decimals, ``NULL`` for None."""
    if value is None:
        return NULL
    if isinstance(value, bool):
        raise TypeError("boolean cells are not part of the schema")
    if isinstance(value, int):
        return str(value)
    return f"{value:.{places}f}"


def write_csv(
    path: str,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, str]],
) -> int:
    """Write ``rows`` as UTF-8 CSV with ``\\n`` line endings; return the row count."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            missing = [column for column in columns if column not in row]
            if missing:
                raise ValueError(f"row is missing column(s) {missing}")
            writer.writerow([row[column] for column in columns])
            count += 1
    return count


def sha256_file(path: str) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
