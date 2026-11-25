# Table contract — `calib.health_snapshot` / `calib.churn_label`

The task names the two target tables but does not ship their DDL. This document
records exactly what the generator emits, the mapping to those tables, and the
one place to change if the client's DDL uses different column names
(`calib/schema.py`: `SNAPSHOT_COLUMNS`, `CHURN_COLUMNS`).

Both tables follow the project-wide convention: `id` (UUID), `org_id`,
`site_id`, `created_at`, `created_by`, `updated_at`, `deleted_at`, soft delete
by `deleted_at`. Timestamps are ISO-8601 UTC strings. There is no name, email,
phone or address column anywhere.

## `calib.health_snapshot` — one row per company per month

| Column | Type | Null | Meaning |
|---|---|---|---|
| `id` | UUID | no | deterministic `uuid5` of seed + company + month |
| `org_id` | UUID | no | the member company; one per `company_code` |
| `site_id` | UUID | no | the club the company belongs to |
| `site_code` | text | no | readable club code, `SYN-SITE-01` … |
| `company_code` | text | no | `SYN-0001` … (no names or contacts by design) |
| `snapshot_month` | text | no | `YYYY-MM` |
| `industry`, `stage`, `tier` | text | no | attributes sampled from the configured proportions |
| `{event,interaction,resource,opportunity,space}_raw` | decimal(3) | space only | raw monthly sub-item quantity |
| `{…}_score` | decimal(2) | space only | 0–100 standardised score |
| `health_score` | decimal(2) | yes | weighted mean of the five scores; NULL when a sub-item is missing |
| `data_flags` | integer | no | bit 0 = space utilisation deliberately missing |
| `created_at` | timestamp | no | first day of the month |
| `created_by` | text | yes | NULL for synthetic rows |
| `updated_at` | timestamp | no | equal to `created_at` |
| `deleted_at` | timestamp | yes | NULL |

`space_raw` is NULL for every company at the configured club in the configured
three-month window; `space_score` and `health_score` are NULL on the same rows.

## `calib.churn_label` — one row per company

| Column | Type | Null | Meaning |
|---|---|---|---|
| `id` | UUID | no | deterministic `uuid5` of seed + company |
| `org_id`, `site_id`, `site_code`, `company_code` | | no | as above |
| `label` | integer | no | 1 = churned inside the 12-month window |
| `label_reason` | text | no | `renewal_not_signed` (1) / `renewed` (0) |
| `churn_month` | text | yes | `YYYY-MM` of the churn event; NULL when `label = 0` |
| `observed_until` | date | no | last day of the label window |
| `created_at` | timestamp | no | first day of the label window |
| `created_by` | text | yes | NULL |
| `updated_at` | timestamp | no | `observed_until` |
| `deleted_at` | timestamp | yes | NULL |

The label window is the last 12 months of the run (`2025-01` … `2025-12` for the
default 24-month run). `churn_month` is additional information for survival
analyses; a plain binary classifier can ignore it.

## How to reproduce the model inputs from the files

`ground_truth.json → standardisation.stats.<item>` holds the population mean and
standard deviation used for scoring, and `clip_z` the clip bound. To rebuild the
model input for every row:

```
z = clip((raw - mean) / pstdev, -clip_z, +clip_z)
score = clip(50 + 15 * z, 0, 100)
p(churn) = logistic(b0 + Σ w_k * z_k)
```

with `b0` and the true `w_k` in `ground_truth.json → model`. When `space_raw`
is NULL, the logit uses
`ground_truth.json → model.renormalised_weights_when_space_missing`, which scales
the four remaining weights so their sum equals the full five-weight sum.

## Renaming for an existing DDL

If the client's DDL uses different names, edit the two tuples in
`calib/schema.py` only; the generator builds every row from those columns and
the tests fail loudly if a row and the header disagree. A database-side mapping
(e.g. `snapshot_month → period_month`) can also be done with a `SELECT … AS`
view, since the file is plain CSV.
