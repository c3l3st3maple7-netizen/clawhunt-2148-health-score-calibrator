# 【S6-CAL-003】 Health-score calibration data generator

`generate_synthetic_health.py` produces **fictitious** data that is structurally
identical to `calib.health_snapshot` and `calib.churn_label`, so a calibration
method can be developed and checked before any real member data exists. The
churn process is a logistic model with **known true weights**, and those weights
are written to `ground_truth.json` for the calibration method to try to recover.

Python 3.11+, **standard library only**, no network, no real personal or company
information, and no client name anywhere in the deliverable.

## What is delivered

| Task deliverable | File |
|---|---|
| generator script | `calib/generate_synthetic_health.py` (run as `python -m calib.generate_synthetic_health`) |
| default YAML | `configs/default.yaml` |
| `ground_truth.json` sample | `deliver/run1/ground_truth.json` |
| unit / integration tests | `tests/` — 68 pytest tests |
| README | this file, plus `docs/schema.md` and `DELIVERY_NOTE.md` |
| evidence | `deliver/` — pytest log, both runs, SHA-256 list, acceptance report, recovery report, AC2 screenshot |

## Quick start

Nothing to install for the generator itself:

```bash
python -m calib.generate_synthetic_health --config configs/default.yaml
# -> runs/default/{health_snapshot.csv,churn_label.csv,ground_truth.json,summary.json}

python -m calib.generate_synthetic_health --config configs/default.yaml \
    --out runs/run1 --seed 7 --companies 500 --months 36
```

Tests need pytest only (`pip install -r requirements.txt`):

```bash
python -m pytest            # 68 passed
./scripts/deliver.sh        # rebuilds every artefact under deliver/
python examples/recover_weights.py deliver/run1
```

A default run (300 companies × 24 months) takes about **0.6 s** and writes
about 2 MB.

## Configuration

Every parameter lives in the YAML file; an unknown key, an out-of-range value or
a malformed weight map is a hard error (no hidden defaults). Floats, ints,
booleans, lists, nested maps and `#` comments are supported; block scalars,
anchors and tab indentation are rejected with a line number. The config is read
by a small built-in YAML-subset loader (`calib/miniyaml.py`) so no third-party
package is needed.

| Key | Default | Meaning |
|---|---|---|
| `seed` | 20260158 | the only source of randomness |
| `dataset.companies` | 300 | member companies |
| `dataset.sites` | 3 | clubs; companies are assigned round-robin |
| `dataset.months` | 24 | length of the simulated history |
| `dataset.start_year` / `start_month` | 2024 / 1 | first simulated month |
| `dataset.annual_churn_target` | 0.12 | annual churn the model is solved to |
| `dataset.label_sampling` | `bernoulli` | each company churns independently at its own probability; `exact_count` fixes the churner count at `round(target × companies)` |
| `attributes.{industry,stage,tier}` | see YAML | sampling proportions for company attributes (normalised on load) |
| `latent_activity.phi` | 0.85 | AR(1) persistence of latent activity |
| `latent_activity.innovation_sd` | 0.45 | monthly Gaussian innovation |
| `latent_activity.seasonal_months` | `[1, 2]` | January and February |
| `latent_activity.seasonal_effect` | −0.70 | holiday dip, in activity units |
| `sub_items.<item>.base/loading/noise_sd/company_bias_sd` | per item | raw sub-item equation (below) |
| `true_weights.<item>` | event .55, interaction .42, resource .30, opportunity .48, space .22 | buried churn weights |
| `standardisation.clip_z` | 3.0 | symmetric z clip |
| `standardisation.score_mean` / `score_sd` | 50 / 15 | score = 50 + 15·z |
| `missing_data.site_index` / `start_month_index` / `months` | 2 / 12 / 3 | the deliberately removed space data |

## Outputs

| File | Rows | Content |
|---|---|---|
| `health_snapshot.csv` | 7 200 | company × month: five raw sub-items, five 0–100 scores, health score, data flags |
| `churn_label.csv` | 300 | one annual label per company, churn month, observation window |
| `ground_truth.json` | — | true weights, solved intercept, standardisation statistics, per-company probabilities and labels |
| `summary.json` | — | the statistics behind the acceptance criteria and SHA-256 of every file |

Full column list, types and the mapping to the two target tables: **`docs/schema.md`**.
Columns follow the project convention (`id`, `org_id`, `site_id`, `created_at`,
`created_by`, `updated_at`, `deleted_at`). No name, email or phone column exists;
company identifiers are only `SYN-0001`-style codes.

## Generation model

For company *i* and month *m*:

```
latent[i,m] = 0.85 · latent[i,m-1] + N(0, 0.45²)
activity[i,m] = latent[i,m] + seasonal(month)            # −0.70 in January and February
raw_k[i,m] = max(0, base_k + loading_k · activity[i,m] + bias_k[i] + N(0, noise_k²))
z_k[i,m] = clip((raw_k − mean_k) / sd_k, ±3)             # population mean/sd over non-missing rows
score_k[i,m] = clip(50 + 15 · z_k, 0, 100)
health[i,m] = Σ w_k · score_k / Σ w_k                    # NULL if any score is missing
p[i,m] = logistic(b0 + Σ w_k · z_k)
```

* **Seasonality** is applied to activity, so the holiday dip is visible in all
  five raw sub-items.
* **Company bias** `bias_k[i]` (~N(0, 0.8²)) is drawn once per company. It is
  part of the noise structure: it keeps the five sub-items correlated through
  the latent activity without making them the same series, which is what lets a
  calibration method tell their weights apart.
* **Intercept** `b0` is solved per run by bisection so that the *realised* annual
  churn of the simulated probabilities — `mean_i(1 − Π_m(1 − p[i,m]))` over the
  last 12 months — equals the configured target. It is not the
  `1 − (1 − monthly)^12` approximation.
* **Labels**: with `label_sampling: bernoulli`, each company churns independently
  at its own annual probability, and the churn month is drawn from its monthly
  hazards. That is the model-faithful reading of the task. `exact_count` is
  available when an exactly-on-target churner count is wanted; it selects
  `round(target × companies)` companies proportionally to their probability.
* **Missing data**: at the configured club, for the configured three consecutive
  months, `space_raw`, `space_score` and `health_score` are NULL, `data_flags`
  bit 0 is set, and the logit uses the remaining four weights renormalised to the
  same total (recorded in `ground_truth.json`). Every other row is complete.

`ground_truth.json` is what makes the set useful: true weights, `b0`,
standardisation statistics, the renormalised weights, and per-company annual
probabilities, labels and final latent activity.

## Determinism

The same seed and config produce **byte-identical** files, including in a
different output directory. This holds because:

* one `random.Random(seed)` draws everything in a fixed order (attributes →
  latent start → per-item bias → monthly walk and noise → churn labels);
* outputs contain no timestamps, paths or host names; UUIDs are `uuid5` of the
  seed and the row key;
* floats are written with fixed decimals and files use `\n` endings, so the
  bytes do not depend on the platform.

`--out` does not change the content: `deliver/run1` and `deliver/run2` were
written to two directories from the same seed and match exactly.

## Acceptance criteria — evidence

| # | Criterion | Result |
|---|---|---|
| 1 | Same seed twice → identical SHA-256 | **PASS** — all four output files identical (`deliver/acceptance.log`, `deliver/sha256.txt`) |
| 2 | Annual churn rate 10 %–14 % under default parameters | **PASS** — 36 / 300 = **12.00 %**; model expectation 12.0000 %; monthly mean 1.2217 % (`deliver/ac2-statistics.png`) |
| 3 | One club's space data missing for 3 consecutive months | **PASS** — `SYN-SITE-03`, 2025-01…2025-03, 300 NULL rows, all 100 companies have a 3-month run (`tests/test_generation.py`) |

Raw outputs:

```
AC1  same seed -> identical SHA-256 for every output file
     health_snapshot.csv    identical  1f3084ae30a4a2fd364dd030fa77682e5da54717bc07bb66f7ea6c7ee063eb09
     churn_label.csv        identical  35b661a72191edaa83019ae58062a058ec5e9aed35d3dbf1a48676e7c707327e
     ground_truth.json      identical  b7005da2dc54bccff2ab320f3f9502b6e7b80a48d89cf4870fc72e6110f1c012
     summary.json           identical  c1b432b6dca5982016f56a6deffc7f9e147d9f1b26e9392a34ce64fe1ea17072
     => PASS

AC2  annual churn rate under the default configuration
     churners 36 / companies 300 = 12.00%   window 2025-01..2025-12 (12 months)
     model expected annual rate 12.0000%
     monthly probability mean 1.2217% (min 0.0022%, max 41.7352%)
     acceptance window 10%-14% -> PASS

AC3  one club's space utilisation missing for three consecutive months
     club SYN-SITE-03 (index 2)
     months 2025-01, 2025-02, 2025-03
     flagged rows 300; companies 100; per-company runs [3]
     space_raw and health_score are NULL on all flagged rows -> PASS
```

Reproduce with:

```bash
./scripts/deliver.sh
cat deliver/acceptance.log
```

## Tests

`python -m pytest` → **68 passed** on Python 3.12.16 (Linux, pytest 9.2.0).
The suite covers the YAML subset, configuration validation, the column contract,
AC1–AC3, the missing-data rule (including a second, differently shaped config),
seasonality, score/health arithmetic, identifier and UUID consistency, the
exact-count option, the CLI (including exit codes for bad configs), runs with
`site-packages` disabled, a run with `socket.socket` patched to fail, and the
reference weight-recovery method. No test needs a database or the network.

## Recovering the true weights (reference method)

`examples/recover_weights.py` shows how a calibration method can use the ground
truth. Output on `deliver/run1`:

```
true weights   {'event': 0.55, 'interaction': 0.42, 'resource': 0.3, 'opportunity': 0.48, 'space': 0.22}
true intercept -5.6454
hazard  weights {'event': 0.4617, 'interaction': 0.4822, 'resource': 0.0027, 'opportunity': 0.8231, 'space': 0.0258}
        MAE 0.197  max 0.343  pearson 0.81  rank 0.60  intercept -5.48
annual  weights {'event': 0.6311, 'interaction': -0.0272, ...}  intercept -2.53
```

The **discrete-time hazard** fit (one row per company-month at risk, event in the
churn month) is the estimator that matches the generator; the intercept is
recovered to 0.17. The naive fit on annual labels with averaged monthly scores
is shown to document how much averaging attenuates and biases the coefficients.

At 300 companies there are only 36 churn events, so per-weight standard errors
are large: in this sample `resource` and `space` are underestimated and
`opportunity` is overestimated. The ground truth is exact; recovery precision is
a function of sample size. Generate a larger run
(`--companies 3000 --months 36`) for tighter estimates.

## Integration and limitations

* **Schema**: the task names `calib.health_snapshot` and `calib.churn_label` but
  does not include their DDL. `docs/schema.md` records the emitted contract and
  the two tuples to edit if names differ.
* **No real data**: no client name, no personal or company identifiers beyond
  `SYN-####` codes, no contact data.
* **No network / no third-party SDK**: the generator imports only the standard
  library; a test runs it with site-packages disabled and with socket creation
  blocked.
* **No environment variables or secrets**: the only input is the YAML file named
  by `--config`; nothing is read from the environment, and the run writes no
  host, path or secret into any output file.
* **One missing block per run**: the task asks for one club with three
  consecutive missing months. Several blocks or other missing sub-items would be
  a small extension of `missing_data`.
* **`label_sampling`** changes only how labels are drawn, not the model or the
  ground truth. `bernoulli` is the default because the task defines churn
  probabilistically.

## Layout

```
calib/
  generate_synthetic_health.py   simulation, standardisation, intercept solver, churn labels, CLI
  config.py                      dataclasses, strict validation, defaults
  miniyaml.py                    dependency-free YAML-subset loader
  schema.py                      column contract, CSV writer, SHA-256 helper
configs/
  default.yaml                   the defaults from the task
docs/
  schema.md                      column-by-column contract and mapping
examples/
  recover_weights.py             reference recovery method (hazard fit vs naive annual fit)
scripts/
  check_acceptance.py            AC1/AC2/AC3 checker with a readable report
  deliver.sh                     tests + two runs + hashes + acceptance + recovery
  render_evidence.py             renders a log as a PNG (AC2 screenshot)
tests/                           68 pytest tests
deliver/                         evidence: logs, two runs, hashes, recovery report, AC2 screenshot
```

## Environment used for the verification record

```
Python 3.12.16 (also compatible with 3.11+)
Linux 6.12.118 x86_64, 2 cores
pytest 9.2.0 (tests only; the generator needs nothing)
generation time   0.58 s for 7 200 snapshot rows
```
