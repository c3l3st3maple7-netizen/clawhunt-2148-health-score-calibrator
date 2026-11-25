# Verification record

Every claim in the delivery note is reproducible from this directory. Raw output
files are listed next to the command that produced them.

Environment

```
Python 3.12.16        deliver/python-version.txt
Linux 6.12.118 x86_64 (2 cores)
pytest 9.2.0          tests only; the generator uses the standard library only
```

| # | Command | Result | Raw output |
|---|---|---|---|
| 1 | `python -m pytest` | 68 passed in 27.89 s | `deliver/pytest.log` |
| 2 | `python -m calib.generate_synthetic_health --config configs/default.yaml --out deliver/run1` | 7 200 + 300 rows, 0.58 s | `deliver/run1.log`, `deliver/run1/` |
| 3 | same command with `--out deliver/run2 --quiet` | identical bytes | `deliver/sha256.txt` |
| 4 | `python scripts/check_acceptance.py --run-a deliver/run1 --run-b deliver/run2` | AC1, AC2, AC3 PASS | `deliver/acceptance.log` |
| 5 | `python examples/recover_weights.py deliver/run1 --json deliver/recovery.json` | hazard fit MAE 0.197, intercept −5.48 vs −5.645 | `deliver/recovery.log`, `deliver/recovery.json` |
| 6 | `python scripts/render_evidence.py --title ... --out deliver/ac2-statistics.png deliver/run1.log deliver/acceptance.log` | screenshot | `deliver/ac2-statistics.png` |
| 7 | fresh copy, no `deliver/` or `runs/`, `python -m pytest` and the generator | 68 passed, acceptance PASS | reproduced 2026-11-25, not stored |

Acceptance results

```
AC1  four output files byte-identical between two runs            PASS
AC2  label rate 36/300 = 12.00 %, model expectation 12.0000 %     PASS (10 %–14 %)
AC3  SYN-SITE-03, 2025-01 .. 2025-03, 300 NULL rows, runs [3]     PASS
```

Notes

* Rows and hashes in `summary.json` are produced by the run itself, so the
  log and the data file can be cross-checked.
* `deliver/run1/` is a complete default run; `ground_truth.json` there is the
  sample referenced by the task.
* Nothing in the verification touches the network, a database or a third-party
  package (pytest and Pillow are used only for tests and the screenshot).
