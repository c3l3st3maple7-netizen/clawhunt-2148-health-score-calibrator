#!/usr/bin/env bash
# Build the delivery evidence: tests, two independent runs of the same seed,
# SHA-256 comparison, the acceptance report and the weight-recovery demo.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
mkdir -p deliver
rm -rf deliver/run1 deliver/run2

echo "== python"
"$PYTHON" -V | tee deliver/python-version.txt

echo "== pytest"
"$PYTHON" -m pytest 2>&1 | tee deliver/pytest.log

echo "== run 1"
"$PYTHON" -m calib.generate_synthetic_health \
  --config configs/default.yaml --out deliver/run1 | tee deliver/run1.log

echo "== run 2 (same seed, separate directory)"
"$PYTHON" -m calib.generate_synthetic_health \
  --config configs/default.yaml --out deliver/run2 --quiet

echo "== sha256"
sha256sum deliver/run1/* deliver/run2/* | tee deliver/sha256.txt

echo "== acceptance criteria"
"$PYTHON" scripts/check_acceptance.py --run-a deliver/run1 --run-b deliver/run2 \
  | tee deliver/acceptance.log

echo "== weight recovery (reference method)"
"$PYTHON" examples/recover_weights.py deliver/run1 --json deliver/recovery.json \
  | tee deliver/recovery.log

echo
echo "done: deliver/pytest.log, deliver/acceptance.log, deliver/recovery.log"
