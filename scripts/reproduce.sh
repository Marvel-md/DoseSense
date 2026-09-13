#!/usr/bin/env bash
#
# Reproduce every number in this repository from scratch.
#
# Each figure in README.md and docs/evaluation.md is produced by this pipeline.
# docs/evaluation.md is rendered from artifacts/metrics.json by a script rather
# than written by hand, so there is no path by which a stale or invented number
# can survive in the documentation.
#
set -euo pipefail

cd "$(dirname "$0")/.."

PATIENTS="${PATIENTS:-600}"

echo "=============================================================="
echo " DoseSense — full reproduction"
echo " cohort size: ${PATIENTS} patients"
echo "=============================================================="

echo
echo "[1/5] Generating the synthetic cohort"
python scripts/generate_data.py --patients "${PATIENTS}"

echo
echo "[2/5] Building the snapshot panel and training the calibrated ensemble"
python ml/train.py

echo
echo "[3/5] Evaluating against the PDC baseline (includes ablation)"
python ml/evaluate.py

echo
echo "[4/5] Rendering docs/evaluation.md from artifacts/metrics.json"
python scripts/make_report.py

echo
echo "[5/5] Running the test suite"
python -m pytest tests/test_dosesense.py -q

if [ "${BENCHMARK:-0}" = "1" ]; then
  echo
  echo "[6/6] Benchmarking seed stability, latency and scaling"
  python scripts/benchmark.py
fi

echo
echo "=============================================================="
echo " Done. Artefacts in ./artifacts, report in docs/evaluation.md"
echo
echo " Start the dashboard:"
echo "   uvicorn backend.app.main:app --reload"
echo "   then open http://127.0.0.1:8000"
echo
echo " For a faster start-up during a live demo:"
echo "   DOSESENSE_PATIENT_LIMIT=120 uvicorn backend.app.main:app"
echo "=============================================================="
