#!/usr/bin/env bash
# One-shot SERVER backtest: fetch fresh data, run the legacy/freshness/all ablation,
# print results to the deploy logs. Meant to run as a TEMPORARY Railway service (or on
# any VPS) on the orchestration-fixes branch — the M1 Air is too slow/throttled.
#
# Reads TWELVE_DATA_API_KEY from the environment (set it in the service's Variables).
# After the results print, the script sleeps so the logs stay readable — DELETE the
# service once you've copied the results.
set -uo pipefail
cd "$(dirname "$0")/.."

JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"
echo "=================================================================="
echo "  SERVER BACKTEST  |  cores=${JOBS}  |  $(date -u 2>/dev/null)"
echo "=================================================================="

echo ">>> [1/3] fetching ~4 months of XAU/USD (for HTF warmup + the test window)"
python scripts/fetch_twelvedata_history.py --months 4 --timeframes 5m,15m,1h,4h || {
  echo "FETCH FAILED — check TWELVE_DATA_API_KEY"; sleep 3600; exit 1; }

echo ">>> [2/3] verifying the parallel backtest tool (chunked == sequential)"
python -u scripts/backtest_sequence_parallel.py --verify --jobs "${JOBS}" || {
  echo "VERIFY FAILED — not trusting parallel numbers"; sleep 3600; exit 1; }

echo ">>> [3/3] running ablation: baseline -> freshness -> all (~8000 bars / ~4 weeks)"
python -u scripts/backtest_sequence_parallel.py \
  --total-bars 8000 --chunk-bars 2000 --jobs "${JOBS}" \
  --variants baseline,freshness,all

echo "=================================================================="
echo "  DONE — copy the ABLATION RESULT table above, then DELETE this service."
echo "=================================================================="
sleep 7200  # keep the container alive 2h so the logs remain readable
