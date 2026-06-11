#!/usr/bin/env bash
# One-shot SERVER backtest: fetch fresh data, run the legacy/freshness/all ablation,
# print results to the deploy logs. Meant to run as a TEMPORARY Railway service (or
# any VPS) on the orchestration-fixes branch — the M1 Air is too slow/throttled.
#
# Reads TWELVE_DATA_API_KEY from the environment (set it in the service's Variables).
# Fetches each timeframe SEPARATELY with a gap, because Twelve Data free is 8 req/min
# and one combined call let the big 5m fetch starve the HTF (they came back with 0
# rows, which means no bias -> 0 signals). HTF first, 5m last.
set -uo pipefail
cd "$(dirname "$0")/.."

JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"
echo "=================================================================="
echo "  SERVER BACKTEST  |  cores=${JOBS}"
echo "=================================================================="

echo ">>> [1/4] fetching per-TF (HTF first) with 65s gaps to respect 8 req/min"
python scripts/fetch_twelvedata_history.py --months 4 --timeframes 4h  || true
sleep 65
python scripts/fetch_twelvedata_history.py --months 4 --timeframes 1h  || true
sleep 65
python scripts/fetch_twelvedata_history.py --months 4 --timeframes 15m || true
sleep 65
python scripts/fetch_twelvedata_history.py --months 4 --timeframes 5m  || true

echo ">>> [2/4] sanity: every timeframe must have data (no HTF = no bias = 0 signals)"
python - <<'PY' || { echo "ABORT — insufficient data; re-deploy to retry (limits reset each minute / 800 per day)."; sleep 3600; exit 1; }
import sys
from core.logging.db import get_db
db = get_db("data/database/trading_bot.sqlite")
bad = []
for tf in ("4h", "1h", "15m", "5m"):
    n = db.fetchone("SELECT COUNT(*) AS n FROM candles WHERE symbol='XAUUSD' AND timeframe=?", (tf,))["n"]
    print(f"    {tf}: {n} rows")
    if n < 300:
        bad.append(tf)
sys.exit(1 if bad else 0)
PY

echo ">>> [3/4] verifying the parallel backtest tool (chunked == sequential)"
python -u scripts/backtest_sequence_parallel.py --verify --jobs "${JOBS}" || {
  echo "VERIFY FAILED — not trusting parallel numbers"; sleep 3600; exit 1; }

echo ">>> [4/4] running ablation: baseline -> freshness -> all (~8000 bars / ~6 weeks)"
python -u scripts/backtest_sequence_parallel.py \
  --total-bars 8000 --chunk-bars 2000 --jobs "${JOBS}" \
  --variants baseline,freshness,all

echo "=================================================================="
echo "  DONE — copy the ABLATION RESULT table above, then DELETE this service."
echo "=================================================================="
sleep 7200  # keep the container alive 2h so the logs remain readable
