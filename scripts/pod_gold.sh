#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ONE-SHOT gold helper for a fresh RunPod pod. Handles deps + gold data fetch
# (only if missing) + the run you ask for. Idempotent & resumable.
#
# HOW TO USE (after cloning the repo into /workspace/xau-trading-bot):
#   export TWELVE_DATA_API_KEY=xxxxxxxx          # your Twelve Data key
#   nohup bash scripts/pod_gold.sh diag  > /workspace/pod.log 2>&1 &   # setup + funnel diagnostic
#   nohup bash scripts/pod_gold.sh bt    > /workspace/pod.log 2>&1 &   # setup + A/B backtest of the fixes
#   nohup bash scripts/pod_gold.sh setup > /workspace/pod.log 2>&1 &   # just deps + fetch, nothing else
#   tail -f /workspace/pod.log            # watch (Ctrl+C to stop watching; the job keeps running)
#
# WHY nohup: closing the browser tab kills FOREGROUND jobs; nohup+& survives it.
# Override anything via env: MAXBARS=3000  TOTAL_BARS=11000  VARIANTS=freshness,gold_relaxbias
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

PY=$(command -v python3.13 || command -v python3 || command -v python)
[ -z "$PY" ] && { echo "!! no python found"; exit 1; }
DB="${DB_PATH:-/workspace/xau_bt/trading_bot.sqlite}"
mkdir -p "$(dirname "$DB")"
CMD="${1:-diag}"
echo "=== pod_gold.sh  cmd=$CMD  python=$PY  db=$DB ==="

echo ">>> [1/3] deps"
$PY -m pip install -q -r requirements.txt || { echo "!! pip failed"; exit 1; }

echo ">>> [2/3] gold data"
HAVE=$($PY -c "import sqlite3; print(sqlite3.connect('$DB').execute(\"SELECT COUNT(*) FROM candles WHERE symbol='XAUUSD' AND timeframe='15m'\").fetchone()[0])" 2>/dev/null || echo 0)
echo "    XAUUSD 15m rows in DB: $HAVE"
if [ "${HAVE:-0}" -lt 5000 ]; then
  if [ -z "${TWELVE_DATA_API_KEY:-}" ]; then
    echo "!! gold data missing and TWELVE_DATA_API_KEY is unset — export it first, then re-run."; exit 1
  fi
  echo "    fetching gold from Twelve Data (~5 min, 65s gaps for the free-tier rate limit)"
  for tf in 4h 1h 15m 5m; do
    $PY scripts/fetch_twelvedata_history.py --months 4 --timeframes "$tf" --db-path "$DB" || true
    sleep 65
  done
else
  echo "    gold data present — skipping fetch"
fi

JOBS=$($PY -c 'import os;print(max(1,(os.cpu_count() or 2)-1))')
echo ">>> [3/3] run: $CMD"
case "$CMD" in
  setup)
    echo "    setup complete — data ready."
    ;;
  diag)
    $PY scripts/diagnose_signals.py --symbol XAUUSD --execution-tf 15m \
      --max-bars "${MAXBARS:-3000}" --db-path "$DB"
    ;;
  bt)
    $PY -u scripts/backtest_sequence_parallel.py --symbol XAUUSD --execution-tf 15m \
      --total-bars "${TOTAL_BARS:-11000}" --chunk-bars 1500 --jobs "$JOBS" \
      --variants "${VARIANTS:-freshness,gold_relaxbias,gold_zone15m,gold_wire02}" \
      --db-path "$DB" --out-dir /workspace/xau_bt/bt_checkpoints/XAUUSD_15m \
      --oos-ratio 0.30 --bootstrap --min-trades 30 --min-oos-trades 10 \
      --baseline freshness --export
    ;;
  *)
    echo "!! unknown cmd '$CMD' — use one of: setup | diag | bt"; exit 1 ;;
esac
echo "=== POD_DONE ($CMD) ==="
