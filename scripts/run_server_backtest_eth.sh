#!/usr/bin/env bash
# SERVER backtest for CRYPTO (ETHUSDT via Binance). Gold stays on Twelve Data.
# Run as a TEMPORARY Railway service — set the service's Custom Start Command to:
#   bash scripts/run_server_backtest_eth.sh
# Binance klines are public (no key needed); the fetcher sends BINANCE_API_KEY if set.
set -uo pipefail
cd "$(dirname "$0")/.."

detect_cpus() {  # Railway nproc reports HOST cores → cap to the real vCPU quota
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null || true
    if [ "${q:-max}" != "max" ] && [ "${p:-0}" -gt 0 ] 2>/dev/null; then echo $(( (q + p - 1) / p )); return; fi
  fi
  nproc 2>/dev/null || echo 2
}
JOBS="$(detect_cpus)"; [ "${JOBS:-0}" -lt 1 ] 2>/dev/null && JOBS=2; [ "${JOBS}" -gt 6 ] 2>/dev/null && JOBS=6
SYMBOL="${SYMBOL:-ETHUSDT}"   # override via a SYMBOL env var to test other coins
echo "=================================================================="
echo "  CRYPTO BACKTEST | ${SYMBOL} (Binance) | real cores -> JOBS=${JOBS}"
echo "=================================================================="

echo ">>> [1/4] fetching ${SYMBOL} from Binance (public klines — fast, no quota)"
python scripts/fetch_binance_history.py --symbol "${SYMBOL}" --start 2026-03-01 --timeframes 4h,1h,15m,5m

echo ">>> [2/4] sanity: every timeframe must have data"
python - "$SYMBOL" <<'PY' || { echo "ABORT — insufficient data; redeploy to retry."; sleep 1800; exit 1; }
import sys
from core.logging.db import get_db
sym = sys.argv[1]
db = get_db("data/database/trading_bot.sqlite")
bad = []
for tf in ("4h", "1h", "15m", "5m"):
    n = db.fetchone("SELECT COUNT(*) AS n FROM candles WHERE symbol=? AND timeframe=?", (sym, tf))["n"]
    print(f"    {tf}: {n} rows")
    if n < 300:
        bad.append(tf)
sys.exit(1 if bad else 0)
PY

echo ">>> [3/4] verifying the parallel tool (chunked == sequential) on ${SYMBOL}"
python -u scripts/backtest_sequence_parallel.py --symbol "${SYMBOL}" --verify --jobs "${JOBS}" || {
  echo "VERIFY FAILED — not trusting numbers"; sleep 1800; exit 1; }

echo ">>> [4/4] backtest ${SYMBOL}: freshness(gold-sessions) vs crypto(24/7, the right one)"
echo "    KEY: does our SMC strategy produce an edge on ETH? (signals, win%, PF, R)"
echo "    crypto = LIVE config + price-sanity + 24/7 (ignore gold kill-zone)."
python -u scripts/backtest_sequence_parallel.py --symbol "${SYMBOL}" \
  --execution-tf 5m --total-bars 12000 --chunk-bars 1500 --jobs "${JOBS}" \
  --variants freshness,crypto

echo "=================================================================="
echo "  DONE — copy the BACKTEST RESULT table. 'crypto' (24/7) is the relevant row for ETH."
echo "  Keep this service; just Redeploy to re-run (or set SYMBOL var for another coin)."
echo "=================================================================="
sleep 10800
