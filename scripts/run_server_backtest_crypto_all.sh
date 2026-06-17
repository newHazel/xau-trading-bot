#!/usr/bin/env bash
# SERVER backtest for the WHOLE crypto fleet with the FIXED (price-proportional) cost
# model — variant 'crypto_pct'. Validates which coins actually have an edge once the
# gold-calibrated absolute costs (which auto-rejected every cheap coin → grade D) are
# replaced by percentage-of-price costs.
#
# Run as a TEMPORARY Railway service — set the service's Custom Start Command to:
#   bash scripts/run_server_backtest_crypto_all.sh
# Binance klines are public (no key needed). Railway's DB starts empty, so each coin's
# history is fetched first. SYMBOLS is overridable via an env var.
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

SYMBOLS="${SYMBOLS:-ETHUSDT DOGEUSDT SOLUSDT LINKUSDT AVAXUSDT NEARUSDT SUIUSDT SANDUSDT}"
START="${START:-2026-03-01}"
TOTAL_BARS="${TOTAL_BARS:-30000}"
VARIANT="${VARIANT:-crypto_pct}"

echo "=================================================================="
echo "  CRYPTO FLEET BACKTEST | variant=${VARIANT} | real cores -> JOBS=${JOBS}"
echo "  coins: ${SYMBOLS}"
echo "=================================================================="

for SYM in ${SYMBOLS}; do
  echo
  echo ">>>>>>>>>>>>>>>>>>>>>>>>  ${SYM}  <<<<<<<<<<<<<<<<<<<<<<<<"
  echo ">>> [1/3] fetching ${SYM} from Binance (public klines)"
  python scripts/fetch_binance_history.py --symbol "${SYM}" --start "${START}" --timeframes 4h,1h,15m,5m || {
    echo "!! fetch failed for ${SYM} — skipping"; continue; }

  echo ">>> [2/3] sanity: every timeframe must have data"
  python - "$SYM" <<'PY' || { echo "!! insufficient data for ${SYM} — skipping"; continue; }
import sys
from core.logging.db import get_db
sym = sys.argv[1]
db = get_db("data/database/trading_bot.sqlite")
bad = [tf for tf in ("4h","1h","15m","5m")
       if db.fetchone("SELECT COUNT(*) AS n FROM candles WHERE symbol=? AND timeframe=?", (sym, tf))["n"] < 300]
print(f"    {sym}: " + ("OK" if not bad else f"MISSING {bad}"))
sys.exit(1 if bad else 0)
PY

  echo ">>> [3/3] backtest ${SYM} (${VARIANT}: 24/7 + price-sanity + %-costs)"
  python -u scripts/backtest_sequence_parallel.py --symbol "${SYM}" \
    --execution-tf 5m --total-bars "${TOTAL_BARS}" --chunk-bars 1500 --jobs "${JOBS}" \
    --variants "${VARIANT}"
done

echo
echo "=================================================================="
echo "  DONE — copy each coin's BACKTEST RESULT table."
echo "  Read the 'crypto_pct' row: PF >= ~1.3 + reasonable win% = real edge."
echo "  (Now that costs are price-proportional, the cheap coins are no longer"
echo "   auto-rejected, so these numbers are MEANINGFUL — unlike the old absolute run.)"
echo "  Keep this service; Redeploy to re-run (or set SYMBOLS / VARIANT env vars)."
echo "=================================================================="
sleep 10800
