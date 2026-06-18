#!/usr/bin/env bash
# CRYPTO FLEET BACKTEST — persistent + fast re-run.
#
# The slow part of a backtest is SIGNAL GENERATION (the engine re-deciding every
# bar), NOT the data download. So to make re-runs short we persist BOTH the candle
# DB and the generated signal checkpoints on a Railway VOLUME, and offer a fast
# "re-score only" mode that skips generation entirely.
#
# ── ONE-TIME SETUP ───────────────────────────────────────────────────────────
#   1. Railway → this service → add a VOLUME, mount path:  /data
#   2. Custom Start Command:  bash scripts/run_server_backtest_crypto_all.sh
#
# ── USAGE ────────────────────────────────────────────────────────────────────
#   First run (full — fetches data + generates signals + scores, ~hours, ONCE):
#       bash scripts/run_server_backtest_crypto_all.sh
#   After fixing a SCORING / cost CALC (re-score cached signals — SECONDS):
#       AGGREGATE_ONLY=1 bash scripts/run_server_backtest_crypto_all.sh
#   (Signals only need regenerating if the SIGNAL LOGIC / gates change.)
#
# Everything lives under the volume ($PERSIST_DIR), so a redeploy/restart no longer
# wipes the data or the progress — the run RESUMES instead of starting from ETH.
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
TOTAL_BARS="${TOTAL_BARS:-15000}"
VARIANT="${VARIANT:-crypto_pct}"

# Persist DB + checkpoints on the volume. Falls back to the local repo dir if the
# volume mount isn't writable (then it's ephemeral, like before — attach the volume!).
PERSIST_DIR="${PERSIST_DIR:-/data}"
if ! mkdir -p "$PERSIST_DIR/bt_checkpoints" 2>/dev/null; then
  echo "!! '$PERSIST_DIR' not writable — no volume? Falling back to ephemeral local dir."
  PERSIST_DIR="data/database"; mkdir -p "$PERSIST_DIR/bt_checkpoints"
fi
DB_PATH="${DB_PATH:-${PERSIST_DIR}/trading_bot.sqlite}"
OUT_ROOT="${OUT_DIR:-${PERSIST_DIR}/bt_checkpoints}"

# Fewer, bigger chunks (≈ one per vCPU) = full parallelism + less warmup re-processing.
CHUNK_BARS="${CHUNK_BARS:-$(( TOTAL_BARS / JOBS ))}"
[ "${CHUNK_BARS}" -lt 1500 ] 2>/dev/null && CHUNK_BARS=1500

# Fast re-score mode: skip fetch + signal-gen, just re-score the cached checkpoints.
AGG=""
if [ "${AGGREGATE_ONLY:-0}" = "1" ]; then AGG="--aggregate-only"; fi

echo "=================================================================="
echo "  CRYPTO FLEET BACKTEST | variant=${VARIANT} | JOBS=${JOBS} | bars=${TOTAL_BARS} chunk=${CHUNK_BARS}"
echo "  persist: ${DB_PATH}  +  ${OUT_ROOT}/<coin>/"
echo "  mode: ${AGG:-FULL (fetch+generate+score)}"
echo "  coins: ${SYMBOLS}"
echo "=================================================================="

for SYM in ${SYMBOLS}; do
  OUT_DIR="${OUT_ROOT}/${SYM}"; mkdir -p "$OUT_DIR"   # per-coin so signals don't mix
  echo
  echo ">>>>>>>>>>>>>>>>>>>>>>>>  ${SYM}  <<<<<<<<<<<<<<<<<<<<<<<<"

  if [ -z "$AGG" ]; then
    echo ">>> [1/3] fetching ${SYM} (INSERT OR IGNORE — cached rows on the volume are skipped)"
    python scripts/fetch_binance_history.py --symbol "${SYM}" --start "${START}" \
      --timeframes 4h,1h,15m,5m --db-path "${DB_PATH}" || { echo "!! fetch failed — skipping ${SYM}"; continue; }
    echo ">>> [2/3] sanity: every timeframe must have data"
    python - "$SYM" "$DB_PATH" <<'PY' || { echo "!! insufficient data — skipping ${SYM}"; continue; }
import sys
from core.logging.db import get_db
sym, dbp = sys.argv[1], sys.argv[2]
db = get_db(dbp)
bad = [tf for tf in ("4h","1h","15m","5m")
       if db.fetchone("SELECT COUNT(*) AS n FROM candles WHERE symbol=? AND timeframe=?", (sym, tf))["n"] < 300]
print(f"    {sym}: " + ("OK" if not bad else f"MISSING {bad}")); sys.exit(1 if bad else 0)
PY
  else
    echo ">>> [re-score] using cached signals in ${OUT_DIR} (no fetch, no generation)"
  fi

  echo ">>> [3/3] backtest ${SYM} (${VARIANT})"
  python -u scripts/backtest_sequence_parallel.py --symbol "${SYM}" --execution-tf 5m \
    --total-bars "${TOTAL_BARS}" --chunk-bars "${CHUNK_BARS}" --jobs "${JOBS}" \
    --variants "${VARIANT}" --db-path "${DB_PATH}" --out-dir "${OUT_DIR}" ${AGG}
done

echo
echo "=================================================================="
echo "  DONE — copy each coin's BACKTEST RESULT table."
echo "  Data + signals are saved on the volume. To re-score after a calc fix:"
echo "      AGGREGATE_ONLY=1 bash scripts/run_server_backtest_crypto_all.sh   (seconds)"
echo "=================================================================="
sleep 10800
