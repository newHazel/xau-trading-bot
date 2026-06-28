#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RunPod backtest of the CORRECTED engine (scope-A fixes: price_zone direction
# gate + Wilder ATR + sizing-suppress + displacement + fvg_freshness + liquidity
# threshold + EMA warmup). One-shot, resumable, runs the whole fleet.
#
#   Crypto (9 coins): committed CSVs ship in the repo → runs OFFLINE, no fetch/key.
#   Gold (XAUUSD)   : OPT-IN (RUN_GOLD=1) — no committed CSV, needs Twelve Data.
#
# ── HOW TO RUN ON A RUNPOD POD ───────────────────────────────────────────────
#   cd /workspace
#   git clone https://github.com/newHazel/xau-trading-bot.git
#   cd xau-trading-bot
#   nohup bash scripts/run_runpod_backtest.sh > /workspace/bt.log 2>&1 &
#   tail -f /workspace/bt.log         # watch; survives a web-terminal disconnect
#
#   Re-score only after a SCORING/cost calc tweak (seconds, reuses checkpoints):
#       AGGREGATE_ONLY=1 bash scripts/run_runpod_backtest.sh
#   Also backtest gold (needs the key):
#       RUN_GOLD=1 TWELVE_DATA_API_KEY=xxxx bash scripts/run_runpod_backtest.sh
#   Override anything via env:
#       SYMBOLS="ETHUSDT SOLUSDT" VARIANT=crypto_pct TOTAL_BARS=15000 JOBS=8 bash ...
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

# --- pick a python that has the deps (RunPod base image = python3.13) ---------
PY="${PY:-}"
if [ -z "$PY" ]; then
  for c in python3.13 python3 python; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
fi
[ -z "$PY" ] && { echo "!! no python found"; exit 1; }
echo "python = $PY ($($PY --version 2>&1))"

# --- deps: fresh pods ship an OLD python-dateutil that imports six.moves without
#     six → `import pandas` crashes. Force-reinstall the two, then the pinned reqs. -
if [ "${SKIP_PIP:-0}" != "1" ]; then
  echo ">>> installing deps (force six + python-dateutil first — fresh-pod six.moves crash)"
  $PY -m pip install --quiet --upgrade pip || true
  $PY -m pip install --quiet --upgrade --force-reinstall six python-dateutil || true
  $PY -m pip install --quiet -r requirements.txt || { echo "!! pip install failed"; exit 1; }
fi

# --- persistence on the RunPod volume (survives restarts; checkpoints = resumable) -
PERSIST_DIR="${PERSIST_DIR:-/workspace/xau_bt}"
if ! mkdir -p "$PERSIST_DIR/bt_checkpoints" 2>/dev/null; then
  echo "!! '$PERSIST_DIR' not writable — using ephemeral ./data/database (attach a volume to persist)"
  PERSIST_DIR="data/database"; mkdir -p "$PERSIST_DIR/bt_checkpoints"
fi
DB_PATH="${DB_PATH:-${PERSIST_DIR}/trading_bot.sqlite}"
OUT_ROOT="${OUT_ROOT:-${PERSIST_DIR}/bt_checkpoints}"

JOBS="${JOBS:-$($PY -c 'import os;print(max(1,(os.cpu_count() or 2)-1))')}"
SYMBOLS="${SYMBOLS:-ETHUSDT DOGEUSDT SOLUSDT LINKUSDT AVAXUSDT NEARUSDT SUIUSDT SANDUSDT ZECUSDT}"
VARIANT="${VARIANT:-crypto_pct}"     # price-proportional costs — the right crypto edge variant
TOTAL_BARS="${TOTAL_BARS:-15000}"
CHUNK_BARS="${CHUNK_BARS:-$(( TOTAL_BARS / JOBS ))}"; [ "${CHUNK_BARS}" -lt 1500 ] 2>/dev/null && CHUNK_BARS=1500
START="${START:-2026-03-01}"
AGG=""; [ "${AGGREGATE_ONLY:-0}" = "1" ] && AGG="--aggregate-only"

echo "=================================================================="
echo "  RUNPOD BACKTEST (corrected engine) | variant=${VARIANT} | JOBS=${JOBS}"
echo "  bars=${TOTAL_BARS} chunk=${CHUNK_BARS}  mode=${AGG:-FULL}"
echo "  persist: ${DB_PATH} + ${OUT_ROOT}/<coin>/"
echo "  coins:  ${SYMBOLS}"
echo "=================================================================="

for SYM in ${SYMBOLS}; do
  OUT_DIR="${OUT_ROOT}/${SYM}"; mkdir -p "$OUT_DIR"   # per-coin: checkpoints key on variant+start, NOT symbol
  echo; echo ">>>>>>>>>>>>>>>>>>>>>>  ${SYM}  <<<<<<<<<<<<<<<<<<<<<<"
  if [ -z "$AGG" ] && [ -f "data/candles/${SYM}/5m.csv" ]; then
    echo ">>> [data] committed CSVs in data/candles/${SYM}/ (offline, no download)"
  elif [ -z "$AGG" ]; then
    echo ">>> [data] no committed CSV — fetching ${SYM} from Binance (public, no key)"
    $PY scripts/fetch_binance_history.py --symbol "${SYM}" --start "${START}" \
      --timeframes 4h,1h,15m,5m --db-path "${DB_PATH}" || { echo "!! fetch failed — skip ${SYM}"; continue; }
  fi
  $PY -u scripts/backtest_sequence_parallel.py --symbol "${SYM}" --execution-tf 5m \
    --total-bars "${TOTAL_BARS}" --chunk-bars "${CHUNK_BARS}" --jobs "${JOBS}" \
    --variants "${VARIANT}" --db-path "${DB_PATH}" --out-dir "${OUT_DIR}" ${AGG}
done

# --- OPTIONAL gold (XAUUSD): no committed CSV → Twelve Data (free tier 8 req/min) ----
if [ "${RUN_GOLD:-0}" = "1" ]; then
  echo; echo ">>>>>>>>>>>>>>>>>>>>>>  XAUUSD (gold)  <<<<<<<<<<<<<<<<<<<<<<"
  if [ -z "${TWELVE_DATA_API_KEY:-}" ]; then
    echo "!! RUN_GOLD=1 but TWELVE_DATA_API_KEY is unset — skipping gold."
  else
    GOUT="${OUT_ROOT}/XAUUSD"; mkdir -p "$GOUT"
    if [ -z "$AGG" ] && [ ! -f "data/candles/XAUUSD/5m.csv" ]; then
      echo ">>> [data] fetching gold per-TF from Twelve Data (HTF first, 65s gaps for 8 req/min)"
      for tf in 4h 1h 15m 5m; do
        $PY scripts/fetch_twelvedata_history.py --months 4 --timeframes "$tf" --db-path "${DB_PATH}" || true
        sleep 65
      done
    fi
    $PY -u scripts/backtest_sequence_parallel.py --symbol XAUUSD --execution-tf 5m \
      --total-bars "${GOLD_BARS:-18000}" --chunk-bars 1500 --jobs "${JOBS}" \
      --variants "${GOLD_VARIANT:-freshness}" --db-path "${DB_PATH}" --out-dir "$GOUT" ${AGG}
  fi
fi

echo
echo "=================================================================="
echo "  DONE — copy each coin's BACKTEST RESULT table (signals / win% / PF / R)."
echo "  Data + signal checkpoints persist under ${OUT_ROOT}."
echo "  Re-score after a calc-only fix (seconds): AGGREGATE_ONLY=1 bash scripts/run_runpod_backtest.sh"
echo "=================================================================="
