#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RunPod INSTITUTIONAL backtest of the CORRECTED engine. One-shot, resumable, whole fleet.
#
# Runs an ABLATION over 3 crypto variants per coin:
#   crypto_pct   = the live crypto BASELINE (price-proportional costs, 24/7, price-sanity)
#   crypto_mom   = + momentum gate (RSI): don't enter against momentum (no falling knife)
#   crypto_sweep = + sweep-early: arm on the provisional wick (catch the move before it reverses)
# and reports, per coin: fill% (how many signals actually FILLED vs ran away), win%, PF,
# expectancy, maxDD, Sortino, payoff, loss-streak, exit-type + long/short breakdown,
# 95% BOOTSTRAP CIs, a pre-committed 70/30 IS/OOS hold-out, and a Holm-corrected promotion
# verdict per lever. Coins with < MIN_TRADES fills are flagged 'insufficient evidence'.
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
#       SYMBOLS="ETHUSDT SOLUSDT"  VARIANTS=crypto_pct,crypto_mom  TOTAL_BARS=31000
#       OOS_RATIO=0.30  MIN_TRADES=30  JOBS=8  bash scripts/run_runpod_backtest.sh
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
# NOTE: use ${VAR-default} (no colon) so an EXPLICIT empty SYMBOLS="" skips the crypto
# fleet (gold-only run) instead of falling back to the default 9 coins.
SYMBOLS="${SYMBOLS-ETHUSDT DOGEUSDT SOLUSDT LINKUSDT AVAXUSDT NEARUSDT SUIUSDT SANDUSDT ZECUSDT}"
# Ablation levers (each = crypto_pct BASELINE + one change):
#   crypto_mom     +momentum gate (no falling-knife)        crypto_sweep   +sweep-early
#   crypto_funding +funding gate (orthogonal, crowded side) crypto_trend   +EMA200 trend gate (no counter-trend)
#   crypto_slfloor +wider SL band (floor 2x / max 3x ATR — fixes the noise-tight-stop case)
#   crypto_confirm +REAL confirmation gate (rejection candle >= 0.3x ATR that reclaims the POI —
#                  the entry-quality root fix; replaces the weak green/red body-color confirm)
# Default = the highest-value test: baseline vs the real confirmation gate (+ trend).
#   VARIANTS=crypto_pct,crypto_funding  or  ...,crypto_mom,crypto_sweep,crypto_funding,crypto_trend,crypto_slfloor,crypto_confirm
VARIANTS="${VARIANTS:-crypto_pct,crypto_confirm,crypto_trend}"
TOTAL_BARS="${TOTAL_BARS:-31000}"    # full committed window (~3.7 months of 5m bars)
CHUNK_BARS="${CHUNK_BARS:-$(( TOTAL_BARS / JOBS ))}"; [ "${CHUNK_BARS}" -lt 1500 ] 2>/dev/null && CHUNK_BARS=1500
START="${START:-2026-03-01}"
OOS_RATIO="${OOS_RATIO:-0.30}"       # pre-committed 70/30 chronological hold-out
MIN_TRADES="${MIN_TRADES:-30}"       # below this = 'insufficient evidence' (not kept, not dropped)
AGG=""; [ "${AGGREGATE_ONLY:-0}" = "1" ] && AGG="--aggregate-only"
# Institutional report flags (OOS split + bootstrap CIs + min-N gate + trade export)
REPORT_FLAGS="--oos-ratio ${OOS_RATIO} --bootstrap --min-trades ${MIN_TRADES} --min-oos-trades 10 --baseline crypto_pct --export"

echo "=================================================================="
echo "  RUNPOD BACKTEST (corrected engine) | variants=${VARIANTS} | JOBS=${JOBS}"
echo "  bars=${TOTAL_BARS} chunk=${CHUNK_BARS}  OOS=${OOS_RATIO}  minN=${MIN_TRADES}  mode=${AGG:-FULL}"
echo "  persist: ${DB_PATH} + ${OUT_ROOT}/<coin>/"
echo "  coins:  ${SYMBOLS}"
echo "=================================================================="

for SYM in ${SYMBOLS}; do
  OUT_DIR="${OUT_ROOT}/${SYM}"; mkdir -p "$OUT_DIR"   # per-coin dirs (tidy; the key now covers symbol too)
  # NOTE: checkpoints are now keyed (variant, start, END, config-hash incl. symbol/tf/
  # schema) — old-format sig_*.json files are ignored, so the first run after the
  # look-ahead fix (schema v2) regenerates everything. Clean old files at will.
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
    --variants "${VARIANTS}" --db-path "${DB_PATH}" --out-dir "${OUT_DIR}" ${REPORT_FLAGS} ${AGG}
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
      --variants "${GOLD_VARIANTS:-freshness,gold_confirm,gold_sweepq,gold_kill,gold_v32}" --db-path "${DB_PATH}" \
      --out-dir "$GOUT" --oos-ratio "${OOS_RATIO}" --bootstrap --min-trades "${MIN_TRADES}" \
      --min-oos-trades 10 --baseline freshness --export ${AGG}
  fi
fi

echo
echo "=================================================================="
echo "  DONE — copy each coin's BACKTEST RESULT table (signals / win% / PF / R)."
echo "  Data + signal checkpoints persist under ${OUT_ROOT}."
echo "  Re-score after a calc-only fix (seconds): AGGREGATE_ONLY=1 bash scripts/run_runpod_backtest.sh"
echo "=================================================================="
