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

# nproc reports the HOST cores on a shared container (e.g. 48) → a Pool that big
# oversubscribes the few REAL vCPUs and thrashes/OOMs with no progress. Detect the
# real CPU quota from cgroup (v2 then v1), fall back to nproc, and HARD-cap at 6.
detect_cpus() {
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null || true
    if [ "${q:-max}" != "max" ] && [ "${p:-0}" -gt 0 ] 2>/dev/null; then
      echo $(( (q + p - 1) / p )); return; fi
  fi
  if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)
    p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null)
    if [ "${q:-0}" -gt 0 ] 2>/dev/null && [ "${p:-0}" -gt 0 ] 2>/dev/null; then
      echo $(( (q + p - 1) / p )); return; fi
  fi
  nproc 2>/dev/null || echo 2
}
JOBS="$(detect_cpus)"
[ "${JOBS:-0}" -lt 1 ] 2>/dev/null && JOBS=2
[ "${JOBS}" -gt 6 ] 2>/dev/null && JOBS=6   # cap: avoid oversubscription / OOM
echo "=================================================================="
echo "  SERVER BACKTEST  |  real cores -> JOBS=${JOBS}  (nproc reported $(nproc 2>/dev/null))"
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

echo ">>> [4/4] running ablation: baseline -> freshness(=LIVE) -> sweep_early (~6000 bars / ~4 weeks)"
echo "    sweep_early = LIVE config + #2 (arm the sequence on a provisional wick-sweep)"
echo "    KEY: does sweep_early ADD signals while keeping PF >= freshness?"
echo "    (small chunks -> watch chunks finish ONE BY ONE as progress)"
python -u scripts/backtest_sequence_parallel.py \
  --total-bars 6000 --chunk-bars 1000 --jobs "${JOBS}" \
  --variants baseline,freshness,sweep_early

echo "=================================================================="
echo "  DONE — copy the ABLATION RESULT table above, then DELETE this service."
echo "=================================================================="
sleep 7200  # keep the container alive 2h so the logs remain readable
