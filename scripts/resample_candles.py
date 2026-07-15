"""Derive a coarser timeframe from a finer one already in the candle DB.

Needed because Twelve Data has NO 3min interval — the only way to a 3m gold
series (without the datacenter-blocked Dukascopy) is: fetch 1min, then resample
1m → 3m here. Also works for any finer→coarser pair the Resampler supports
(1m→3m/5m/15m, 5m→15m, 15m→1h, 1h→4h, …). Wires the previously-dormant
core.data.Resampler into a real path.

    # after: fetch_twelvedata_history.py --timeframes 1m ...
    python scripts/resample_candles.py --symbol XAUUSD --from-tf 1m --to-tf 3m \
        --db-path /workspace/xau_bt/trading_bot.sqlite

Idempotent: store_candles is INSERT OR IGNORE, so re-running never duplicates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from core.logging.db import get_db
from core.logging.system_logger import SystemLogger
from core.data.resampler import Resampler, SUPPORTED_TIMEFRAMES

SOURCE = "resampled"


def _load(db, symbol: str, tf: str) -> pd.DataFrame:
    rows = db.fetchall(
        "SELECT timestamp,open,high,low,close,volume FROM candles "
        "WHERE symbol=? AND timeframe=? ORDER BY timestamp ASC", (symbol, tf))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--from-tf", required=True, help="Source (finer) TF already in the DB, e.g. 1m.")
    p.add_argument("--to-tf", required=True, help="Target (coarser) TF to produce, e.g. 3m.")
    p.add_argument("--db-path", default="data/database/trading_bot.sqlite")
    p.add_argument("--store-symbol", default=None)
    a = p.parse_args()

    for tf in (a.from_tf, a.to_tf):
        if tf not in SUPPORTED_TIMEFRAMES:
            sys.exit(f"unsupported timeframe {tf!r} (allowed: {list(SUPPORTED_TIMEFRAMES)})")
    if pd.Timedelta(SUPPORTED_TIMEFRAMES[a.from_tf]) >= pd.Timedelta(SUPPORTED_TIMEFRAMES[a.to_tf]):
        sys.exit(f"--from-tf ({a.from_tf}) must be FINER than --to-tf ({a.to_tf}).")

    store_symbol = a.store_symbol or a.symbol
    db = get_db(a.db_path)
    src = _load(db, a.symbol, a.from_tf)
    if src.empty:
        sys.exit(f"no {a.from_tf} candles for {a.symbol} in {a.db_path} — fetch them first.")

    print(f"=== resample {a.symbol} {a.from_tf} → {a.to_tf} | "
          f"{len(src)} source bars ({src.index[0]:%Y-%m-%d} → {src.index[-1]:%Y-%m-%d}) ===")
    out = Resampler(base_timeframe=a.from_tf).resample_one(src, a.to_tf, now=None)
    if out.empty:
        sys.exit("resample produced 0 bars (source too short?).")

    logger = SystemLogger(db)
    stored = logger.store_candles(out, store_symbol, a.to_tf, SOURCE)
    print(f"  produced {len(out)} {a.to_tf} bars "
          f"({out.index[0]:%Y-%m-%d %H:%M} → {out.index[-1]:%Y-%m-%d %H:%M}), stored {stored} "
          f"(INSERT OR IGNORE — existing rows skipped)")


if __name__ == "__main__":
    main()
