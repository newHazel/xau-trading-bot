"""
Export stored candles to per-coin / per-timeframe CSV files inside the project.

Layout (committed to the repo, so the OHLCV data ships with the code — no re-download,
survives Railway restarts, version-controlled):

    data/candles/<SYMBOL>/4h.csv
    data/candles/<SYMBOL>/1h.csv
    data/candles/<SYMBOL>/15m.csv
    data/candles/<SYMBOL>/5m.csv

Each CSV: timestamp(ISO-8601 UTC),open,high,low,close,volume

Usage:
    # one-time / refresh from the DB you already populated:
    python scripts/export_candles_csv.py
    # to refresh the DATA first, run fetch_binance_history.py per coin, then this.
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

DEFAULT_SYMBOLS = ["ETHUSDT", "DOGEUSDT", "SOLUSDT", "LINKUSDT",
                   "AVAXUSDT", "NEARUSDT", "SUIUSDT", "SANDUSDT",
                   "ZECUSDT", "HYPEUSDT"]
TFS = ["4h", "1h", "15m", "5m"]


def main() -> None:
    p = argparse.ArgumentParser(description="Export candles to per-coin/per-tf CSVs.")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--db-path", default="data/database/trading_bot.sqlite")
    p.add_argument("--out-dir", default="data/candles")
    a = p.parse_args()

    db = get_db(a.db_path)
    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    out_root = Path(a.out_dir)
    grand = 0
    for sym in symbols:
        d = out_root / sym
        d.mkdir(parents=True, exist_ok=True)
        for tf in TFS:
            rows = db.fetchall(
                "SELECT timestamp, open, high, low, close, volume FROM candles "
                "WHERE symbol=? AND timeframe=? ORDER BY timestamp ASC", (sym, tf))
            df = pd.DataFrame([dict(r) for r in rows],
                              columns=["timestamp", "open", "high", "low", "close", "volume"])
            path = d / f"{tf}.csv"
            df.to_csv(path, index=False)
            grand += len(df)
            print(f"  {sym}/{tf}.csv — {len(df)} rows")
    print(f"DONE — {grand} candles exported to {out_root}/")


if __name__ == "__main__":
    main()
