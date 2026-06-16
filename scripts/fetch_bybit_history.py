"""Fetch historical XAUUSDT candles from Bybit (public, no API key) into SQLite.

Bybit returns klines newest-first, max 1000/request. We paginate FORWARD by
advancing the cursor to (newest_ts + one interval) after each batch. Stored via
SystemLogger.store_candles (INSERT OR IGNORE — safe to re-run).

NOTE: XAUUSDT is a USDT-margined perpetual that tracks gold — NOT spot XAU/USD.
Use for relative A/B testing (indicators on vs off), not absolute live validation.

Usage:
    python scripts/fetch_bybit_history.py --start 2026-03-01
    python scripts/fetch_bybit_history.py --start 2026-03-01 --timeframes 5m,15m,1h,4h
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from core.logging.db import get_db
from core.logging.system_logger import SystemLogger

SYMBOL = "XAUUSDT"
SOURCE = "bybit"
CATEGORY = "linear"

_TF_MAP = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240"}
_TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
_MAX_LIMIT = 1000


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Bybit XAUUSDT history into SQLite.")
    p.add_argument("--start", type=str, default="2026-03-01", help="ISO date (YYYY-MM-DD).")
    p.add_argument("--end", type=str, default=None, help="ISO date. Defaults to now.")
    p.add_argument("--timeframes", type=str, default="1m,5m,15m,1h,4h",
                   help="Comma-separated subset of 1m,5m,15m,1h,4h")
    p.add_argument("--db-path", type=str, default="data/database/trading_bot.sqlite")
    p.add_argument("--symbol", type=str, default="XAUUSDT",
                   help="Bybit linear perp symbol, e.g. ETHUSDT, SOLUSDT.")
    return p.parse_args()


def _make_session():
    try:
        from pybit.unified_trading import HTTP
    except ImportError:
        print("ERROR: pybit not installed. Run: pip install pybit")
        sys.exit(1)
    return HTTP()  # no key — public market data only


def _parse_rows(rows: list, end_ms: int) -> pd.DataFrame:
    parsed = []
    for row in rows:
        ts_ms = int(row[0])
        if ts_ms >= end_ms:
            continue
        parsed.append({
            "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
            "open": float(row[1]), "high": float(row[2]),
            "low": float(row[3]), "close": float(row[4]),
            "volume": float(row[5]),
        })
    if not parsed:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(parsed).sort_values("timestamp").set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    return df


def fetch_timeframe(session, tf: str, start_ms: int, end_ms: int, logger: SystemLogger) -> int:
    interval = _TF_MAP[tf]
    step_ms = _TF_MS[tf]
    cursor = start_ms
    total = 0
    reqs = 0

    print(f"\n[{tf}] fetching {SYMBOL} interval={interval}")

    while cursor < end_ms:
        try:
            resp = session.get_kline(category=CATEGORY, symbol=SYMBOL,
                                     interval=interval, start=cursor, limit=_MAX_LIMIT)
        except Exception as exc:
            print(f"  request failed at {datetime.utcfromtimestamp(cursor/1000).date()}: {exc}")
            time.sleep(2)
            cursor += step_ms * _MAX_LIMIT
            continue

        rows = resp.get("result", {}).get("list", [])
        if not rows:
            break

        reqs += 1
        df = _parse_rows(rows, end_ms)
        if not df.empty:
            logger.store_candles(df, SYMBOL, tf, SOURCE)
            total += len(df)

        newest_ts = max(int(r[0]) for r in rows)  # rows are newest-first
        new_cursor = newest_ts + step_ms
        if new_cursor <= cursor:
            new_cursor = cursor + step_ms * _MAX_LIMIT
        cursor = new_cursor

        last_date = datetime.utcfromtimestamp(newest_ts / 1000)
        print(f"  req#{reqs}: total {total} candles, cursor→{last_date.date()}", end="\r")
        time.sleep(0.12)

    print(f"\n[{tf}] DONE — {total} candles stored ({reqs} requests)")
    return total


def main() -> None:
    global SYMBOL
    args = _parse_args()
    SYMBOL = args.symbol  # allow any Bybit linear perp (ETHUSDT, SOLUSDT, ...)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
           if args.end else datetime.now(timezone.utc))
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip() in _TF_MAP]
    if not timeframes:
        print(f"ERROR: no valid timeframes. Choose from {list(_TF_MAP)}")
        sys.exit(1)

    session = _make_session()
    db = get_db(args.db_path)
    logger = SystemLogger(db)

    print(f"=== Bybit fetch: {SYMBOL} | {start.date()} → {end.date()} | TFs: {timeframes} ===")
    grand_total = 0
    for tf in timeframes:
        grand_total += fetch_timeframe(session, tf, start_ms, end_ms, logger)

    print(f"\n=== COMPLETE: {grand_total} total candles stored to {args.db_path} ===")
    for tf in timeframes:
        n = db.fetchone("SELECT COUNT(*) AS n FROM candles WHERE symbol=? AND timeframe=?",
                        (SYMBOL, tf))
        print(f"  {SYMBOL} {tf}: {n['n']} rows in DB")


if __name__ == "__main__":
    main()
