"""Fetch historical XAU_USD candles from OANDA into the SQLite candles table.

Paginates forward respecting OANDA's 5000-candles-per-request limit (uses
from + count, NOT from + to + count which OANDA rejects). Stores each timeframe
via SystemLogger.store_candles (INSERT OR IGNORE — safe to re-run).

Usage:
    python scripts/fetch_oanda_history.py --months 6
    python scripts/fetch_oanda_history.py --months 6 --timeframes 5m,15m,1h,4h
    python scripts/fetch_oanda_history.py --start 2025-01-01 --end 2025-07-01

Credentials come from .env (OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENVIRONMENT).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv

from core.logging.db import get_db
from core.logging.system_logger import SystemLogger

INSTRUMENT = "XAU_USD"
SOURCE = "oanda"

_TF_MAP = {"1m": "M1", "5m": "M5", "15m": "M15", "1h": "H1", "4h": "H4"}
_MAX_COUNT = 5000

# Approx candle duration in seconds — used to advance the pagination cursor.
_TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch OANDA XAU_USD history into SQLite.")
    p.add_argument("--months", type=int, default=6, help="How many months back from now.")
    p.add_argument("--start", type=str, default=None, help="ISO date (YYYY-MM-DD). Overrides --months.")
    p.add_argument("--end", type=str, default=None, help="ISO date (YYYY-MM-DD). Defaults to now.")
    p.add_argument("--timeframes", type=str, default="1m,5m,15m,1h,4h",
                   help="Comma-separated: subset of 1m,5m,15m,1h,4h")
    p.add_argument("--db-path", type=str, default="data/database/trading_bot.sqlite")
    return p.parse_args()


def _make_client():
    try:
        import oandapyV20
        import oandapyV20.endpoints.instruments as instruments
    except ImportError:
        print("ERROR: oandapyV20 not installed. Run: pip install oandapyV20")
        sys.exit(1)

    api_key = os.environ.get("OANDA_API_KEY", "")
    env = os.environ.get("OANDA_ENVIRONMENT", "practice")
    if not api_key:
        print("ERROR: OANDA_API_KEY not set in .env")
        sys.exit(1)

    client = oandapyV20.API(access_token=api_key, environment=env)
    return client, instruments


def _parse_candles(candles: list) -> pd.DataFrame:
    rows = []
    for c in candles:
        if not c.get("complete", False):
            continue  # skip in-progress candle
        mid = c.get("mid", {})
        rows.append({
            "timestamp": pd.Timestamp(c["time"]).tz_convert("UTC"),
            "open": float(mid.get("o", 0)),
            "high": float(mid.get("h", 0)),
            "low": float(mid.get("l", 0)),
            "close": float(mid.get("c", 0)),
            "volume": float(c.get("volume", 0)),
        })
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    return df.sort_index()


def fetch_timeframe(client, instruments, tf: str, start: datetime, end: datetime,
                    logger: SystemLogger) -> int:
    granularity = _TF_MAP[tf]
    cursor = start
    total_stored = 0
    step = timedelta(seconds=_TF_SECONDS[tf])
    request_count = 0

    print(f"\n[{tf}] fetching {INSTRUMENT} {granularity} from {start.date()} to {end.date()}")

    while cursor < end:
        params = {
            "granularity": granularity,
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": _MAX_COUNT,
            "price": "M",
        }
        req = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
        try:
            client.request(req)
        except Exception as exc:
            print(f"  request failed at {cursor.date()}: {exc}")
            time.sleep(2)
            cursor += step * _MAX_COUNT  # skip ahead to avoid an infinite loop
            continue

        request_count += 1
        candles = req.response.get("candles", [])
        if not candles:
            break

        df = _parse_candles(candles)
        if not df.empty:
            df = df[df.index <= pd.Timestamp(end)]
            if not df.empty:
                logger.store_candles(df, INSTRUMENT, tf, SOURCE)
                total_stored += len(df)
                last_ts = df.index[-1]
            else:
                last_ts = pd.Timestamp(candles[-1]["time"]).tz_convert("UTC")
        else:
            last_ts = pd.Timestamp(candles[-1]["time"]).tz_convert("UTC")

        # Advance cursor just past the last candle we saw.
        new_cursor = last_ts.to_pydatetime() + step
        if new_cursor <= cursor:
            new_cursor = cursor + step * _MAX_COUNT
        cursor = new_cursor

        print(f"  req#{request_count}: +{len(df)} candles, cursor→{cursor.date()} "
              f"(total {total_stored})", end="\r")
        time.sleep(0.15)  # be polite to the API

    print(f"\n[{tf}] DONE — {total_stored} candles stored ({request_count} requests)")
    return total_stored


def main() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
    args = _parse_args()

    end = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
           if args.end else datetime.now(timezone.utc))
    if args.start:
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    else:
        start = end - timedelta(days=args.months * 30)

    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip() in _TF_MAP]
    if not timeframes:
        print(f"ERROR: no valid timeframes. Choose from {list(_TF_MAP)}")
        sys.exit(1)

    client, instruments = _make_client()
    db = get_db(args.db_path)
    logger = SystemLogger(db)

    print(f"=== OANDA fetch: {INSTRUMENT} | {start.date()} → {end.date()} | TFs: {timeframes} ===")
    grand_total = 0
    for tf in timeframes:
        grand_total += fetch_timeframe(client, instruments, tf, start, end, logger)

    print(f"\n=== COMPLETE: {grand_total} total candles stored to {args.db_path} ===")
    for tf in timeframes:
        n = db.fetchone("SELECT COUNT(*) AS n FROM candles WHERE symbol=? AND timeframe=?",
                        (INSTRUMENT, tf))
        print(f"  {INSTRUMENT} {tf}: {n['n']} rows in DB")


if __name__ == "__main__":
    main()
