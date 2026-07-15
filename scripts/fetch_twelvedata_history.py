"""Fetch real spot XAU/USD history from Twelve Data into the SQLite candles table.

Stores under the internal symbol XAUUSD (source="twelvedata"), so it sits
alongside the Bybit XAUUSDT perpetual — you can compare spot vs perpetual, and
scan/chart either by passing the right --symbol.

Free tier: 800 req/day, 8 req/min, 5000 points/call. The script paginates by
date window and sleeps between calls to stay under the rate limit.

Setup:
    1. Get a FREE key (email, no deposit): https://twelvedata.com/pricing
    2. Put it in .env:   TWELVE_DATA_API_KEY=your_key_here
    3. Run:
         python scripts/fetch_twelvedata_history.py --months 6
         python scripts/fetch_twelvedata_history.py --start 2025-01-01 --timeframes 5m,15m,1h,4h
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

from core.logging.db import get_db
from core.logging.system_logger import SystemLogger
from core.data.twelvedata_fetcher import TwelveDataFetcher

STORE_SYMBOL = "XAUUSD"   # internal name we store under
SOURCE = "twelvedata"

# approx seconds per candle — used to size each pagination window (~5000 bars/call)
_TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
_VALID_TFS = list(_TF_SECONDS.keys())
_PER_CALL = 4500            # stay safely under the 5000-point cap
_RATE_SLEEP = 8.0          # seconds between calls (free tier: 8 req/min)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Twelve Data spot XAU/USD into SQLite.")
    p.add_argument("--months", type=int, default=6, help="Months back from now.")
    p.add_argument("--start", type=str, default=None, help="ISO date YYYY-MM-DD (overrides --months).")
    p.add_argument("--end", type=str, default=None, help="ISO date. Defaults to now.")
    p.add_argument("--timeframes", type=str, default="5m,15m,1h,4h",
                   help="Comma list of 1m,5m,15m,1h,4h (1m burns the daily quota fast).")
    p.add_argument("--db-path", type=str, default="data/database/trading_bot.sqlite")
    return p.parse_args()


def fetch_timeframe(fetcher, tf, start, end, logger) -> int:
    window = timedelta(seconds=_TF_SECONDS[tf] * _PER_CALL)
    cursor = start
    total = 0
    reqs = 0
    print(f"\n[{tf}] fetching XAU/USD spot")

    while cursor < end:
        chunk_end = min(cursor + window, end)
        res = fetcher.fetch_candles(STORE_SYMBOL, tf, cursor, chunk_end)
        reqs += 1

        if res.status.value != "ok" or res.data is None or res.data.empty:
            msg = res.error_message or "no data"
            # Rate limit / quota messages from the API surface here.
            print(f"  {cursor.date()}→{chunk_end.date()}: {msg}")
            if "run out of API credits" in str(msg) or "limit" in str(msg).lower():
                print("  [stop] API quota/limit reached — try again later or fewer TFs.")
                break
            cursor = chunk_end
            time.sleep(_RATE_SLEEP)
            continue

        df = res.data
        logger.store_candles(df, STORE_SYMBOL, tf, SOURCE)
        total += len(df)
        last_ts = df.index[-1].to_pydatetime()
        new_cursor = last_ts + timedelta(seconds=_TF_SECONDS[tf])
        cursor = new_cursor if new_cursor > cursor else chunk_end

        print(f"  req#{reqs}: +{len(df)} bars, total {total}, cursor→{cursor.date()}", end="\r")
        time.sleep(_RATE_SLEEP)

    print(f"\n[{tf}] DONE — {total} candles stored ({reqs} requests)")
    return total


def main() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
    args = _parse_args()

    fetcher = TwelveDataFetcher()
    if not fetcher.is_available():
        print("ERROR: Twelve Data not reachable. Check TWELVE_DATA_API_KEY in .env "
              "(get a free key at https://twelvedata.com/pricing).")
        sys.exit(1)

    end = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
           if args.end else datetime.now(timezone.utc))
    start = (datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
             if args.start else end - timedelta(days=args.months * 30))

    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip() in _VALID_TFS]
    if not timeframes:
        print(f"ERROR: no valid timeframes. Choose from {_VALID_TFS}")
        sys.exit(1)

    db = get_db(args.db_path)
    logger = SystemLogger(db)

    print(f"=== Twelve Data fetch: XAU/USD spot | {start.date()} → {end.date()} | TFs {timeframes} ===")
    grand = 0
    for tf in timeframes:
        grand += fetch_timeframe(fetcher, tf, start, end, logger)

    print(f"\n=== COMPLETE: {grand} candles stored as '{STORE_SYMBOL}' (source={SOURCE}) ===")
    for tf in timeframes:
        n = db.fetchone("SELECT COUNT(*) AS n FROM candles WHERE symbol=? AND timeframe=?",
                        (STORE_SYMBOL, tf))
        print(f"  {STORE_SYMBOL} {tf}: {n['n']} rows in DB")
    print("\nNext: python scripts/scan_signals.py --fresh --symbol XAUUSD")


if __name__ == "__main__":
    main()
