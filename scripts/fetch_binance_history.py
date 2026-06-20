"""Fetch historical CRYPTO candles from Binance into SQLite.

Gold (XAUUSD) stays on Twelve Data; crypto (ETHUSDT, SOLUSDT, ...) uses Binance.
Binance klines are PUBLIC (no signature needed for candles), but we send the
BINANCE_API_KEY from .env as the X-MBX-APIKEY header so the request uses YOUR key
(higher rate limits / IP binding). Falls back to the public data host if the main
API host is geo-blocked. Stored via SystemLogger.store_candles (INSERT OR IGNORE —
safe to re-run), tz-aware UTC (passes the data validator).

Usage:
    python scripts/fetch_binance_history.py --symbol ETHUSDT --start 2026-02-16 \
        --timeframes 5m,15m,1h,4h
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
from dotenv import load_dotenv

from core.logging.db import get_db
from core.logging.system_logger import SystemLogger

SOURCE = "binance"
# Binance interval strings == our TF labels (no remapping needed).
_TF = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h"}
_TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
_MAX_LIMIT = 1000
# Spot hosts first; USD-M FUTURES last as a fallback for symbols not listed on spot
# (e.g. HYPEUSDT). Futures klines share the spot array format. (base_url, klines_path)
_ENDPOINTS = [
    ("https://api.binance.com", "/api/v3/klines"),
    ("https://data-api.binance.vision", "/api/v3/klines"),
    ("https://fapi.binance.com", "/fapi/v1/klines"),
]


def _headers() -> dict:
    key = (os.getenv("BINANCE_API_KEY") or "").strip()
    return {"X-MBX-APIKEY": key} if key else {}


def _get_klines(symbol: str, interval: str, start_ms: int, headers: dict) -> list:
    last = None
    for base, path in _ENDPOINTS:
        try:
            r = requests.get(f"{base}{path}",
                             params={"symbol": symbol, "interval": interval,
                                     "startTime": start_ms, "limit": _MAX_LIMIT},
                             headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json()
            last = f"{base} HTTP {r.status_code}: {r.text[:140]}"
        except Exception as exc:  # network / timeout — try the next endpoint
            last = f"{base}: {exc}"
    raise RuntimeError(last)


def _parse(rows: list, end_ms: int) -> pd.DataFrame:
    # Binance kline: [openTime, open, high, low, close, volume, closeTime, ...]
    parsed = []
    for row in rows:
        ts_ms = int(row[0])
        if ts_ms >= end_ms:
            continue
        parsed.append({
            "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
            "open": float(row[1]), "high": float(row[2]),
            "low": float(row[3]), "close": float(row[4]), "volume": float(row[5]),
        })
    if not parsed:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(parsed).sort_values("timestamp").set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    return df


def fetch_timeframe(symbol, tf, start_ms, end_ms, logger, headers) -> int:
    interval, step = _TF[tf], _TF_MS[tf]
    cursor, total, reqs = start_ms, 0, 0
    print(f"\n[{tf}] fetching {symbol} (binance)")
    while cursor < end_ms:
        try:
            rows = _get_klines(symbol, interval, cursor, headers)
        except Exception as exc:
            print(f"  request failed: {exc}")
            time.sleep(2); cursor += step * _MAX_LIMIT; continue
        if not rows:
            break
        reqs += 1
        df = _parse(rows, end_ms)
        if not df.empty:
            logger.store_candles(df, symbol, tf, SOURCE)
            total += len(df)
        newest = max(int(r[0]) for r in rows)  # Binance is oldest-first
        cursor = max(newest + step, cursor + step * _MAX_LIMIT) if newest + step <= cursor else newest + step
        print(f"  req#{reqs}: total {total}, cursor→{datetime.utcfromtimestamp(newest/1000).date()}", end="\r")
        time.sleep(0.15)
    print(f"\n[{tf}] DONE — {total} candles stored ({reqs} requests)")
    return total


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch Binance crypto history into SQLite.")
    p.add_argument("--symbol", default="ETHUSDT", help="e.g. ETHUSDT, SOLUSDT, AVAXUSDT")
    p.add_argument("--start", default="2026-03-01", help="ISO date (YYYY-MM-DD).")
    p.add_argument("--end", default=None, help="ISO date. Defaults to now.")
    p.add_argument("--timeframes", default="5m,15m,1h,4h")
    p.add_argument("--db-path", default="data/database/trading_bot.sqlite")
    a = p.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    headers = _headers()
    start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    end = (datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
           if a.end else datetime.now(timezone.utc))
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    tfs = [t.strip() for t in a.timeframes.split(",") if t.strip() in _TF]
    if not tfs:
        print(f"ERROR: no valid timeframes. Choose from {list(_TF)}"); sys.exit(1)

    db = get_db(a.db_path); logger = SystemLogger(db)
    print(f"=== Binance fetch: {a.symbol} | {start.date()} → {end.date()} | "
          f"key={'yes' if headers else 'public'} | TFs {tfs} ===")
    grand = 0
    for tf in tfs:
        grand += fetch_timeframe(a.symbol, tf, start_ms, end_ms, logger, headers)
    print(f"\n=== COMPLETE: {grand} candles stored as '{a.symbol}' (source=binance) ===")
    for tf in tfs:
        n = db.fetchone("SELECT COUNT(*) AS n FROM candles WHERE symbol=? AND timeframe=?", (a.symbol, tf))
        print(f"  {a.symbol} {tf}: {n['n']} rows in DB")


if __name__ == "__main__":
    main()
