"""Dukascopy tick-history fetcher → OHLCV candles for the backtest DB.

WHY: the whole project is gated on data quantity — every gold ablation ends
!INSUFF at 5-18 fills because Twelve Data's free tier caps XAU history at ~4
months. Dukascopy publishes FREE tick data years deep, which is the one source
that can clear the min-N gate (N>=30-100) and settle whether gold_kill is a real
edge or noise. As a bonus it yields a real bid/ask SPREAD model per hour-of-day
(the review's 'empirical cost model' item), instead of the hardcoded 0.25.

FORMAT (public, no key): one LZMA-compressed .bi5 file per instrument per HOUR:
  https://datafeed.dukascopy.com/datafeed/{INST}/{YYYY}/{MM0}/{DD}/{HH}h_ticks.bi5
  MM0 is 0-indexed (January = 00). Each decompressed record is 20 bytes,
  big-endian '>IIIff': ms-since-hour, ask(int pts), bid(int pts), askVol, bidVol.
  Real price = int / point_divisor (XAUUSD = 3 digits → 1000). No file / empty =
  market closed (weekend/holiday) → skipped.

MEMORY: ticks are accumulated per UTC DAY and flushed (resampled → stored) at
each day rollover, so peak memory is one day of ticks — safe for years of range.
Downloaded .bi5 files are cached to --cache-dir so a re-run never re-downloads.

RUN (do the heavy multi-year pull on RunPod, not the laptop):
    python scripts/fetch_dukascopy_history.py --symbol XAUUSD \
        --start 2019-01-01 --end 2026-01-01 \
        --timeframes 5m,15m,1h,4h \
        --db-path /workspace/xau_bt/trading_bot.sqlite \
        --cache-dir /workspace/duka_cache

Then backtest exactly as before, pointing --db-path at the same file. The
committed data/candles CSVs are NOT touched — this only fills the SQLite DB.
"""
from __future__ import annotations

import argparse
import lzma
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import requests

from core.logging.db import get_db
from core.logging.system_logger import SystemLogger

SOURCE = "dukascopy"
_BASE = "https://datafeed.dukascopy.com/datafeed"
_REC = struct.Struct(">IIIff")   # ms, ask, bid, askVol, bidVol (big-endian)
_TF_RULE = {"3m": "3min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}
_TF_MIN = {"3m": 3, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
_HEADERS = {"User-Agent": "Mozilla/5.0 (xau-trading-bot dukascopy fetch)"}


def _hour_url(inst: str, dt: datetime) -> str:
    return f"{_BASE}/{inst}/{dt.year:04d}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"


def _fetch_hour(inst: str, dt: datetime, cache_dir: Path | None) -> bytes | None:
    """Return the raw .bi5 bytes for one hour (cached), or None if not published."""
    cache = None
    if cache_dir is not None:
        cache = cache_dir / inst / f"{dt.year:04d}" / f"{dt.month - 1:02d}" / f"{dt.day:02d}" / f"{dt.hour:02d}.bi5"
        if cache.exists():
            return cache.read_bytes() or None
    url = _hour_url(inst, dt)
    for attempt in (1, 2, 3):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=30)
        except Exception:
            time.sleep(2 * attempt)
            continue
        if r.status_code == 404:
            return None                       # market closed — not an error
        if r.status_code == 200:
            data = r.content
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
            return data or None
        time.sleep(2 * attempt)
    print(f"  ⚠️ giving up on {url}")
    return None


def _reachable(inst: str) -> bool:
    """Preflight: can this host reach Dukascopy at all? A KNOWN-good past weekday
    hour must return SOME http response (200 with data, or 404). A connection
    error means the datafeed is blocked (datacenter-IP block) — fail fast with a
    clear message instead of grinding 'giving up' hour-by-hour for years."""
    probe = datetime(2024, 6, 5, 13, tzinfo=timezone.utc)   # Wed, liquid gold hour
    try:
        r = requests.get(_hour_url(inst, probe), headers=_HEADERS, timeout=30)
        return r.status_code in (200, 404)
    except Exception:
        return False


def _parse_hour(raw: bytes, hour_start: datetime, divisor: float) -> list[tuple]:
    """Decompress one .bi5 and yield (ts, mid, spread, volume) per tick."""
    try:
        buf = lzma.decompress(raw)
    except lzma.LZMAError:
        return []
    out = []
    for ms, ask_i, bid_i, av, bv in _REC.iter_unpack(buf):
        ask = ask_i / divisor
        bid = bid_i / divisor
        ts = hour_start + timedelta(milliseconds=ms)
        out.append((ts, (ask + bid) / 2.0, ask - bid, float(av + bv)))
    return out


def _resample_day(ticks: list[tuple], timeframes: list[str]) -> dict[str, pd.DataFrame]:
    """One UTC day of ticks → {tf: OHLCV+spread DataFrame}. Day boundaries are also
    5m/15m/1h/4h bar boundaries, so per-day resampling never splits a bar."""
    if not ticks:
        return {}
    idx = pd.DatetimeIndex([t[0] for t in ticks], tz="UTC")
    mid = pd.Series([t[1] for t in ticks], index=idx)
    spr = pd.Series([t[2] for t in ticks], index=idx)
    vol = pd.Series([t[3] for t in ticks], index=idx)
    frames = {}
    for tf in timeframes:
        rule = _TF_RULE[tf]
        o = mid.resample(rule).ohlc()
        o["volume"] = vol.resample(rule).sum()
        o["spread"] = spr.resample(rule).mean()
        o = o.dropna(subset=["open"])          # empty buckets = no ticks = closed
        if not o.empty:
            frames[tf] = o
    return frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD", help="Dukascopy instrument (e.g. XAUUSD).")
    p.add_argument("--store-symbol", default=None, help="Symbol to store under (default = --symbol).")
    p.add_argument("--start", required=True, help="UTC start date YYYY-MM-DD (inclusive).")
    p.add_argument("--end", required=True, help="UTC end date YYYY-MM-DD (exclusive).")
    p.add_argument("--timeframes", default="5m,15m,1h,4h")
    p.add_argument("--db-path", default="data/database/trading_bot.sqlite")
    p.add_argument("--cache-dir", default=None, help="Cache .bi5 files here (re-runs skip download).")
    p.add_argument("--point-divisor", type=float, default=1000.0,
                   help="Integer→price divisor (XAUUSD=1000; sanity-check the first bar).")
    p.add_argument("--spread-out", default=None,
                   help="Write an hour-of-day mean/median spread CSV here (empirical cost model).")
    p.add_argument("--no-preflight", action="store_true",
                   help="Skip the reachability probe (force the run even if the probe fails).")
    a = p.parse_args()

    store_symbol = a.store_symbol or a.symbol
    timeframes = [t.strip() for t in a.timeframes.split(",") if t.strip() in _TF_RULE]
    if not timeframes:
        sys.exit(f"no valid timeframes in {a.timeframes!r} (allowed: {list(_TF_RULE)})")
    start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
    cache_dir = Path(a.cache_dir) if a.cache_dir else None

    if not a.no_preflight and not _reachable(a.symbol):
        sys.exit(
            "🔴 Dukascopy datafeed is UNREACHABLE from this host — a probe of a known-good\n"
            "   past weekday hour got no HTTP response. This is almost always a datacenter-IP\n"
            "   block (RunPod/AWS/GCP are commonly blocked). Options:\n"
            "     • Fetch on a RESIDENTIAL connection (e.g. your own laptop) then copy the\n"
            "       SQLite DB to the pod, or\n"
            "     • Use Twelve Data instead: GOLD_DATA=twelvedata in run_runpod_backtest.sh,\n"
            "       or scripts/fetch_twelvedata_history.py (needs TWELVE_DATA_API_KEY).\n"
            "   (Re-run with --no-preflight to force past this check.)")

    db = get_db(a.db_path)
    logger = SystemLogger(db)
    print(f"=== Dukascopy {a.symbol} → '{store_symbol}' | {a.start} → {a.end} | "
          f"TFs {timeframes} | divisor {a.point_divisor:g} ===", flush=True)

    totals = {tf: 0 for tf in timeframes}
    spread_rows: list[tuple[int, float, float]] = []   # (hour_of_day, spread, weight)
    first_bar_shown = False

    cur_day = start
    while cur_day < end:
        day_ticks: list[tuple] = []
        hours_with_data = 0
        for h in range(24):
            hour_start = cur_day.replace(hour=h)
            if hour_start >= end:
                break
            raw = _fetch_hour(a.symbol, hour_start, cache_dir)
            if raw:
                t = _parse_hour(raw, hour_start, a.point_divisor)
                if t:
                    day_ticks.extend(t)
                    hours_with_data += 1

        frames = _resample_day(day_ticks, timeframes)
        for tf, df in frames.items():
            # Drop a bar whose close spills past `end` (defensive; day-flush already
            # keeps bars whole). close = open + tf duration.
            keep = df.index + pd.Timedelta(minutes=_TF_MIN[tf]) <= end
            df = df[keep]
            if df.empty:
                continue
            if not first_bar_shown:
                r0 = df.iloc[0]
                print(f"  [sanity] first {tf} bar {df.index[0]:%Y-%m-%d %H:%M}  "
                      f"O{r0['open']:.2f} H{r0['high']:.2f} L{r0['low']:.2f} C{r0['close']:.2f}  "
                      f"(gold should read ~1000s; if not, fix --point-divisor)", flush=True)
                first_bar_shown = True
            totals[tf] += logger.store_candles(df, store_symbol, tf, SOURCE)
            if tf == "5m":                       # densest TF drives the spread model
                for ts, row in df.iterrows():
                    spread_rows.append((ts.hour, float(row["spread"]), float(row["volume"])))

        if hours_with_data:
            print(f"  {cur_day:%Y-%m-%d}: {hours_with_data}h data, {len(day_ticks)} ticks | "
                  f"stored so far " + " ".join(f"{tf}:{totals[tf]}" for tf in timeframes), flush=True)
        cur_day += timedelta(days=1)

    print("\n=== DONE ===")
    for tf in timeframes:
        print(f"  {store_symbol} {tf}: {totals[tf]} candles stored")

    if a.spread_out and spread_rows:
        sdf = pd.DataFrame(spread_rows, columns=["hour", "spread", "weight"])
        agg = sdf.groupby("hour").agg(
            mean_spread=("spread", "mean"),
            median_spread=("spread", "median"),
            n=("spread", "size"),
        ).reset_index()
        out = Path(a.spread_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(out, index=False)
        overall = float(sdf["spread"].median())
        print(f"\n  spread model → {out}  (overall median {overall:.3f}; "
              f"replaces the hardcoded 0.25 default in the cost model)")


if __name__ == "__main__":
    main()
