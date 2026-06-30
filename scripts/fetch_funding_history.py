"""
Fetch ORTHOGONAL crypto market data (funding rate + open interest) from Binance
Futures, aligned to the backtest candle window — for the funding-aware ablation.

WHY: funding/OI are NOT transforms of price (unlike RSI/MACD/etc.) — they carry
NEW information about perp positioning. Positive funding = longs pay shorts =
long-crowded (squeeze risk); negative = short-crowded. The hypothesis: an extreme
funding gate explains/avoids the counter-trend longs that bled in the backtest.

DATA AVAILABILITY (verified 2026-06-30):
  - Funding rate  : FULL history, every 8h  → /fapi/v1/fundingRate (paginated). ✅ backtestable.
  - Open interest : LAST ~30 DAYS ONLY      → /futures/data/openInterestHist (5m). ⚠️ partial.
  - Liquidations  : no free history          → not fetched here (Coinglass paid / live-collect).

Output (per coin, alongside the candles):
  data/funding/<SYM>/funding.csv   columns: timestamp,funding_rate,mark_price
  data/funding/<SYM>/oi.csv        columns: timestamp,open_interest,oi_value   (recent 30d)

Run (RunPod / any host that can reach fapi.binance.com — futures endpoints are NOT
mirrored on data-api.binance.vision, and fapi may be geo-blocked from US datacenters):
  python scripts/fetch_funding_history.py --symbol ETHUSDT --start 2026-03-01
  python scripts/fetch_funding_history.py --all   # the 9-coin fleet
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

_ROOT = Path(__file__).parent.parent
FAPI = "https://fapi.binance.com"
FLEET = ["ETHUSDT", "DOGEUSDT", "SOLUSDT", "LINKUSDT", "AVAXUSDT",
         "NEARUSDT", "SUIUSDT", "SANDUSDT", "ZECUSDT"]


def _ms(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def _get(path: str, params: dict, tries: int = 4) -> list:
    key = (os.getenv("BINANCE_API_KEY") or "").strip()
    headers = {"X-MBX-APIKEY": key} if key else {}
    for i in range(tries):
        try:
            r = requests.get(FAPI + path, params=params, headers=headers, timeout=25)
            if r.status_code == 200:
                return r.json()
            # 429/418 rate limit → back off
            if r.status_code in (418, 429):
                time.sleep(2 * (i + 1))
                continue
            print(f"  !! {path} HTTP {r.status_code}: {r.text[:160]}", flush=True)
            return []
        except Exception as exc:
            print(f"  !! {path} error {type(exc).__name__}: {exc} (retry {i+1})", flush=True)
            time.sleep(1.5 * (i + 1))
    return []


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> list:
    """Full funding history (every 8h). Paginates forward by fundingTime."""
    out, cur = [], start_ms
    while cur < end_ms:
        page = _get("/fapi/v1/fundingRate",
                    {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": 1000})
        if not page:
            break
        out.extend(page)
        last = page[-1]["fundingTime"]
        if len(page) < 1000 or last <= cur:
            break
        cur = last + 1
        time.sleep(0.25)
    # dedup + sort
    seen, rows = set(), []
    for e in sorted(out, key=lambda x: x["fundingTime"]):
        ft = e["fundingTime"]
        if ft in seen:
            continue
        seen.add(ft)
        rows.append((ft, e["fundingRate"], e.get("markPrice", "")))
    return rows


def fetch_oi(symbol: str, period: str = "5m") -> list:
    """Open-interest history — Binance retains only ~30 days for sub-daily periods."""
    out, cur_end = [], None
    for _ in range(40):  # 40 * 500 * 5m ≈ way past the 30d cap; loop breaks on empty
        params = {"symbol": symbol, "period": period, "limit": 500}
        if cur_end:
            params["endTime"] = cur_end
        page = _get("/futures/data/openInterestHist", params)
        if not page:
            break
        out = page + out
        earliest = page[0]["timestamp"]
        if len(page) < 500:
            break
        cur_end = earliest - 1
        time.sleep(0.25)
    seen, rows = set(), []
    for e in sorted(out, key=lambda x: x["timestamp"]):
        ts = e["timestamp"]
        if ts in seen:
            continue
        seen.add(ts)
        rows.append((ts, e["sumOpenInterest"], e["sumOpenInterestValue"]))
    return rows


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _write(path: Path, header: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_iso(r[0])] + list(r[1:]))


def run_symbol(symbol: str, start: str, end: str | None) -> None:
    start_ms = _ms(start)
    end_ms = _ms(end) if end else int(time.time() * 1000)
    out_dir = _ROOT / "data" / "funding" / symbol
    print(f">>> {symbol}: funding {start}..{end or 'now'}", flush=True)

    fund = fetch_funding(symbol, start_ms, end_ms)
    _write(out_dir / "funding.csv", ["timestamp", "funding_rate", "mark_price"], fund)
    span = f"{_iso(fund[0][0])[:10]}..{_iso(fund[-1][0])[:10]}" if fund else "EMPTY"
    print(f"    funding: {len(fund)} rows ({span})", flush=True)

    oi = fetch_oi(symbol, "5m")
    _write(out_dir / "oi.csv", ["timestamp", "open_interest", "oi_value"], oi)
    ospan = f"{_iso(oi[0][0])[:10]}..{_iso(oi[-1][0])[:10]}" if oi else "EMPTY"
    print(f"    OI(5m): {len(oi)} rows ({ospan})  [Binance retains ~30d only]", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--all", action="store_true", help="fetch the 9-coin fleet")
    p.add_argument("--start", default="2026-03-01")
    p.add_argument("--end", default=None)
    a = p.parse_args()
    syms = FLEET if a.all else [a.symbol]
    for s in syms:
        run_symbol(s, a.start, a.end)
    print("DONE. -> data/funding/<SYM>/{funding.csv,oi.csv}", flush=True)


if __name__ == "__main__":
    main()
