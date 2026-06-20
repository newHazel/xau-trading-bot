"""
Live candle collector — keeps the per-coin CSVs current.

Every ROUND 5-minute mark (…:00, :05, :10 UTC, +small buffer) it fetches the latest
CLOSED candle(s) for each coin × timeframe (4h/1h/15m/5m) and APPENDS any new rows to
data/candles/<SYMBOL>/<tf>.csv (deduped by timestamp). The 5m file gains a row every
cycle; the higher TFs gain a row only when their candle closes (already-present rows
are skipped), so one simple loop keeps all four files current.

Run:
    python scripts/update_candles_csv.py            # loop, every round 5 min
    python scripts/update_candles_csv.py --once     # single top-up (test)

Where to run so the data PERSISTS:
  • locally — the CSVs grow; `git commit` them when you like; or
  • on the server with a VOLUME mounted at data/candles; or
  • add --git-push-minutes N to auto commit+push every N min (needs git creds on the
    box — keeps the repo's CSVs current by itself).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
import pandas as pd

from core.data.binance_fetcher import BinanceFetcher
from core.monitoring.cycle_timing import seconds_until_next_mark

DEFAULT_SYMBOLS = ["ETHUSDT", "DOGEUSDT", "SOLUSDT", "LINKUSDT",
                   "AVAXUSDT", "NEARUSDT", "SUIUSDT", "SANDUSDT",
                   "ZECUSDT"]
TFS = ["4h", "1h", "15m", "5m"]
COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _csv_path(out_root: str, sym: str, tf: str) -> Path:
    return Path(out_root) / sym / f"{tf}.csv"


def _last_ts(path: Path):
    """Last candle timestamp in the CSV, read from the tail only (cheap)."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 400))
        tail = f.read().decode(errors="ignore")
    lines = [ln for ln in tail.strip().splitlines() if ln]
    if not lines:
        return None
    first_field = lines[-1].split(",")[0]
    if first_field == "timestamp":  # header only → empty file
        return None
    try:
        return pd.to_datetime(first_field, utc=True)
    except Exception:
        return None


def _append(path: Path, df_new: pd.DataFrame) -> None:
    out = df_new.reset_index()
    out = out.rename(columns={out.columns[0]: "timestamp"})  # index → timestamp col
    # MUST match the existing files' format exactly (T-separated ISO, e.g.
    # 2026-03-01T00:00:00+00:00) — a space-separated row would make the CSV mixed-format
    # and break pd.to_datetime in _load.
    out["timestamp"] = out["timestamp"].map(lambda t: pd.Timestamp(t).isoformat())
    out = out[COLS]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, mode="a", header=not path.exists(), index=False)


def update_once(fetcher: BinanceFetcher, symbols, out_root: str) -> int:
    added = 0
    for sym in symbols:
        for tf in TFS:
            path = _csv_path(out_root, sym, tf)
            try:
                last = _last_ts(path)
                res = fetcher.fetch_latest_candles(sym, tf, 5)  # latest few CLOSED
                if getattr(res.status, "value", res.status) != "ok" or res.data is None or res.data.empty:
                    continue
                df = res.data
                if last is not None:
                    df = df[df.index > last]  # only candles newer than the last stored
                if not df.empty:
                    _append(path, df)
                    added += len(df)
            except Exception as exc:  # one coin/tf must never kill the loop
                print(f"  [{sym}/{tf}] error: {type(exc).__name__}: {exc}")
    return added


def _git_push(out_dir: str) -> None:
    try:
        subprocess.run(["git", "add", out_dir], cwd=str(_ROOT), check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(_ROOT)).returncode != 0:
            subprocess.run(["git", "commit", "-q", "-m", "chore: update candle CSVs"], cwd=str(_ROOT), check=True)
            subprocess.run(["git", "push"], cwd=str(_ROOT), check=True)
            print("  git: committed + pushed candle update")
    except Exception as exc:
        print(f"  git push failed: {exc}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append newly-closed candles to per-coin CSVs.")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--out-dir", default="data/candles")
    p.add_argument("--once", action="store_true", help="Single top-up and exit.")
    p.add_argument("--git-push-minutes", type=int, default=0,
                   help="If >0, commit+push the CSVs every N minutes (needs git creds).")
    return p.parse_args()


def main() -> None:
    load_dotenv(_ROOT / ".env")
    a = _parse_args()
    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    fetcher = BinanceFetcher()

    n = update_once(fetcher, symbols, a.out_dir)  # immediate top-up on start
    print(f"[{datetime.now(timezone.utc):%H:%M} UTC] startup top-up: +{n} candles "
          f"({len(symbols)} coins × {len(TFS)} TFs)")
    if a.once:
        return

    last_push = datetime.now(timezone.utc)
    print(f"=== candle collector: every round 5 min | {len(symbols)} coins | "
          f"auto-push={'every '+str(a.git_push_minutes)+'m' if a.git_push_minutes else 'off'} ===")
    try:
        while True:
            time.sleep(seconds_until_next_mark(300))  # fire just after each :00/:05/:10
            now = datetime.now(timezone.utc)
            added = update_once(fetcher, symbols, a.out_dir)
            print(f"[{now:%H:%M} UTC] +{added} candles", flush=True)
            if a.git_push_minutes and (now - last_push).total_seconds() >= a.git_push_minutes * 60:
                _git_push(a.out_dir)
                last_push = now
    except KeyboardInterrupt:
        print("\n(stopped)")


if __name__ == "__main__":
    main()
