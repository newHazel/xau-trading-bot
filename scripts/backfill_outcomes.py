"""
Backfill / re-evaluate past alert outcomes (Step 3 — DRY-RUN report, no DB writes).

The live OutcomeTracker used to book a LOSS for unfilled entries (fixed in
core/alerts/outcome_tracker.py). This tool re-evaluates the PERSISTED past alerts
(`signals` table) against stored candles using the CORRECTED entry-fill state machine,
so you get a CLEAN forward record and can SEE how many recorded "losses" were actually
phantom (NULLIFIED / EXPIRED — entry never filled).

It DOES NOT rewrite anything: it reads `signals` + candles, prints a report, and writes
a per-signal CSV under data/processed/. Deciding whether to persist a corrected record
is a separate, explicit step.

Candle source per symbol (via the backtest loader): crypto = committed data/candles CSVs,
gold (XAUUSD) = the SQLite `candles` table. Alerts with no candle coverage are reported
as NO_DATA, not guessed.

    python scripts/backfill_outcomes.py                 # all symbols, 5m
    python scripts/backfill_outcomes.py --symbols ETHUSDT,XAUUSD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import pandas as pd

from core.logging.db import get_db
from core.alerts.outcome_tracker import OutcomeTracker, DEFAULT_ENTRY_EXPIRY_BARS
import backtest_sequence_parallel as bsp  # _load: CSV-prefer (crypto) then DB (gold)

DB_DEFAULT = str(_ROOT / "data" / "database" / "trading_bot.sqlite")
OUT_DEFAULT = str(_ROOT / "data" / "processed" / "backfill_outcomes.csv")


class _Sig:
    """Minimal signal object the OutcomeTracker.record() expects."""
    def __init__(self, direction, entry, sl, tp1, grade):
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.tp1 = tp1
        self.grade = grade


def _load_signals(db, symbols=None):
    q = ("SELECT setup_id, symbol, timestamp, direction, entry, stop_loss, tp1, tp2, grade "
         "FROM signals ORDER BY timestamp ASC")
    try:
        rows = db.fetchall(q)
    except Exception as exc:
        print(f"⚠️  could not read `signals` table ({type(exc).__name__}: {exc}). "
              f"No persisted alerts to backfill.")
        return []
    out = [dict(r) for r in rows]
    if symbols:
        want = {s.strip().upper() for s in symbols}
        out = [r for r in out if str(r["symbol"]).upper() in want]
    return out


def _classify_corrected(row, df, entry_expiry_bars, max_hold_bars):
    """Re-evaluate one alert with the CORRECTED state machine. Returns (outcome, r)."""
    sig_ts = pd.to_datetime(row["timestamp"], utc=True)
    fwd = df[df.index > sig_ts]
    if max_hold_bars > 0:
        fwd = fwd.head(entry_expiry_bars + max_hold_bars)
    if fwd.empty:
        return "NO_DATA", None
    direction = str(row["direction"]).lower()
    t = OutcomeTracker(entry_expiry_bars=entry_expiry_bars)
    t.record(_Sig(direction, float(row["entry"]), float(row["stop_loss"]),
                  float(row["tp1"]), row["grade"]), sig_ts, last_bar_ts=sig_ts)
    t.update(fwd, now=None)
    if t.wins:
        return "WIN", t.total_r
    if t.losses:
        return "LOSS", t.total_r
    if t.nullified:
        return "NULLIFIED", 0.0
    if t.expired:
        return "EXPIRED", 0.0
    return "OPEN", None


def _classify_old_buggy(row, df, max_hold_bars):
    """Reproduce the OLD buggy logic: resolve on the first forward bar that touches SL or
    TP, with NO entry-fill check (SL-first). Quantifies how many losses were phantom."""
    sig_ts = pd.to_datetime(row["timestamp"], utc=True)
    fwd = df[df.index > sig_ts]
    if max_hold_bars > 0:
        fwd = fwd.head(max_hold_bars + 50)
    direction = str(row["direction"]).lower()
    sl = float(row["stop_loss"]); tp = float(row["tp1"])
    for hi, lo in zip(fwd["high"].to_numpy(), fwd["low"].to_numpy()):
        if direction == "long":
            if lo <= sl:
                return "LOSS"
            if hi >= tp:
                return "WIN"
        else:
            if hi >= sl:
                return "LOSS"
            if lo <= tp:
                return "WIN"
    return "OPEN"


def _pf(rs):
    gw = sum(r for r in rs if r > 0)
    gl = -sum(r for r in rs if r < 0)
    if gl <= 0:
        return float("inf") if gw > 0 else 0.0
    return gw / gl


def main() -> None:
    p = argparse.ArgumentParser(description="Re-evaluate past alerts with the corrected tracker (dry-run).")
    p.add_argument("--db-path", default=DB_DEFAULT)
    p.add_argument("--execution-tf", default="5m")
    p.add_argument("--symbols", default="", help="Comma-separated filter; default = all in `signals`.")
    p.add_argument("--entry-expiry-bars", type=int, default=DEFAULT_ENTRY_EXPIRY_BARS)
    p.add_argument("--max-hold-bars", type=int, default=500, help="0 = scan all forward bars.")
    p.add_argument("--out", default=OUT_DEFAULT)
    a = p.parse_args()

    db = get_db(a.db_path)
    symbols = [s for s in a.symbols.split(",") if s.strip()] or None
    signals = _load_signals(db, symbols)
    if not signals:
        print(f"No alerts found in `signals` (db={a.db_path}). Nothing to backfill — the "
              f"forward record will simply accumulate cleanly from now on with the fixed tracker.")
        return

    tss = [pd.to_datetime(r["timestamp"], utc=True) for r in signals]
    print(f"=== backfill (DRY-RUN) | {len(signals)} alerts | "
          f"{min(tss):%Y-%m-%d}→{max(tss):%Y-%m-%d} | exec_tf={a.execution_tf} ===")

    # cache candle frames per symbol
    cache: dict = {}
    rows_out = []
    for r in signals:
        sym = str(r["symbol"])
        if sym not in cache:
            try:
                cache[sym] = bsp._load(db, sym, a.execution_tf)
            except Exception:
                cache[sym] = pd.DataFrame()
        df = cache[sym]
        if df is None or df.empty:
            outcome, rr = "NO_DATA", None
            old = "NO_DATA"
        else:
            outcome, rr = _classify_corrected(r, df, a.entry_expiry_bars, a.max_hold_bars)
            old = _classify_old_buggy(r, df, a.max_hold_bars)
        rows_out.append({"setup_id": r["setup_id"], "symbol": sym, "ts": r["timestamp"],
                         "direction": r["direction"], "entry": r["entry"], "sl": r["stop_loss"],
                         "tp1": r["tp1"], "grade": r["grade"],
                         "corrected": outcome, "r": rr, "old_buggy": old})

    # ---- report ----
    def _summary(rows, title):
        from collections import Counter
        c = Counter(x["corrected"] for x in rows)
        real_r = [x["r"] for x in rows if x["corrected"] in ("WIN", "LOSS") and x["r"] is not None]
        n = c["WIN"] + c["LOSS"]
        wr = (100.0 * c["WIN"] / n) if n else 0.0
        old_loss = sum(1 for x in rows if x["old_buggy"] == "LOSS")
        print(f"\n  {title} ({len(rows)} alerts)")
        print(f"    CORRECTED: {c['WIN']}W / {c['LOSS']}L / {c['NULLIFIED']} nullified / "
              f"{c['EXPIRED']} expired / {c['OPEN']} open / {c['NO_DATA']} no-data")
        if n:
            print(f"    real-trade stats: {wr:.0f}% win | {sum(real_r):+.1f}R | PF {_pf(real_r):.2f}")
        phantom = old_loss - c["LOSS"]
        print(f"    OLD(buggy) would log {old_loss} losses → ~{max(phantom,0)} were PHANTOM "
              f"(unfilled entries the old tracker wrongly counted as losses)")

    _summary(rows_out, "ALL SYMBOLS")
    by_sym = {}
    for x in rows_out:
        by_sym.setdefault(x["symbol"], []).append(x)
    for sym in sorted(by_sym):
        _summary(by_sym[sym], sym)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    print(f"\n  per-alert detail → {out_path}")
    print("  (DRY-RUN: nothing written to the DB. Decide separately whether to persist a "
          "corrected record.)")


if __name__ == "__main__":
    main()
