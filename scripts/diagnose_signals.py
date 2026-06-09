"""
Diagnostic — WHY are signals so sparse?

Runs the pipeline over a recent window and tallies:
  - bars where NO setup formed at all (no sweep+FVG+valid SL) vs setups formed
  - grade distribution of formed setups (A+/A/B/C/D)
  - which MANDATORY conditions fail most (the blockers)
  - which OPTIONAL conditions are most often missing

Tells us whether sparsity is (a) correct ultra-selectivity, (b) the setup rarely
forms, or (c) one condition over-filtering / a bug.

    python scripts/diagnose_signals.py --symbol XAUUSD --execution-tf 15m --max-bars 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from core.logging.db import get_db
from core.engine.rulebook_engine import RulebookEngine
from core.engine.signal_pipeline import SignalPipeline
from core.engine.pipeline_hooks import build_default_hooks

DEFAULT_CONFIG = {
    "rr_tiers": {"min_to_enter": 2.0, "required_for_grade_b": 1.5,
                 "required_for_grade_a": 2.0, "required_for_grade_a_plus": 2.5},
    "tp1_r": 2.0, "tp2_r": 3.5,
    "costs": {"default_spread": 0.25, "default_slippage": 0.10,
              "point_value_per_lot": 100.0, "commission_per_lot": 0.0},
    "risk_per_trade_percent": 0.5,
    "session": {"timezone": "Asia/Jerusalem"},
}

_TFS = ["4h", "1h", "15m", "5m", "1m"]


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--execution-tf", default="15m")
    p.add_argument("--max-bars", type=int, default=1000)
    p.add_argument("--window", type=int, default=300)
    p.add_argument("--db-path", default="data/database/trading_bot.sqlite")
    return p.parse_args()


def _load(db, symbol, tf):
    rows = db.fetchall(
        "SELECT timestamp,open,high,low,close,volume FROM candles "
        "WHERE symbol=? AND timeframe=? ORDER BY timestamp ASC", (symbol, tf))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


def main():
    a = _args()
    db = get_db(a.db_path)
    full = {tf: _load(db, a.symbol, tf) for tf in _TFS}
    exec_df = full.get(a.execution_tf)
    if exec_df is None or exec_df.empty:
        print(f"no {a.execution_tf} candles for {a.symbol}")
        return

    # Use the REAL YAML config (has session kill-zone definitions); fall back to minimal.
    try:
        from core.engine.pipeline_config import assemble_pipeline_config
        config = assemble_pipeline_config("config")
        print("[config] using real YAML config (config/)")
    except Exception as exc:
        config = DEFAULT_CONFIG
        print(f"[config] WARNING: fallback to minimal config ({exc})")

    pipe = SignalPipeline(RulebookEngine(config),
                          **build_default_hooks(config, 10000.0, a.execution_tf))

    idx = exec_df.index
    start = max(a.window, len(idx) - a.max_bars)
    positions = list(range(start, len(idx)))
    total = len(positions)

    no_setup = 0
    setups = 0
    grades = Counter()
    failed_mand = Counter()
    passed_mand_when_setup = Counter()
    failed_opt = Counter()

    print(f"=== diagnosing {a.symbol} {a.execution_tf}: {total} bars "
          f"({idx[start]:%Y-%m-%d} → {idx[-1]:%Y-%m-%d}) ===")
    t0 = time.time()
    for i, pos in enumerate(positions):
        cutoff = idx[pos]
        hist = {tf: df[df.index <= cutoff].tail(a.window) for tf, df in full.items() if not df.empty}
        sig = pipe.process_bar({"timestamp": cutoff.to_pydatetime(), "bar_index": pos,
                                "symbol": a.symbol}, hist)
        if sig is None:
            no_setup += 1
        else:
            setups += 1
            grades[sig.grade] += 1
            dec = sig.decision
            if dec and dec.grade:
                for k in dec.grade.failed_mandatory:
                    failed_mand[k] += 1
                if dec.mandatory_results:
                    for k, v in dec.mandatory_results.items():
                        if v:
                            passed_mand_when_setup[k] += 1
                for k in (dec.grade.failed_optional or []):
                    failed_opt[k] += 1
        if (i + 1) % 100 == 0 or (i + 1) == total:
            r = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{total} | {r:.1f} bar/s | setups {setups} | grades {dict(grades)}", end="\r")

    print(f"\n\n=== RESULTS ({time.time()-t0:.0f}s) ===")
    print(f"bars scanned:        {total}")
    print(f"  NO setup formed:   {no_setup}  ({100*no_setup/total:.1f}%)  ← no sweep+FVG+valid SL")
    print(f"  setup formed:      {setups}  ({100*setups/total:.1f}%)")
    print(f"\ngrade distribution (of formed setups):")
    for g in ["A+", "A", "B", "C", "D"]:
        print(f"  {g:>3}: {grades.get(g, 0)}")
    tradeable = grades.get("A+", 0) + grades.get("A", 0)
    print(f"  → tradeable (A/A+): {tradeable}")

    if setups:
        print(f"\nMOST COMMON MANDATORY FAILURES (of {setups} formed setups):")
        for k, c in failed_mand.most_common():
            print(f"  {k:<22}: failed {c:>4}  ({100*c/setups:.0f}%)")
        print(f"\noptional conditions most often MISSING:")
        for k, c in failed_opt.most_common(6):
            print(f"  {k:<22}: missing {c:>4}  ({100*c/setups:.0f}%)")

    print("\n=== READING IT ===")
    print("• If 'NO setup formed' is ~99%+ → the SETUP (sweep+FVG) is the bottleneck, not grading.")
    print("• If many setups form but grade C/D → grading is the bottleneck; see top failures above.")
    print("• A single condition failing ~100% of the time = the prime suspect to tune.")


if __name__ == "__main__":
    main()
