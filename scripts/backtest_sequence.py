"""
Faithful PF backtest of the LIVE engine — uses SequenceRunner (the stateful state
machine the live bot runs), NOT the one-shot SignalPipeline that backtest_signals.py
uses. Collected signals are replayed through the conservative FillEngine (SL fills
before TP on the same candle) with spread+slippage, then scored.

Compares the legacy FVG selection (newest-first) vs the fresh+near + periodic re-pin
upgrade (config: fvg_freshness_enabled) SIDE BY SIDE, so you can see whether the
upgrade's extra signals are actually profitable — not just more frequent.

Multi-timeframe, exactly like live: 4h/1h drive bias, 15m alignment, the --execution-tf
bar (default 5m) is the trigger. Daily limits are lifted so every signal's raw quality
is measured. Prints a per-signal log (sweep, FVG zone, entry/SL/TP, win/loss in R) so
you can see WHAT it marked and traded.

    # local sanity (fast, ~5 trading days):
    python scripts/backtest_sequence.py --max-bars 1500 --mode compare
    # big run (server / Railway, statistically meaningful):
    python scripts/backtest_sequence.py --max-bars 30000 --mode compare
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from core.logging.db import get_db
from core.engine.sequence_runner import SequenceRunner
from core.engine.pipeline_config import assemble_pipeline_config
from backtesting.backtest_runner import BacktestRunner, BacktestConfig
from backtesting.metrics import compute_metrics

_TFS = ["4h", "1h", "15m", "5m", "1m"]


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--execution-tf", default="5m")
    p.add_argument("--max-bars", type=int, default=1500)
    p.add_argument("--window", type=int, default=350)
    p.add_argument("--mode", default="compare", choices=["compare", "legacy", "new"])
    p.add_argument("--db-path", default="data/database/trading_bot.sqlite")
    p.add_argument("--max-signals-log", type=int, default=30)
    return p.parse_args()


def _load(db, sym, tf):
    rows = db.fetchall("SELECT timestamp,open,high,low,close,volume FROM candles "
                       "WHERE symbol=? AND timeframe=? ORDER BY timestamp ASC", (sym, tf))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


def _win(df, ts, w):
    """Fast rolling window: last `w` bars with index <= ts (searchsorted, O(log n))."""
    pos = df.index.searchsorted(ts, side="right")
    return df.iloc[max(0, pos - w):pos]


def _zone(f):
    if not f:
        return None
    return (round(min(f["top"], f["bottom"]), 2), round(max(f["top"], f["bottom"]), 2))


def run_mode(enabled, full, exec_df, a):
    cfg = dict(assemble_pipeline_config("config"))
    cfg["fvg_freshness_enabled"] = enabled
    runner = SequenceRunner(cfg, execution_tf=a.execution_tf, account_balance=10000.0,
                            tradeable_grades=("A+", "A", "B"))
    start = max(a.window, len(exec_df) - a.max_bars)
    exec_slice = exec_df.iloc[start:].copy()
    signals, logs, repins, prev_zone = [], [], 0, None
    t0 = time.time()

    for local_pos, gpos in enumerate(range(start, len(exec_df))):
        ts = exec_df.index[gpos]
        hist = {tf: _win(df, ts, a.window) for tf, df in full.items() if not df.empty}
        bar = {"timestamp": ts.to_pydatetime(), "bar_index": local_pos, "symbol": a.symbol}
        sig = runner.on_bar(bar, hist)

        # mark re-pins: captured FVG changed while still waiting for retrace
        cz = _zone(runner._captured.get("fvg"))
        if (cz is not None and prev_zone is not None and cz != prev_zone
                and runner.state.name == "WAITING_FOR_RETRACE_TO_ZONE"):
            repins += 1
        prev_zone = cz

        if sig is not None:
            sweep = runner._captured.get("sweep") or {}
            signals.append({
                "setup_id": sig.setup_id, "direction": sig.direction,
                "entry": sig.entry, "sl": sig.sl, "tp1": sig.tp1,
                "tp2": sig.tp2 if sig.tp2 is not None else sig.tp1,
                "lot_size": 0.1, "bar_index": local_pos, "grade": sig.grade,
            })
            logs.append({
                "ts": ts, "setup_id": sig.setup_id, "grade": sig.grade, "dir": sig.direction,
                "entry": sig.entry, "sl": sig.sl, "tp1": sig.tp1,
                "fvg": _zone(runner._captured.get("fvg")),
                "sweep": sweep.get("level") if isinstance(sweep, dict) else None,
            })
            prev_zone = None

    trades, m = [], None
    if signals:
        bt = BacktestRunner(BacktestConfig(
            initial_balance=10000.0, conservative_fills=True, costs_inclusive=True,
            default_spread=0.25, default_slippage=0.10,
            max_daily_trades=999, max_daily_losses=999, base_timeframe=a.execution_tf,
        ))
        result = bt.run(exec_slice, signals=signals)
        trades = [t.to_dict() for t in result.trades]
        m = compute_metrics(trades)

    return {"signals": signals, "logs": logs, "repins": repins, "trades": trades,
            "m": m, "secs": time.time() - t0, "bars": len(exec_slice)}


def _print_mode(label, r, a):
    print(f"\n{'='*66}\n  {label}\n{'='*66}")
    print(f"  bars: {r['bars']} | runtime: {r['secs']:.0f}s | signals: {len(r['signals'])} "
          f"| re-pins: {r['repins']}")
    m = r["m"]
    if m is None:
        print("  no tradeable signals — nothing to score.")
        return
    print(f"  WIN RATE:      {m.win_rate*100:.1f}%   ({m.wins}W / {m.losses}L)")
    print(f"  avg R:         {m.avg_r:+.2f}   (win {m.avg_win_r:+.2f}R / loss {m.avg_loss_r:+.2f}R)")
    print(f"  PROFIT FACTOR: {m.profit_factor:.2f}")
    print(f"  expectancy:    {m.expectancy:+.3f} R/trade")
    print(f"  TOTAL R:       {m.total_r:+.1f}")
    print(f"  max drawdown:  {m.max_drawdown_r:.1f}R ({m.max_drawdown_pct:.1f}%)")

    rmap = {t.get("setup_id"): t.get("r_multiple") for t in r["trades"]}
    print(f"\n  per-signal log (first {a.max_signals_log}):")
    print(f"  {'time UTC':<17}{'grade':>5} {'dir':>6} {'entry':>9}{'SL':>9}{'TP1':>9}"
          f"  {'FVG zone':>18} {'R':>7}")
    for lg in r["logs"][:a.max_signals_log]:
        rr = rmap.get(lg["setup_id"])
        rs = f"{rr:+.2f}" if rr is not None else "  -"
        fz = f"[{lg['fvg'][0]}-{lg['fvg'][1]}]" if lg["fvg"] else "-"
        print(f"  {lg['ts']:%Y-%m-%d %H:%M}{lg['grade']:>5} {lg['dir']:>6} "
              f"{lg['entry']:>9.2f}{lg['sl']:>9.2f}{lg['tp1']:>9.2f}  {fz:>18} {rs:>7}")


def main():
    a = _args()
    db = get_db(a.db_path)
    full = {tf: _load(db, a.symbol, tf) for tf in _TFS}
    exec_df = full.get(a.execution_tf)
    if exec_df is None or exec_df.empty:
        print(f"no {a.execution_tf} candles for {a.symbol}")
        return
    span = exec_df.iloc[max(a.window, len(exec_df) - a.max_bars):]
    print(f"=== faithful SequenceRunner backtest: {a.symbol} {a.execution_tf} | "
          f"{len(span)} bars ({span.index[0]:%Y-%m-%d} → {span.index[-1]:%Y-%m-%d}) | mode={a.mode} ===")

    results = {}
    if a.mode in ("compare", "legacy"):
        results["legacy"] = run_mode(False, full, exec_df, a)
        _print_mode("LEGACY  (newest-first FVG)", results["legacy"], a)
    if a.mode in ("compare", "new"):
        results["new"] = run_mode(True, full, exec_df, a)
        _print_mode("NEW  (fresh+near + periodic re-pin)", results["new"], a)

    if a.mode == "compare" and results["legacy"]["m"] and results["new"]["m"]:
        L, N = results["legacy"]["m"], results["new"]["m"]
        print(f"\n{'='*66}\n  VERDICT — legacy vs new\n{'='*66}")
        print(f"  {'metric':<16}{'legacy':>12}{'new':>12}")
        print(f"  {'signals':<16}{len(results['legacy']['signals']):>12}{len(results['new']['signals']):>12}")
        print(f"  {'win rate %':<16}{L.win_rate*100:>12.1f}{N.win_rate*100:>12.1f}")
        print(f"  {'profit factor':<16}{L.profit_factor:>12.2f}{N.profit_factor:>12.2f}")
        print(f"  {'expectancy R':<16}{L.expectancy:>12.3f}{N.expectancy:>12.3f}")
        print(f"  {'total R':<16}{L.total_r:>12.1f}{N.total_r:>12.1f}")
        print(f"  {'max DD R':<16}{L.max_drawdown_r:>12.1f}{N.max_drawdown_r:>12.1f}")
        better = N.total_r > L.total_r and N.profit_factor >= L.profit_factor
        print(f"\n  → NEW {'IMPROVES' if better else 'does NOT clearly improve'} net R "
              f"({N.total_r:+.1f} vs {L.total_r:+.1f}). "
              f"{'Worth deploying after a larger run.' if better else 'Do NOT deploy on this sample.'}")
        print("  (small sample — confirm on the big server run before deploying.)")


if __name__ == "__main__":
    main()
