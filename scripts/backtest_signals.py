"""
Scan + backtest in one pass — measures whether the tradeable signals are PROFITABLE.

Walks the execution-TF candles, collects every tradeable signal (A+/A/B, approved),
then replays them through the conservative FillEngine (SL fills before TP on the
same candle) with spread+slippage costs, and reports win rate / avg R / profit
factor / total R / max drawdown.

Daily limits are lifted here so EVERY signal's raw quality is measured (the live
system still applies them).

    python scripts/backtest_signals.py --symbol XAUUSDT --execution-tf 5m --max-bars 2500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from core.logging.db import get_db
from core.engine.rulebook_engine import RulebookEngine
from core.engine.signal_pipeline import SignalPipeline
from core.engine.pipeline_hooks import build_default_hooks
from backtesting.backtest_runner import BacktestRunner, BacktestConfig
from backtesting.metrics import compute_metrics

_TFS = ["4h", "1h", "15m", "5m", "1m"]
TRADEABLE = {"A+", "A", "B"}

DEFAULT_CONFIG = {
    "rr_tiers": {"min_to_enter": 2.0, "required_for_grade_b": 1.5,
                 "required_for_grade_a": 2.0, "required_for_grade_a_plus": 2.5},
    "tp1_r": 2.0, "tp2_r": 3.5,
    "costs": {"default_spread": 0.25, "default_slippage": 0.10,
              "point_value_per_lot": 100.0, "commission_per_lot": 0.0},
    "risk_per_trade_percent": 0.5,
    "session": {"timezone": "Asia/Jerusalem"},
}


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSDT")
    p.add_argument("--execution-tf", default="5m")
    p.add_argument("--max-bars", type=int, default=2500)
    p.add_argument("--window", type=int, default=300)
    p.add_argument("--db-path", default="data/database/trading_bot.sqlite")
    return p.parse_args()


def _load(db, symbol, tf):
    rows = db.fetchall("SELECT timestamp,open,high,low,close,volume FROM candles "
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

    try:
        from core.engine.pipeline_config import assemble_pipeline_config
        config = assemble_pipeline_config("config")
        print("[config] real YAML config")
    except Exception as exc:
        config = DEFAULT_CONFIG
        print(f"[config] fallback ({exc})")

    pipe = SignalPipeline(RulebookEngine(config),
                          **build_default_hooks(config, 10000.0, a.execution_tf))

    start = max(a.window, len(exec_df) - a.max_bars)
    exec_slice = exec_df.iloc[start:].copy()
    positions = list(range(start, len(exec_df)))

    print(f"=== scanning {a.symbol} {a.execution_tf}: {len(positions)} bars "
          f"({exec_df.index[start]:%Y-%m-%d} → {exec_df.index[-1]:%Y-%m-%d}) ===")
    signals = []
    t0 = time.time()
    for local_pos, gpos in enumerate(positions):
        cutoff = exec_df.index[gpos]
        hist = {tf: df[df.index <= cutoff].tail(a.window) for tf, df in full.items() if not df.empty}
        sig = pipe.process_bar({"timestamp": cutoff.to_pydatetime(), "bar_index": local_pos,
                                "symbol": a.symbol}, hist)
        if sig is not None and sig.approved and sig.grade in TRADEABLE:
            signals.append({
                "setup_id": sig.setup_id, "direction": sig.direction,
                "entry": sig.entry, "sl": sig.sl, "tp1": sig.tp1, "tp2": sig.tp2,
                "lot_size": 0.1, "bar_index": local_pos, "grade": sig.grade,
            })
        if (local_pos + 1) % 200 == 0:
            r = (local_pos + 1) / (time.time() - t0)
            print(f"  {local_pos+1}/{len(positions)} | {r:.1f} bar/s | signals {len(signals)}", end="\r")

    print(f"\n\ncollected {len(signals)} tradeable signals in {time.time()-t0:.0f}s")
    if not signals:
        print("no tradeable signals — nothing to backtest.")
        return

    # replay through the conservative fill engine (daily limits lifted to test all)
    bt = BacktestRunner(BacktestConfig(
        initial_balance=10000.0, conservative_fills=True, costs_inclusive=True,
        default_spread=0.25, default_slippage=0.10,
        max_daily_trades=999, max_daily_losses=999, base_timeframe=a.execution_tf,
    ))
    result = bt.run(exec_slice, signals=signals)
    m = compute_metrics([t.to_dict() for t in result.trades])

    print("\n=== BACKTEST RESULT (conservative fills + costs) ===")
    print(f"signals taken:    {m.total_trades}")
    print(f"wins / losses:    {m.wins} / {m.losses}")
    print(f"WIN RATE:         {m.win_rate*100:.1f}%")
    print(f"avg R:            {m.avg_r:+.2f}")
    print(f"  avg win:        {m.avg_win_r:+.2f}R   avg loss: {m.avg_loss_r:+.2f}R")
    print(f"PROFIT FACTOR:    {m.profit_factor:.2f}")
    print(f"expectancy:       {m.expectancy:+.3f} R/trade")
    print(f"TOTAL R:          {m.total_r:+.1f}")
    print(f"max drawdown:     {m.max_drawdown_r:.1f}R ({m.max_drawdown_pct:.1f}%)")
    print(f"best / worst:     {m.best_trade_r:+.1f}R / {m.worst_trade_r:+.1f}R")

    print("\n=== VERDICT ===")
    if m.profit_factor >= 1.5 and m.expectancy > 0:
        print("✅ PROFITABLE — positive expectancy + PF≥1.5. The relaxed version holds up.")
    elif m.expectancy > 0:
        print("🟡 MARGINAL — positive but thin. Worth refining (filters/RR) before trusting.")
    else:
        print("🔴 NEGATIVE expectancy — more signals but losing. Do NOT trade this as-is.")


if __name__ == "__main__":
    main()
