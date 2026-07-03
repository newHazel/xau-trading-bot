"""
Diagnostic — WHY are signals so sparse?  (SequenceRunner funnel)

Drives the REAL stateful engine — SequenceRunner, the exact path the backtest and
the live bot use — over a recent window and instruments the true funnel:

  • FUNNEL — how many setups PASS each mandatory gate, in order:
      HTF bias → 15m aligned → price in zone → liquidity sweep → valid FVG
      → retrace to zone → micro-CHoCH → confirmation → SIGNAL_READY
    The biggest drop between two consecutive gates is the tightest bottleneck.
  • SETUP DEATHS — why in-progress setups reset (expiry / htf bias lost-or-flipped /
    completed-but-not-tradeable / sizing failed / cooldown).
  • AT-ENTRY BLOCKERS — for setups that COMPLETED the full sequence, which at-entry
    gate rejected them (R:R < 2 net / off kill-zone / news window / blocking filter…).
  • grade distribution of emitted signals + state occupancy (where the engine dwells).

This REPLACES the old single-bar SignalPipeline.process_bar() diagnostic. That path
required all ~15 mandatory conditions TRUE on ONE bar (near-impossible — sweep + FVG +
retrace + CHoCH + confirmation never coincide on a single bar), so it could not describe
the STATEFUL engine that actually produces the signals. Its drop-reasons were for a
funnel neither the live bot nor the backtest uses.

    python scripts/diagnose_signals.py --symbol XAUUSD --execution-tf 15m --max-bars 1500
    python scripts/diagnose_signals.py --symbol XAUUSD --execution-tf 15m --db-path /workspace/xau_bt/trading_bot.sqlite
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
from core.engine.signal_pipeline import PipelineContext
from core.engine.sequence_runner import SequenceRunner, _SEQUENCE
from core.utils.visibility import visible_window
from core.engine.state_machine import State

# Fallback if the YAML config/ dir can't be loaded. The REAL run uses
# assemble_pipeline_config('config') — without it, SessionFilter has no kill-zone
# hours and no A/A+ signal can form.
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
    p = argparse.ArgumentParser(description="Diagnose the SequenceRunner funnel — where do gold setups die?")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--execution-tf", default="15m", choices=_TFS)
    p.add_argument("--max-bars", type=int, default=1500, help="Scan the last N execution bars.")
    p.add_argument("--window", type=int, default=350, help="Trailing history window fed to detectors per bar.")
    p.add_argument("--account-balance", type=float, default=10000.0)
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


# Human labels for the funnel gates, in sequence order (from _SEQUENCE).
_GATE_LABEL = {
    "htf_bias": "HTF bias (4h/1h)",
    "15m_aligned": "15m aligned",
    "price_zone": "price in zone (prem/disc)",
    "sweep": "liquidity sweep",
    "fvg": "valid FVG",
    "retrace": "retrace to zone",
    "micro_choch": "micro-CHoCH",
    "confirmation": "confirmation candle",
}


def main():
    a = _args()
    db = get_db(a.db_path)

    # Real YAML config (kill-zone hours, risk tiers, smc params) — same as scan/backtest.
    try:
        from core.engine.pipeline_config import assemble_pipeline_config
        config = assemble_pipeline_config("config")
        print("[config] loaded from config/ (real session/risk/smc params)")
    except Exception as exc:
        config = DEFAULT_CONFIG
        print(f"[config] WARNING: fallback to minimal config ({exc}) — signals will be sparse")

    full = {tf: _load(db, a.symbol, tf) for tf in _TFS}
    exec_df = full.get(a.execution_tf)
    if exec_df is None or exec_df.empty:
        print(f"no {a.execution_tf} candles for {a.symbol} in {a.db_path}. Fetch data first.")
        return

    def setup_id_fn(ctx: PipelineContext) -> str:
        ts = ctx.timestamp.strftime("%Y%m%d-%H%M") if hasattr(ctx.timestamp, "strftime") else "na"
        return f"{a.symbol}-{ts}-{ctx.direction.upper()}"

    # Construct EXACTLY as the backtest/scan do (defaults for pacing) so the funnel
    # describes the engine that produced the real signal count.
    runner = SequenceRunner(
        config, execution_tf=a.execution_tf,
        account_balance=a.account_balance, setup_id_fn=setup_id_fn,
    )

    idx = exec_df.index
    start = max(a.window, len(idx) - a.max_bars)
    positions = list(range(start, len(idx)))  # continuous — the state machine needs consecutive bars
    total = len(positions)
    if total <= 0:
        print("not enough bars after the warmup window")
        return

    occupancy = Counter()        # bars spent in each state (where the engine dwells)
    near_miss = Counter()        # at-entry gate that rejected a COMPLETED setup
    grades = Counter()
    signals = 0

    print(f"=== diagnosing {a.symbol} {a.execution_tf} (SequenceRunner): {total} bars "
          f"({idx[start]:%Y-%m-%d} → {idx[-1]:%Y-%m-%d}), window={a.window} ===")
    t0 = time.time()
    for i, pos in enumerate(positions):
        cutoff = idx[pos]
        # close-time visibility: exclude the still-forming HTF bar (look-ahead fix)
        hist = {tf: visible_window(df, cutoff, a.window, tf, a.execution_tf)
                for tf, df in full.items() if not df.empty}
        bar = {"timestamp": cutoff.to_pydatetime(), "bar_index": pos, "symbol": a.symbol}

        sig = runner.on_bar(bar, hist)

        occupancy[runner.state] += 1
        nm = runner.last_near_miss
        if nm:
            near_miss[nm["reason"]] += 1
        if sig is not None:
            signals += 1
            grades[sig.grade] += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            r = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{total} | {r:.1f} bar/s | signals {signals} | state {runner.state.value}", end="\r")

    # ---- analyse the full state-transition history (advances + forced resets) ----
    fwd = Counter()      # (from_state, to_state) -> count
    resets = Counter()   # forced-reset reason -> count
    for tr in runner._sm.history:
        fwd[(tr.from_state, tr.to_state)] += 1
        if tr.to_state == State.WAITING_FOR_HTF_BIAS and tr.reason.startswith("FORCED:"):
            resets[tr.reason[len("FORCED:"):].strip()] += 1

    print(f"\n\n{'='*72}")
    print(f"  FUNNEL — SequenceRunner (the real engine path) | {total} bars scanned")
    print(f"{'='*72}")
    print(f"  {'gate passed':<28}{'count':>8}{'retained':>12}")
    prev = None
    tightest = None
    for cur, cond, nxt in _SEQUENCE:
        cnt = fwd.get((cur, nxt), 0)
        if prev is None:
            ret_s = "   —"
        else:
            ret = (cnt / prev) if prev else 0.0
            ret_s = f"{100*ret:>6.0f}%"
            if tightest is None or ret < tightest[1]:
                tightest = (cond, ret, prev, cnt)
        print(f"  {_GATE_LABEL.get(cond, cond):<28}{cnt:>8}{ret_s:>12}")
        prev = cnt
    ready = fwd.get((State.WAITING_FOR_CONFIRMATION_CANDLE, State.SIGNAL_READY), 0)
    if tightest:
        print(f"\n  ⇒ TIGHTEST gate: '{_GATE_LABEL.get(tightest[0], tightest[0])}' — "
              f"retains {100*tightest[1]:.0f}% ({tightest[3]} of {tightest[2]} setups that reached it)")

    print(f"\n  SETUP DEATHS (why in-progress setups reset):")
    if resets:
        for reason, c in resets.most_common():
            print(f"    {reason:<52}{c:>6}")
    else:
        print("    (none)")

    print(f"\n  AT-ENTRY BLOCKERS (completed setups rejected at emit):")
    if near_miss:
        for reason, c in near_miss.most_common():
            print(f"    {reason:<40}{c:>6}")
    else:
        print("    (none reached SIGNAL_READY, or none rejected at entry)")

    print(f"\n  OUTCOME:")
    print(f"    completed sequences (SIGNAL_READY): {ready}")
    print(f"    emitted signals:                    {signals}   grades {dict(grades)}")

    print(f"\n  STATE OCCUPANCY (bars spent waiting in each state):")
    for st, c in occupancy.most_common():
        print(f"    {st.value:<34}{c:>6}  ({100*c/total:.0f}%)")

    print(f"\n=== READING IT ===")
    print("• The gate with the LOWEST 'retained' % is the tightest bottleneck to work on.")
    print("• Many 'setup expired' deaths → setups form but never complete in time (loosen a")
    print("  mid-sequence gate, or raise setup_expiry).")
    print("• AT-ENTRY dominated by 'R:R < 2 net' → completed setups die at the rr_minimum gate")
    print("  (lower rr_tiers.min_to_enter 2.0→1.5 in config/risk.yaml — trades reward for count).")
    print("• High occupancy in one WAITING_* state = the engine dwells there (that gate rarely fires).")


if __name__ == "__main__":
    main()
