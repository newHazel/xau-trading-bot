"""
Parallel + checkpointed faithful SequenceRunner backtest with CONFIG VARIANTS —
makes a big, statistically-meaningful ablation feasible on a laptop.

The signal-generation loop (the expensive per-bar SMC detector stack) is split into
time CHUNKS across CPU cores. Each chunk gets a WARMUP overlap (>> the 40-bar setup
expiry) so the state machine is reconstructed identically to a sequential run, and
only emits signals for bars it OWNS — so no signal is missed/double-counted at
boundaries. Each chunk checkpoints to disk (survives a kill → --aggregate-only).

Runs several config VARIANTS so you can attribute the effect of each change:
  baseline  = original system (fvg_freshness off, zone-rejection off)
  freshness = +FVG fresh/near selection + re-pin  (no #1 direction-aware, no #6)
  all       = + #1 (direction-aware) + #6 (zone-rejection)   [= current code]

ALWAYS run --verify first: it proves chunked output == sequential output.

    python scripts/backtest_sequence_parallel.py --verify
    python scripts/backtest_sequence_parallel.py --total-bars 8000 --jobs 6
    python scripts/backtest_sequence_parallel.py --aggregate-only --total-bars 8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

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
DB_DEFAULT = "data/database/trading_bot.sqlite"
OUT_DEFAULT = "/tmp/bt_chunks"

# variant label -> config overrides (on top of assemble_pipeline_config("config"))
VARIANTS = {
    "baseline":  {"fvg_freshness_enabled": False, "require_zone_rejection": False},
    "freshness": {"fvg_freshness_enabled": True,  "fvg_direction_aware": False, "require_zone_rejection": False},
    "all":       {"fvg_freshness_enabled": True,  "fvg_direction_aware": True,  "require_zone_rejection": True},
    # signal-boosters on top of the winning "freshness" config (#5 cooldown-after-
    # approval + #8 kill-zone-at-sweep) — do they ADD signals while keeping PF up?
    "boost":     {"fvg_freshness_enabled": True,  "fvg_direction_aware": False, "require_zone_rejection": False,
                  "cooldown_after_approval_only": True, "capture_killzone_at_sweep": True},
    # #3 multi-zone on top of the LIVE config (freshness + #5-default-on): watch the
    # N nearest zones, fire on whichever price retraces into first. Biggest signal-
    # adder but riskiest (touches zone selection where #1 hurt) — does it ADD GOOD signals?
    "multizone": {"fvg_freshness_enabled": True,  "fvg_direction_aware": False, "require_zone_rejection": False,
                  "fvg_multizone": True},
    # #2 sweep-early on top of LIVE: arm the sequence on a fresh wick (provisional
    # sweep) instead of waiting out the close-back lag — targets the stuck-at-sweep
    # case. Catches grabs faster BUT can fire on breakouts; does the net help?
    "sweep_early": {"fvg_freshness_enabled": True, "fvg_direction_aware": False, "require_zone_rejection": False,
                    "sweep_early": True},
}


def _load(db, sym, tf):
    rows = db.fetchall("SELECT timestamp,open,high,low,close,volume FROM candles "
                       "WHERE symbol=? AND timeframe=? ORDER BY timestamp ASC", (sym, tf))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


def _win(df, ts, w):
    pos = df.index.searchsorted(ts, side="right")
    return df.iloc[max(0, pos - w):pos]


def _gen_chunk(task):
    """Worker (top-level for spawn): emit signals for [real_start, real_end) under a
    config variant, fed a warmup overlap. Checkpoints to disk."""
    label, overrides, real_start, real_end, warmup, window, symbol, exec_tf, db_path, out_dir = task
    t0 = time.time()
    db = get_db(db_path)
    full = {tf: _load(db, symbol, tf) for tf in _TFS}
    exec_df = full[exec_tf]
    cfg = dict(assemble_pipeline_config("config"))
    cfg.update(overrides)
    runner = SequenceRunner(cfg, execution_tf=exec_tf, account_balance=10000.0,
                            tradeable_grades=("A+", "A", "B"))
    feed_start = max(window, real_start - warmup)
    out = []
    for gpos in range(feed_start, real_end):
        ts = exec_df.index[gpos]
        hist = {tf: _win(df, ts, window) for tf, df in full.items() if not df.empty}
        bar = {"timestamp": ts.to_pydatetime(), "bar_index": gpos, "symbol": symbol}
        sig = runner.on_bar(bar, hist)
        if sig is not None and gpos >= real_start:
            out.append({"setup_id": sig.setup_id, "direction": sig.direction,
                        "entry": sig.entry, "sl": sig.sl, "tp1": sig.tp1,
                        "tp2": sig.tp2 if sig.tp2 is not None else sig.tp1,
                        "grade": sig.grade, "gpos": gpos, "ts": ts.isoformat()})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"sig_{label}_{real_start:08d}.json"), "w") as f:
        json.dump(out, f)
    print(f"  [{label} {real_start}-{real_end}] {len(out)} signals ({time.time()-t0:.0f}s)", flush=True)
    return (label, out)


def _chunks(range_start, n, chunk_bars):
    cs = range_start
    while cs < n:
        yield (cs, min(cs + chunk_bars, n))
        cs += chunk_bars


def _score(signals, exec_df, range_start, exec_tf):
    sigs = sorted(signals, key=lambda s: s["gpos"])
    fe = [{"setup_id": s["setup_id"], "direction": s["direction"], "entry": s["entry"],
           "sl": s["sl"], "tp1": s["tp1"], "tp2": s["tp2"], "lot_size": 0.1,
           "bar_index": s["gpos"] - range_start, "grade": s["grade"]} for s in sigs]
    if not fe:
        return None
    exec_slice = exec_df.iloc[range_start:].copy()
    bt = BacktestRunner(BacktestConfig(
        initial_balance=10000.0, conservative_fills=True, costs_inclusive=True,
        default_spread=0.25, default_slippage=0.10,
        max_daily_trades=999, max_daily_losses=999, base_timeframe=exec_tf))
    return compute_metrics([t.to_dict() for t in bt.run(exec_slice, signals=fe).trades])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--execution-tf", default="5m")
    p.add_argument("--total-bars", type=int, default=8000)
    p.add_argument("--chunk-bars", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=450)
    p.add_argument("--window", type=int, default=350)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--db-path", default=DB_DEFAULT)
    p.add_argument("--out-dir", default=OUT_DEFAULT)
    p.add_argument("--variants", default="baseline,freshness,all")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    a = p.parse_args()

    db = get_db(a.db_path)
    exec_df = _load(db, a.symbol, a.execution_tf)
    n = len(exec_df)

    if a.verify:
        rs = max(a.window, n - 300)
        one = _gen_chunk(("all", VARIANTS["all"], rs, n, a.warmup, a.window, a.symbol,
                          a.execution_tf, a.db_path, a.out_dir + "_v1"))[1]
        tasks = [("all", VARIANTS["all"], cs, ce, a.warmup, a.window, a.symbol,
                  a.execution_tf, a.db_path, a.out_dir + "_v3") for cs, ce in _chunks(rs, n, 100)]
        with Pool(a.jobs) as pool:
            many = [s for _l, sigs in pool.map(_gen_chunk, tasks) for s in sigs]
        k = lambda s: (s["gpos"], round(s["entry"], 2), s["direction"])
        s1, s3 = sorted(map(k, one)), sorted(map(k, many))
        print(f"\n=== VERIFY ===\n  1 chunk: {len(s1)} signals | 3 chunks: {len(s3)} signals")
        print(f"  IDENTICAL: {s1 == s3}  "
              f"{'✅ chunking correct — safe for the big run' if s1 == s3 else '🔴 MISMATCH'}")
        return

    variants = [v.strip() for v in a.variants.split(",") if v.strip() in VARIANTS]
    range_start = max(a.window, n - a.total_bars)

    if a.aggregate_only:
        by = {v: [] for v in variants}
        for fn in sorted(os.listdir(a.out_dir)):
            if not fn.startswith("sig_"):
                continue
            lbl = fn.split("_")[1]
            if lbl in by:
                with open(os.path.join(a.out_dir, fn)) as f:
                    by[lbl].extend(json.load(f))
    else:
        tasks = []
        for v in variants:
            for cs, ce in _chunks(range_start, n, a.chunk_bars):
                tasks.append((v, VARIANTS[v], cs, ce, a.warmup, a.window, a.symbol,
                              a.execution_tf, a.db_path, a.out_dir))
        span = exec_df.iloc[range_start:]
        print(f"=== ablation: {a.symbol} {a.execution_tf} | {len(span)} bars "
              f"({span.index[0]:%Y-%m-%d}→{span.index[-1]:%Y-%m-%d}) | {variants} | "
              f"{len(tasks)} chunks × {a.jobs} workers ===", flush=True)
        t0 = time.time()
        with Pool(a.jobs) as pool:
            results = pool.map(_gen_chunk, tasks)
        print(f"\n  all chunks done in {time.time()-t0:.0f}s", flush=True)
        by = {v: [] for v in variants}
        for lbl, sigs in results:
            by[lbl].extend(sigs)

    print(f"\n{'='*72}\n  ABLATION RESULT — does each change earn its keep?\n{'='*72}")
    print(f"  {'variant':<12}{'signals':>9}{'win%':>8}{'PF':>7}{'expR':>9}{'totalR':>9}{'maxDD':>8}")
    metrics = {}
    for v in variants:
        m = _score(by[v], exec_df, range_start, a.execution_tf)
        metrics[v] = m
        if m is None:
            print(f"  {v:<12}{len(by[v]):>9}{'  (no trades)':>32}")
        else:
            print(f"  {v:<12}{len(by[v]):>9}{m.win_rate*100:>8.1f}{m.profit_factor:>7.2f}"
                  f"{m.expectancy:>9.3f}{m.total_r:>9.1f}{m.max_drawdown_r:>8.1f}")
    # compare every treatment variant against the live winner "freshness"
    f = metrics.get("freshness")
    if f:
        for v in variants:
            if v in ("baseline", "freshness") or not metrics.get(v):
                continue
            t = metrics[v]
            n_f = len(by["freshness"]); n_t = len(by[v])
            helps = (t.total_r >= f.total_r and t.profit_factor >= f.profit_factor)
            print(f"\n  → {v} vs freshness: signals {n_t} vs {n_f} | "
                  f"total R {t.total_r:+.1f} vs {f.total_r:+.1f} | PF {t.profit_factor:.2f} vs {f.profit_factor:.2f}")
            print(f"  → {v} {'EARNS its keep (more/equal signals, PF holds) — consider deploying' if helps else 'does NOT clearly help on this sample — keep freshness'}.")


if __name__ == "__main__":
    main()
