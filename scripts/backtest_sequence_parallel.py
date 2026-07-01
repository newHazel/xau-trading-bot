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
import re
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
from backtesting.bootstrap import bootstrap_ci, bootstrap_diff, holm_threshold
from core.data.funding_provider import load_funding

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
    # --- "loosen the brakes" experiment (vs the live freshness baseline) ---
    # rr_15: accept R:R down to 1.5 net (live floor is 2.0) → the marginal-R:R setups.
    "rr_15":      {"fvg_freshness_enabled": True, "fvg_direction_aware": False, "require_zone_rejection": False,
                   "rr_tiers": {"min_to_enter": 1.5}},
    # no_killzone: ignore the kill-zone gate (trade ANY session) → the single biggest
    # filter removed. Shows how much frequency the kill-zone costs AND what it saves (PF).
    "no_killzone": {"fvg_freshness_enabled": True, "fvg_direction_aware": False, "require_zone_rejection": False,
                    "ignore_kill_zone": True},
    # momentum-confirmation gate (the 2026-06-15 losers entered at low RSI = falling knife).
    # mom_rsi45: long needs RSI>=45 / short<=55.  mom_rsi50: stricter (RSI on the right side of 50).
    "mom_rsi45": {"fvg_freshness_enabled": True, "fvg_direction_aware": False, "require_zone_rejection": False,
                  "momentum_gate": True, "rsi_long_min": 45.0, "rsi_short_max": 55.0},
    "mom_rsi50": {"fvg_freshness_enabled": True, "fvg_direction_aware": False, "require_zone_rejection": False,
                  "momentum_gate": True, "rsi_long_min": 50.0, "rsi_short_max": 50.0},
    # price-sanity gate: skip "dead-on-arrival" signals where current price already
    # broke past the SL (the FVG was blown through). Does removing them raise win%/PF?
    "sane":      {"fvg_freshness_enabled": True, "fvg_direction_aware": False, "require_zone_rejection": False,
                  "price_sanity_gate": True},
    # CRYPTO (e.g. ETHUSDT, Binance): the live LIVE gold config (freshness + price-sanity)
    # but 24/7 — ignore the gold kill-zone sessions since crypto trades around the clock.
    "crypto":    {"fvg_freshness_enabled": True, "fvg_direction_aware": False, "require_zone_rejection": False,
                  "price_sanity_gate": True, "ignore_kill_zone": True},
    # CRYPTO with PRICE-PROPORTIONAL costs (the fix): the absolute 0.25/0.10 cost model
    # is gold-calibrated and auto-rejects cheap coins (rr_minimum + spread filter both
    # fail → grade D). 'crypto_pct' scales spread/slippage with price (~0.04% total) so
    # every coin pays the same fraction. THIS is the variant to judge crypto edge on.
    "crypto_pct": {"fvg_freshness_enabled": True, "fvg_direction_aware": False, "require_zone_rejection": False,
                   "price_sanity_gate": True, "ignore_kill_zone": True,
                   "costs": {"cost_model": "percent", "spread_pct": 0.0002, "slippage_pct": 0.0002},
                   "spread": {"cost_model": "percent", "spread_pct": 0.0002}},
}

# --- CRYPTO ABLATION LEVERS (institutional study) ---------------------------------
# Each lever = the crypto_pct BASELINE + exactly ONE change, so the ablation attributes
# the effect cleanly (NOT built on the gold 'freshness' base, which would confound the
# kill-zone). Compare each to crypto_pct, validated OOS with bootstrap CIs.
_CRYPTO_PCT = VARIANTS["crypto_pct"]
VARIANTS.update({
    # momentum gate (WIRED): don't enter against momentum (falling-knife) — long needs
    # RSI>=45 / short<=55. Hypothesis: higher win% / fewer immediate-reversal entries.
    "crypto_mom":   {**_CRYPTO_PCT, "momentum_gate": True, "rsi_long_min": 45.0, "rsi_short_max": 55.0},
    # sweep-early: arm the sequence on the PROVISIONAL wick (with a breakout guard)
    # instead of waiting for the confirmed close-back — catch the move before it's too
    # late / already reversed (the user's core complaint). Hypothesis: more/earlier fills.
    "crypto_sweep": {**_CRYPTO_PCT, "sweep_early": True},
    # funding (ORTHOGONAL): block a fresh trade on the crowded perp side (long into
    # crowded-long funding / short into crowded-short). Needs data/funding/<SYM>/funding.csv
    # (scripts/fetch_funding_history.py). Hypothesis: avoids squeeze-prone counter-trend
    # longs — the bucket that bled. The first NON-price-derived signal in the ablation.
    "crypto_funding": {**_CRYPTO_PCT, "funding_filter": True},
    # trend gate: require the exec-TF EMA50/200 trend to NOT oppose the trade (neutral
    # allowed). Targets the proven counter-trend-LONG bleed in a down regime.
    "crypto_trend": {**_CRYPTO_PCT, "trend_gate": True},
    # wider SL band: floor the SL at 2x ATR (no noise-tight stops) and raise the max to 3x.
    # Targets the ETH 11:50 case — a tight SL wicked by a bounce, then price went the trade's
    # way. Trade-off: wider SL lowers R:R → fewer pass the rr gate. The ablation decides.
    "crypto_slfloor": {**_CRYPTO_PCT, "sl_atr_floor_mult": 2.0, "atr_sl_multiplier": 3.0},
    # REAL confirmation gate (the entry-quality root fix, "Layer 4"): replace the weak
    # green/red body-color confirmation with a genuine REJECTION candle — decisive body
    # (>= 0.3x ATR) that closed back THROUGH the proximal FVG edge (reclaimed the level).
    # Stops the bot firing into bounces/breakouts that never rejected (the ETH 11:50 loser).
    "crypto_confirm": {**_CRYPTO_PCT, "confirm_gate": True, "confirm_min_body_atr": 0.3},
})

# --- GOLD (XAUUSD) variants: the live gold baseline is 'freshness' (kill-zone sessions ON,
# absolute costs) — NOT crypto_pct. Same improvement flags, gold-flavoured. ---
_FRESHNESS = VARIANTS["freshness"]
VARIANTS.update({
    "gold_confirm": {**_FRESHNESS, "confirm_gate": True, "confirm_min_body_atr": 0.4},
    "gold_trend":   {**_FRESHNESS, "trend_gate": True},
})


def _load(db, sym, tf):
    # Prefer the committed per-coin CSV (data/candles/<SYM>/<tf>.csv) — it ships with
    # the repo, so the server has the OHLCV on deploy with NO download and it survives
    # restarts. Fall back to the SQLite DB if the CSV isn't present.
    csv = _ROOT / "data" / "candles" / sym / f"{tf}.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        if df.empty:
            return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp").sort_index()
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
    # Resume: if this chunk was already generated (e.g. on a persistent volume), reuse
    # it instead of regenerating — signal gen is the slow part, scoring is cheap.
    ckpt = os.path.join(out_dir, f"sig_{label}_{real_start:08d}.json")
    if os.path.exists(ckpt):
        try:
            with open(ckpt) as f:
                out = json.load(f)
            print(f"  [{label} {real_start}-{real_end}] {len(out)} signals (cached)", flush=True)
            return (label, out)
        except Exception:
            pass  # corrupt/partial checkpoint → regenerate
    db = get_db(db_path)
    full = {tf: _load(db, symbol, tf) for tf in _TFS}
    # Orthogonal funding series (per-coin) for the funding_filter variant. Added as a
    # pseudo-timeframe so the existing _win() windowing + hist plumbing carry it to the
    # hooks leak-free; absent -> simply not in history and the gate stays off.
    _fdf = load_funding(symbol)
    if _fdf is not None and not _fdf.empty:
        full["funding"] = _fdf
    exec_df = full[exec_tf]
    cfg = dict(assemble_pipeline_config("config"))
    for _k, _v in overrides.items():  # one-level deep-merge so e.g. rr_tiers isn't clobbered
        cfg[_k] = {**cfg[_k], **_v} if isinstance(_v, dict) and isinstance(cfg.get(_k), dict) else _v
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


def _run_trades(signals, exec_df, range_start, exec_tf, overrides=None):
    """Fill/backtest `signals` (rebased to range_start) → list of trade dicts.
    A signal whose limit entry is never bracketed within the trigger window is DROPPED
    (no trade), so len(trades) < len(signals) measures the fill rate (the 'entered too
    late / price ran away' rate the user cares about)."""
    sigs = sorted(signals, key=lambda s: s["gpos"])
    fe = [{"setup_id": s["setup_id"], "direction": s["direction"], "entry": s["entry"],
           "sl": s["sl"], "tp1": s["tp1"], "tp2": s["tp2"], "lot_size": 0.1,
           "bar_index": s["gpos"] - range_start, "grade": s["grade"]} for s in sigs]
    if not fe:
        return []
    exec_slice = exec_df.iloc[range_start:].copy()
    # The fill engine's costs MUST match the variant's cost model. BacktestRunner only
    # knows absolute spread/slippage, so for a "percent" variant we scale them to the
    # coin's price level — else a cheap coin gets a gold-sized 0.35 cost that dwarfs its
    # price-proportional risk, making win%=0 / expR=-800R (a PnL artefact, not a result).
    costs_cfg = (overrides or {}).get("costs", {})
    if str(costs_cfg.get("cost_model", "absolute")).lower() == "percent":
        ref_price = float(exec_slice["close"].median())
        spread = float(costs_cfg.get("spread_pct", 0.0)) * ref_price
        slippage = float(costs_cfg.get("slippage_pct", 0.0)) * ref_price
    else:
        spread, slippage = 0.25, 0.10
    bt = BacktestRunner(BacktestConfig(
        initial_balance=10000.0, conservative_fills=True, costs_inclusive=True,
        default_spread=spread, default_slippage=slippage,
        max_daily_trades=999, max_daily_losses=999, base_timeframe=exec_tf))
    return [t.to_dict() for t in bt.run(exec_slice, signals=fe).trades]


def _score(signals, exec_df, range_start, exec_tf, overrides=None):
    """Metrics for `signals` (None only when there are NO signals at all; if signals
    exist but none fill, returns a real MetricsResult with total_trades=0)."""
    if not signals:
        return None
    trades = _run_trades(signals, exec_df, range_start, exec_tf, overrides)
    total_bars = max(0, len(exec_df) - range_start)
    return compute_metrics(trades, total_bars=total_bars)


def _evaluate(signals, exec_df, range_start, exec_tf, overrides=None):
    """(metrics, r_values, n_signals) in one backtest pass — for the institutional
    report (metrics + bootstrap on the same trade list, plus the fill rate)."""
    n_sig = len(signals)
    if n_sig == 0:
        return None, [], 0
    trades = _run_trades(signals, exec_df, range_start, exec_tf, overrides)
    total_bars = max(0, len(exec_df) - range_start)
    metrics = compute_metrics(trades, total_bars=total_bars)
    r_values = [t.get("r_multiple", 0) for t in trades]
    return metrics, r_values, n_sig


def _pf(x):
    return "inf" if x == float("inf") else f"{x:.2f}"


def _reach_rate(signals, exec_df, range_start, expiry=12):
    """CLEAN 'entered vs ran away' rate: of N signals, the fraction whose limit price was
    bracketed by SOME bar within `expiry` bars of the signal — computed directly from the
    bars, independent of the backtest's one-position-at-a-time occupancy.

    This is the honest answer to the user's complaint ('entered too late / price already
    reversed'). It differs from the runner's fill rate (filled/signals), which ALSO drops
    signals that arrived while another trade was open (an occupancy skip, NOT a runaway).
    Returns (reach_fraction, reached, n)."""
    if not signals:
        return 0.0, 0, 0
    lo = exec_df["low"].to_numpy(dtype=float)
    hi = exec_df["high"].to_numpy(dtype=float)
    nbars = len(exec_df)
    reached = 0
    for s in signals:
        g = int(s["gpos"]); e = float(s["entry"])
        end = min(nbars, g + expiry + 1)
        if g < nbars and any(lo[b] <= e <= hi[b] for b in range(g, end)):
            reached += 1
    return reached / len(signals), reached, len(signals)


def _fmt_row(label, n_sig, m, min_trades, reach_pct):
    """One institutional metrics row. n_sig = signals GENERATED; m.total_trades = FILLED
    (occupancy-limited); reach_pct = clean entered-vs-ran-away rate."""
    if m is None or n_sig == 0:
        return f"  {label:<14}{n_sig:>8}{'  -- no signals':>30}"
    filled = m.total_trades
    flag = "  !INSUFF" if filled < min_trades else ""
    return (f"  {label:<14}{n_sig:>8}{filled:>8}{reach_pct*100:>6.0f}%{m.win_rate*100:>7.1f}"
            f"{_pf(m.profit_factor):>7}{m.expectancy:>8.3f}{m.total_r:>8.1f}{m.max_drawdown_r:>7.1f}"
            f"{_pf(m.sortino):>7}{_pf(m.payoff_ratio):>7}{m.longest_loss_streak:>6}{flag}")


def _report(by, exec_df, range_start, n, exec_tf, variants, span, months, args):
    sym = args.symbol
    print(f"\n{'='*104}\n  BACKTEST RESULT — {sym} exec_tf={exec_tf}  (corrected engine: price_zone gate + Wilder ATR + sizing)")
    print(f"  period {span.index[0]:%Y-%m-%d}->{span.index[-1]:%Y-%m-%d}  ({months:.1f} months, {len(span)} bars)")
    print(f"  reach% = of N signals, how many price actually RETURNED to (entered vs ran away) — the")
    print(f"           clean entry-timing measure. 'filled' = trades actually taken (occupancy-limited);")
    print(f"           R-stats below are computed on those. !INSUFF = < {args.min_trades} fills = insufficient evidence.")
    print(f"{'='*104}")
    print(f"  {'variant':<14}{'signals':>8}{'filled':>8}{'reach':>7}{'win%':>7}"
          f"{'PF':>7}{'expR':>8}{'totR':>8}{'maxDD':>7}{'Sortino':>7}{'payoff':>7}{'Lstrk':>6}")
    full = {}
    for v in variants:
        m, rv, nsig = _evaluate(by[v], exec_df, range_start, exec_tf, VARIANTS[v])
        full[v] = (m, rv, nsig)
        reach, _rc, _rn = _reach_rate(by[v], exec_df, range_start)
        print(_fmt_row(v, nsig, m, args.min_trades, reach))

    # per-variant detail: long/short + exit-type + optional bootstrap CIs
    for v in variants:
        m, rv, nsig = full[v]
        if m is None or m.total_trades == 0:
            continue
        bd = m.breakdowns.get("by_direction", {})
        ds = " | ".join(f"{d}:{s['count']}@{s['win_rate']*100:.0f}%/{s['total_r']:+.1f}R" for d, s in bd.items())
        ex = " | ".join(f"{et}:{s['count']}/{s['total_r']:+.1f}R" for et, s in m.exit_types.items())
        print(f"\n  [{v}] dir : {ds}")
        print(f"  [{v}] exit: {ex}")
        if args.bootstrap:
            ci = bootstrap_ci(rv)
            c = ci["ci"]
            print(f"  [{v}] 95%CI PF[{_pf(c['profit_factor'][0])}..{_pf(c['profit_factor'][1])}] "
                  f"expR[{c['expectancy'][0]:+.2f}..{c['expectancy'][1]:+.2f}] "
                  f"totR[{c['total_r'][0]:+.1f}..{c['total_r'][1]:+.1f}]  P(no edge)={ci['p_no_edge']*100:.0f}%")

    # OUT-OF-SAMPLE split (pre-committed chronological holdout)
    oos_data = {}
    if args.oos_ratio and args.oos_ratio > 0:
        cut = range_start + int((1 - args.oos_ratio) * (n - range_start))
        cut_ts = exec_df.index[cut]
        print(f"\n{'-'*104}\n  OUT-OF-SAMPLE  ({int((1-args.oos_ratio)*100)}/{int(args.oos_ratio*100)} chronological split, OOS starts {cut_ts:%Y-%m-%d %H:%M})")
        print(f"  {'variant':<14}{'IS_sig':>7}{'IS_fil':>7}{'IS_totR':>9}{'IS_PF':>7}{'   ':>3}"
              f"{'OOS_sig':>8}{'OOS_fil':>8}{'OOS_totR':>9}{'OOS_PF':>8}{'OOS_exp':>9}")
        for v in variants:
            is_sigs = [s for s in by[v] if s["gpos"] < cut]
            oos_sigs = [s for s in by[v] if s["gpos"] >= cut]
            mi, ri, ni = _evaluate(is_sigs, exec_df.iloc[:cut], range_start, exec_tf, VARIANTS[v])
            mo, ro, no = _evaluate(oos_sigs, exec_df, cut, exec_tf, VARIANTS[v])
            oos_data[v] = (mi, ri, ni, mo, ro, no)
            isf = mi.total_trades if mi else 0
            ist = f"{mi.total_r:+.1f}" if mi else "-"
            isp = _pf(mi.profit_factor) if mi else "-"
            if mo and no:
                flg = " !" if mo.total_trades < args.min_oos_trades else ""
                print(f"  {v:<14}{ni:>7}{isf:>7}{ist:>9}{isp:>7}{'':>3}"
                      f"{no:>8}{mo.total_trades:>8}{mo.total_r:>+9.1f}{_pf(mo.profit_factor):>8}{mo.expectancy:>+9.3f}{flg}")
            else:
                print(f"  {v:<14}{ni:>7}{isf:>7}{ist:>9}{isp:>7}{'':>3}{no:>8}{'   (no OOS signals)':>34}")

    # ABLATION: each lever vs baseline, multiple-testing aware (Holm). The scope (OOS vs
    # full-window) is decided PER LEVER and applied to BOTH arms identically — never diff a
    # lever's full-window R against the baseline's OOS R. A lever whose relevant arm (or the
    # baseline's) has too few trades is reported 'insufficient', never promoted.
    base = args.baseline
    if base in full and full[base][0] is not None:
        oos_on = bool(args.oos_ratio and args.oos_ratio > 0 and oos_data)

        def _arm(v, scope):
            """(r_values, n_trades) for variant v in the given scope ('oos' | 'full')."""
            if scope == "oos" and v in oos_data and oos_data[v][3] is not None:
                return oos_data[v][4], oos_data[v][3].total_trades
            return full[v][1], (full[v][0].total_trades if full[v][0] else 0)

        levers = [v for v in variants
                  if v != base and v not in ("baseline", "freshness", "all") and full[v][0] is not None]
        rows = []  # (v, scope, eff, ci, p, floor_ok)
        for v in levers:
            # prefer OOS only when BOTH arms have OOS data; else full-window for BOTH (consistent).
            scope = "oos" if (oos_on and v in oos_data and oos_data[v][3] is not None
                              and oos_data[base][3] is not None) else "full"
            floor = args.min_oos_trades if scope == "oos" else args.min_trades
            t_rv, t_n = _arm(v, scope)
            b_rv, b_n = _arm(base, scope)
            floor_ok = (t_n >= floor and b_n >= floor)
            d = bootstrap_diff(t_rv, b_rv, metric="expectancy")
            rows.append((v, scope, d, floor_ok))
        # Holm only over the levers that clear the sample floor (the ones we can actually judge).
        judged_idx = [i for i, r in enumerate(rows) if r[3]]
        surv_map = {}
        if judged_idx:
            ps = [rows[i][2].get("p_treatment_worse_or_equal", 1.0) for i in judged_idx]
            for i, s in zip(judged_idx, holm_threshold(ps, alpha=0.05)):
                surv_map[i] = s
        print(f"\n{'-'*104}\n  ABLATION vs baseline '{base}' — promote a lever ONLY if its expectancy gain is real")
        print(f"  (Holm-corrected across judged levers, CI excludes 0), >= +0.1R, AND both arms >= the")
        print(f"  sample floor (OOS>={args.min_oos_trades} / full>={args.min_trades} trades):")
        for i, (v, scope, d, floor_ok) in enumerate(rows):
            lo, hi = d.get("ci", (float("nan"), float("nan")))
            eff = d.get("point_diff", 0.0)
            if not floor_ok or d.get("insufficient", False):
                print(f"  {v:<14} [{scope:>4}] d.exp {eff:+.3f}R  -> insufficient evidence (too few trades)")
                continue
            surv = surv_map.get(i, False)
            ci_excl0 = (lo > 0) or (hi < 0)
            promote = bool(surv and ci_excl0 and eff >= 0.1)
            verdict = "PROMOTE" if promote else ("inconclusive" if eff > 0 else "no help")
            print(f"  {v:<14} [{scope:>4}] d.exp {eff:+.3f}R  CI[{lo:+.3f}..{hi:+.3f}]  Holm={'pass' if surv else 'fail'}  -> {verdict}")
        print(f"  (small sample: most levers should read 'inconclusive'/'insufficient' — the honest outcome.)")

    # optional trade-level export for offline robustness / plots
    if args.export:
        out = {v: {"n_signals": len(by[v]),
                   "trades": _run_trades(by[v], exec_df, range_start, exec_tf, VARIANTS[v])}
               for v in variants}
        path = os.path.join(args.out_dir, f"results_{sym}.json")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(out, fh, default=str)
        print(f"\n  exported per-variant trades + signals -> {path}")


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
    # --- institutional reporting (all OFF by default → legacy table unchanged) ---
    p.add_argument("--oos-ratio", type=float, default=0.0,
                   help="Hold out the last R fraction as OUT-OF-SAMPLE (e.g. 0.30). 0 = off.")
    p.add_argument("--bootstrap", action="store_true",
                   help="95%% bootstrap CIs for PF/expR/totalR + P(no edge) per variant.")
    p.add_argument("--min-trades", type=int, default=30,
                   help="Below this trade count a result is flagged 'insufficient evidence'.")
    p.add_argument("--min-oos-trades", type=int, default=10,
                   help="Min OOS trades for an OOS verdict to count.")
    p.add_argument("--baseline", default="crypto_pct",
                   help="Variant the ablation compares every lever against.")
    p.add_argument("--export", action="store_true",
                   help="Dump per-variant trade list + equity curve to <out-dir>/results_<sym>.json.")
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
    _span = exec_df.iloc[range_start:]
    _months = max((_span.index[-1] - _span.index[0]).days, 1) / 30.44  # for signals/month

    if a.aggregate_only:
        by = {v: [] for v in variants}
        for fn in sorted(os.listdir(a.out_dir)):
            # filename is sig_<label>_<8-digit-start>.json; the label itself may contain
            # underscores (e.g. "crypto_pct"), so strip the prefix + numeric suffix
            # instead of naively splitting on "_".
            m = re.match(r"^sig_(.+)_(\d{8})\.json$", fn)
            if not m:
                continue
            lbl = m.group(1)
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

    _report(by, exec_df, range_start, n, a.execution_tf, variants, _span, _months, a)


if __name__ == "__main__":
    main()
