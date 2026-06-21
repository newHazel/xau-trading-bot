"""
Phase 12.2 — Outcome labeler (SKELETON).

Given a setup (direction + entry/SL/TP1) and the FUTURE exec-TF candles after the
signal bar, compute the ML training label in a leakage-safe way.

Two labels, matching the two canonical engines already in the codebase so the
dataset is definitionally consistent with reported metrics:

  1) BINARY  `tp1_before_sl`  (primary)  — did price reach TP1 before SL?
     Mirrors core/alerts/outcome_tracker.OutcomeTracker._resolve EXACTLY
     (SL checked FIRST on a straddling bar = conservative; win/loss only, no costs).
     Cost-free, so it never drifts with the cost config. This is the recommended
     target for the first model.

  2) REGRESSION  `net_r`  (secondary) — realized R via the conservative fill engine
     (backtesting.BacktestRunner + FillEngine), cost-aware. Optional; lazy-imported.

ENTRY-TRIGGER (F3) DISCIPLINE: a setup is only a real trade if price actually trades
through the limit `entry` within `entry_expiry_bars` of the signal (matches
BacktestRunner._check_pending_entry, default 12). Setups whose entry is never touched
are returned with triggered=False and NO label — they must be dropped from the
supervised set, NOT imputed as losses (that would be survivorship bias).

NO-LOOKAHEAD: the caller passes ONLY bars strictly AFTER the signal bar. This module
never sees the signal bar or anything before it, so a feature can never overlap a
label's outcome window.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Canonical entry-trigger horizon (BacktestRunner default).
DEFAULT_ENTRY_EXPIRY_BARS = 12


def resolve_outcome(
    direction: str, entry: float, sl: float, tp1: float, hi: float, lo: float
) -> Tuple[Optional[str], float]:
    """One-bar TP1-vs-SL resolution. EXACT mirror of OutcomeTracker._resolve.

    SL is checked FIRST so a bar that straddles both counts as a LOSS (conservative).
    Returns ("WIN", r) | ("LOSS", -1.0) | (None, 0.0 = not hit yet).
    `r` for a win is the gross reward-to-risk (cost-free), matching the forward tracker.
    """
    risk = abs(entry - sl)
    if risk <= 0:
        return None, 0.0
    if direction == "long":
        if lo <= sl:
            return "LOSS", -1.0
        if hi >= tp1:
            return "WIN", (tp1 - entry) / risk
    elif direction == "short":
        if hi >= sl:
            return "LOSS", -1.0
        if lo <= tp1:
            return "WIN", (entry - tp1) / risk
    return None, 0.0


def label_binary(
    direction: str,
    entry: float,
    sl: float,
    tp1: float,
    future_highs: List[float],
    future_lows: List[float],
    entry_expiry_bars: int = DEFAULT_ENTRY_EXPIRY_BARS,
    max_hold_bars: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute the TP1-before-SL binary label for one setup.

    Args:
        future_highs/future_lows: exec-TF highs/lows for bars STRICTLY AFTER the
            signal bar, in chronological order (index 0 = first bar after signal).
        entry_expiry_bars: drop the setup if entry isn't touched within this many bars.
        max_hold_bars: stop resolving after this many bars past FILL (None = to end).

    Returns dict:
        triggered     : bool  — did the limit entry fill within the expiry window?
        outcome       : "WIN" | "LOSS" | "NO_FILL" | "OPEN"
        tp1_before_sl : 1 | 0 | None   (None when NO_FILL or OPEN/censored)
        win_r         : float | None   (gross R of a win; -1.0 for a loss)
        fill_offset   : int | None     (bars after signal where entry filled)
        resolve_offset: int | None     (bars after signal where TP1/SL hit)
    """
    n = min(len(future_highs), len(future_lows))
    risk = abs(entry - sl)
    base = {
        "triggered": False, "outcome": "NO_FILL", "tp1_before_sl": None,
        "win_r": None, "fill_offset": None, "resolve_offset": None,
    }
    if risk <= 0 or n == 0:
        return base

    # --- step 1: entry trigger (limit fills when a bar brackets `entry`) ---
    fill_idx: Optional[int] = None
    for i in range(min(entry_expiry_bars, n)):
        if future_lows[i] <= entry <= future_highs[i]:
            fill_idx = i
            break
    if fill_idx is None:
        return base  # never filled → no trade, no label

    # --- step 2: resolve TP1 vs SL on bars AFTER the fill bar (conservative) ---
    end = n if max_hold_bars is None else min(n, fill_idx + 1 + max_hold_bars)
    for j in range(fill_idx + 1, end):
        label, r = resolve_outcome(direction, entry, sl, tp1, future_highs[j], future_lows[j])
        if label is not None:
            return {
                "triggered": True,
                "outcome": label,
                "tp1_before_sl": 1 if label == "WIN" else 0,
                "win_r": r,
                "fill_offset": fill_idx,
                "resolve_offset": j,
            }
    # filled but never resolved within the window → censored (exclude from training)
    return {
        "triggered": True, "outcome": "OPEN", "tp1_before_sl": None,
        "win_r": None, "fill_offset": fill_idx, "resolve_offset": None,
    }


def label_net_r(
    signal_dict: Dict[str, Any],
    exec_slice: Any,
    cost_overrides: Optional[Dict[str, Any]] = None,
    entry_expiry_bars: int = DEFAULT_ENTRY_EXPIRY_BARS,
) -> Dict[str, Any]:
    """Secondary cost-aware label: realized R via the conservative fill engine.

    Reuses backtesting.BacktestRunner end-to-end so net_r matches the project's
    reported backtest metrics (the whole system judges by TradeRecord.r_multiple).
    `exec_slice` MUST start at the signal bar (the signal is placed at bar_index 0).
    `cost_overrides` mirrors a VARIANT's {"costs": {...}} block: when cost_model is
    "percent", spread/slippage are scaled to the slice's median price (the crypto fix);
    otherwise the gold-absolute 0.25 / 0.10 model is used.
    `entry_expiry_bars` is forced onto the BacktestConfig so net_r uses the SAME
    entry-trigger window as the binary label (otherwise the runner defaults to 12 and
    the two labels could disagree on which setups are 'triggered').

    SEMANTICS / KNOWN APPROXIMATION: net_r is the realized R of the FINAL exit of the
    RESIDUAL position. The backtest engine books a 50% partial at TP1 but does NOT blend
    that partial profit into the recorded TradeRecord.r_multiple — so a setup that taps
    TP1 (+partial) then reverses into the original SL is recorded as ~ -1R / sl_hit. That
    is why a row can have tp1_before_sl=1 (TP1 WAS touched first) yet net_r exit_type
    'sl_hit' — it is EXPECTED, not a bug, and mirrors how the live/backtest system already
    measures R everywhere. Prefer tp1_before_sl as the primary training target; treat
    net_r as a secondary, conservative signal.

    Returns {"triggered": bool, "net_r": float|None, "exit_type": str|None}.
    Lazy-imports the backtest engine so importing this module stays cheap/dependency-free.
    """
    from backtesting.backtest_runner import BacktestRunner, BacktestConfig

    costs_cfg = (cost_overrides or {}).get("costs", {})
    if str(costs_cfg.get("cost_model", "absolute")).lower() == "percent":
        ref_price = float(exec_slice["close"].median())
        spread = float(costs_cfg.get("spread_pct", 0.0)) * ref_price
        slippage = float(costs_cfg.get("slippage_pct", 0.0)) * ref_price
    else:
        spread, slippage = 0.25, 0.10

    fe = [{
        "setup_id": signal_dict["setup_id"], "direction": signal_dict["direction"],
        "entry": signal_dict["entry"], "sl": signal_dict["sl"],
        "tp1": signal_dict["tp1"], "tp2": signal_dict.get("tp2", signal_dict["tp1"]),
        "lot_size": 0.1, "bar_index": 0, "grade": signal_dict.get("grade", ""),
    }]
    cfg_obj = BacktestConfig(
        initial_balance=10000.0, conservative_fills=True, costs_inclusive=True,
        default_spread=spread, default_slippage=slippage,
        max_daily_trades=999, max_daily_losses=999,
        base_timeframe=signal_dict.get("execution_tf", "5m"),
    )
    # _check_pending_entry reads getattr(config, 'entry_trigger_expiry_bars', 12) — set it
    # on the instance so net_r shares the binary label's fill window (no dataclass edit).
    try:
        cfg_obj.entry_trigger_expiry_bars = int(entry_expiry_bars)
    except Exception:
        pass
    bt = BacktestRunner(cfg_obj)
    trades = bt.run(exec_slice, signals=fe).trades
    if not trades:
        return {"triggered": False, "net_r": None, "exit_type": None}
    t = trades[0]
    exit_type = getattr(t, "exit_type", None)
    if exit_type == "forced_close":  # censored (ran out of data) — not a real outcome
        return {"triggered": True, "net_r": None, "exit_type": exit_type}
    return {"triggered": True, "net_r": float(getattr(t, "r_multiple", 0.0)), "exit_type": exit_type}
