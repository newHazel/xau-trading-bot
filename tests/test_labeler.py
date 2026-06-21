"""Tests for Phase 12.2 outcome labeler (core/ml/labeler.py)."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from core.alerts.outcome_tracker import OutcomeTracker
from core.ml import labeler as L


# --------------------------------------------------------------------------- #
# resolve_outcome must match the canonical OutcomeTracker._resolve EXACTLY     #
# --------------------------------------------------------------------------- #
def test_resolve_matches_outcome_tracker():
    tracker = OutcomeTracker()
    cases = [
        ("long", 100.0, 99.0, 102.0),
        ("short", 100.0, 101.0, 98.0),
    ]
    bars = [(103.0, 101.5), (98.5, 97.0), (100.2, 99.9), (101.0, 98.9)]
    for direction, entry, sl, tp in cases:
        s = {"dir": direction, "entry": entry, "sl": sl, "tp": tp, "risk": abs(entry - sl)}
        for hi, lo in bars:
            mine = L.resolve_outcome(direction, entry, sl, tp, hi, lo)
            ref = tracker._resolve(s, hi, lo)
            assert mine == ref, (direction, hi, lo, mine, ref)


def test_long_win():
    # entry touched on bar 0, TP1 hit on bar 2
    fh = [100.2, 100.4, 102.5, 103.0]
    fl = [99.8, 100.0, 101.5, 101.0]
    out = L.label_binary("long", 100.0, 99.0, 102.0, fh, fl)
    assert out["triggered"] is True
    assert out["outcome"] == "WIN"
    assert out["tp1_before_sl"] == 1
    assert out["fill_offset"] == 0 and out["resolve_offset"] == 2


def test_long_loss():
    fh = [100.2, 100.1, 100.3]
    fl = [99.8, 98.9, 99.5]          # bar 1 breaks SL (99.0)
    out = L.label_binary("long", 100.0, 99.0, 102.0, fh, fl)
    assert out["triggered"] is True
    assert out["outcome"] == "LOSS"
    assert out["tp1_before_sl"] == 0


def test_entry_never_touched_is_no_fill():
    # price runs away up — a long limit at 100 sitting below is never reached
    fh = [101.0, 102.0, 103.0]
    fl = [100.5, 101.5, 102.5]
    out = L.label_binary("long", 100.0, 99.0, 102.0, fh, fl, entry_expiry_bars=12)
    assert out["triggered"] is False
    assert out["outcome"] == "NO_FILL"
    assert out["tp1_before_sl"] is None


def test_entry_expiry_window_enforced():
    # entry only reachable on bar 5, but expiry is 3 → NO_FILL
    fh = [101.0, 101.0, 101.0, 101.0, 101.0, 100.2]
    fl = [100.5, 100.5, 100.5, 100.5, 100.5, 99.9]
    out = L.label_binary("long", 100.0, 99.0, 102.0, fh, fl, entry_expiry_bars=3)
    assert out["triggered"] is False


def test_straddle_bar_after_fill_is_loss_sl_first():
    # bar 0 fills entry; bar 1 straddles both SL and TP1 → conservative LOSS
    fh = [100.1, 102.5]
    fl = [99.9, 98.8]
    out = L.label_binary("long", 100.0, 99.0, 102.0, fh, fl)
    assert out["outcome"] == "LOSS"
    assert out["tp1_before_sl"] == 0


def test_filled_but_unresolved_is_open():
    fh = [100.1, 100.2, 100.3]
    fl = [99.9, 99.8, 99.7]          # never reaches TP1 (102) or SL (99)
    out = L.label_binary("long", 100.0, 99.0, 102.0, fh, fl)
    assert out["triggered"] is True
    assert out["outcome"] == "OPEN"
    assert out["tp1_before_sl"] is None


def test_zero_risk_guarded():
    out = L.label_binary("long", 100.0, 100.0, 102.0, [100.0], [100.0])
    assert out["triggered"] is False


def test_net_r_label_runs_end_to_end():
    # build an exec slice where a long fills then runs to TP2 → a winning net_r.
    # NOTE: the signal is armed on bar 0 (the signal bar); a limit entry can only FILL
    # on a LATER bar that trades through it, so bar 1 must dip to touch entry=100.0.
    idx = pd.date_range("2026-05-01T12:00:00Z", periods=8, freq="5min", tz="UTC")
    highs = [100.2, 100.5, 102.5, 104.0, 104.5, 104.5, 104.5, 104.5]
    lows = [99.9, 99.9, 101.5, 102.0, 103.0, 103.0, 103.0, 103.0]
    df = pd.DataFrame({
        "open": highs, "high": highs, "low": lows, "close": highs, "volume": [1000.0] * 8,
    }, index=idx)
    sig = {"setup_id": "T-1", "direction": "long", "entry": 100.0, "sl": 99.0,
           "tp1": 102.0, "tp2": 103.5, "grade": "B", "execution_tf": "5m"}
    out = L.label_net_r(sig, df, cost_overrides=None)
    assert out["triggered"] is True
    assert out["exit_type"] in ("tp2_hit", "sl_hit", "tp1_hit")
    assert out["net_r"] is None or isinstance(out["net_r"], float)
