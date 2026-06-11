"""Tests for the forward paper-trade OutcomeTracker (measurement only)."""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from core.alerts.outcome_tracker import OutcomeTracker

NOW = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)


class _Sig:
    def __init__(self, direction, entry, sl, tp1, grade="A"):
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.tp1 = tp1
        self.grade = grade


def _bar(high, low):
    return pd.DataFrame({"high": [high], "low": [low], "close": [(high + low) / 2]})


class TestOutcomeTracker:
    def test_long_hits_tp_is_win(self):
        t = OutcomeTracker()
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        t.update(_bar(105.0, 101.0), now=NOW + timedelta(minutes=5))  # high >= tp
        assert t.wins == 1 and t.losses == 0
        assert t.total_r == pytest.approx(2.0)  # (104-100)/(100-98) = 2R

    def test_short_hits_sl_is_loss(self):
        t = OutcomeTracker()
        t.record(_Sig("short", 100.0, 102.0, 96.0), NOW)
        t.update(_bar(103.0, 99.0), now=NOW + timedelta(minutes=5))  # high >= sl
        assert t.losses == 1 and t.total_r == pytest.approx(-1.0)

    def test_sl_checked_first_on_straddle(self):
        # a bar that reaches BOTH SL and TP resolves conservatively as a LOSS
        t = OutcomeTracker()
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        t.update(_bar(105.0, 97.0), now=NOW + timedelta(minutes=5))
        assert t.losses == 1 and t.wins == 0

    def test_unresolved_stays_open(self):
        t = OutcomeTracker()
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        t.update(_bar(101.0, 99.5), now=NOW + timedelta(minutes=5))  # neither hit
        assert t.wins == 0 and t.losses == 0 and len(t._open) == 1

    def test_stale_open_dropped_not_counted(self):
        t = OutcomeTracker(max_open_hours=1.0)
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        t.update(_bar(101.0, 99.5), now=NOW + timedelta(hours=2))  # aged out, never hit
        assert t.wins == 0 and t.losses == 0 and len(t._open) == 0

    def test_profit_factor_and_summary(self):
        t = OutcomeTracker()
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)   # +2R win
        t.update(_bar(105.0, 101.0), now=NOW)
        t.record(_Sig("short", 100.0, 102.0, 96.0), NOW)  # -1R loss
        t.update(_bar(103.0, 99.0), now=NOW)
        assert t.profit_factor() == pytest.approx(2.0)     # 2R won / 1R lost
        s = t.summary_line()
        assert "2 closed" in s and "50% win" in s

    def test_bad_signal_ignored(self):
        t = OutcomeTracker()
        t.record(_Sig("long", 100.0, 100.0, 104.0), NOW)  # risk 0 → ignored
        assert len(t._open) == 0
