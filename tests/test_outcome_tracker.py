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


class _CaptureSender:
    def __init__(self):
        self.msgs = []

    def send(self, text):
        self.msgs.append(text)


def _resolve_loss(t, sender=None):
    t.record(_Sig("short", 100.0, 102.0, 96.0), NOW)   # short setup
    t.update(_bar(103.0, 101.0), sender=sender, now=NOW)  # high >= SL → loss


def _resolve_big_win(t, sender=None):
    t.record(_Sig("long", 100.0, 98.0, 110.0), NOW)    # 5R target (110-100 over 2)
    t.update(_bar(112.0, 99.0), sender=sender, now=NOW)   # high >= TP → big win


class TestPsychNotes:
    """Low-win% / high-PF strategies fail without psychological scaffolding —
    the streak warning + recovery note keep the user in the game."""

    def test_losing_streak_triggers_warning_at_threshold(self):
        t = OutcomeTracker()
        sender = _CaptureSender()
        for _ in range(OutcomeTracker.LOSING_STREAK_WARN_AT):
            _resolve_loss(t, sender)
        warns = [m for m in sender.msgs if "Losing streak" in m]
        assert len(warns) == 1                        # exactly one warning
        assert f"{OutcomeTracker.LOSING_STREAK_WARN_AT} in a row" in warns[0]

    def test_no_warning_below_threshold(self):
        t = OutcomeTracker()
        sender = _CaptureSender()
        for _ in range(OutcomeTracker.LOSING_STREAK_WARN_AT - 1):
            _resolve_loss(t, sender)
        assert not any("Losing streak" in m for m in sender.msgs)

    def test_big_win_after_losses_triggers_recovery_note(self):
        t = OutcomeTracker()
        sender = _CaptureSender()
        for _ in range(4):
            _resolve_loss(t, sender)
        _resolve_big_win(t, sender)
        notes = [m for m in sender.msgs if "covered the last" in m]
        assert len(notes) == 1 and "4 losses" in notes[0]

    def test_streak_resets_on_win(self):
        t = OutcomeTracker()
        sender = _CaptureSender()
        for _ in range(3):
            _resolve_loss(t, sender)
        _resolve_big_win(t, sender)
        assert t._current_loss_streak == 0
        # next 4 losses should be able to trigger a NEW warning
        sender.msgs.clear()
        for _ in range(4):
            _resolve_loss(t, sender)
        assert any("Losing streak" in m for m in sender.msgs)

    def test_summary_shows_active_streak(self):
        t = OutcomeTracker()
        for _ in range(3):
            _resolve_loss(t)
        assert "streak: 3L" in t.summary_line()
