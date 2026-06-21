"""Tests for the forward paper-trade OutcomeTracker state machine (measurement only).

The tracker is PENDING → OPEN → CLOSED (+ NULLIFIED / EXPIRED). A limit only fills when
a bar brackets the entry (low <= entry <= high); resolution happens on LATER bars with
SL checked first. Unfilled entries are NOT losses."""

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


def _bars(rows, start_min=5, freq_min=5):
    """Build a DataFrame of (high, low) bars with a tz-aware UTC index AFTER NOW."""
    start = NOW + timedelta(minutes=start_min)
    idx = pd.date_range(start, periods=len(rows), freq=f"{freq_min}min", tz="UTC")
    return pd.DataFrame(
        {"high": [r[0] for r in rows], "low": [r[1] for r in rows],
         "close": [(r[0] + r[1]) / 2 for r in rows]},
        index=idx,
    )


class TestStateMachine:
    def test_long_fill_then_tp_is_win(self):
        t = OutcomeTracker()
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        # bar1 brackets entry 100 (fill → OPEN); bar2 hits TP (resolve on a LATER bar)
        t.update(_bars([(101.0, 99.5), (105.0, 101.0)]), now=NOW + timedelta(minutes=10))
        assert t.wins == 1 and t.losses == 0
        assert t.total_r == pytest.approx(2.0)  # (104-100)/(100-98)

    def test_short_fill_then_sl_is_loss(self):
        t = OutcomeTracker()
        t.record(_Sig("short", 100.0, 102.0, 96.0), NOW)
        t.update(_bars([(100.5, 99.5), (103.0, 99.0)]), now=NOW + timedelta(minutes=10))
        assert t.losses == 1 and t.total_r == pytest.approx(-1.0)

    def test_unfilled_long_that_hits_sl_is_NULLIFIED_not_loss(self):
        # THE core bug: entry above price, price drops through SL without ever filling.
        t = OutcomeTracker()
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        t.update(_bars([(99.5, 97.5)]), now=NOW + timedelta(minutes=5))  # never brackets 100
        assert t.wins == 0 and t.losses == 0
        assert t.nullified == 1 and t.total_r == 0.0

    def test_near_real_world_unfilled_long(self):
        # The exact NEAR LONG that exposed the bug (entry 2.1850 / SL 2.1796, price ~2.179).
        t = OutcomeTracker()
        t.record(_Sig("long", 2.1850, 2.1796, 2.2000), NOW)
        t.update(_bars([(2.1800, 2.1780)]), now=NOW + timedelta(minutes=5))
        assert t.losses == 0 and t.wins == 0 and t.nullified == 1

    def test_entry_never_filled_expires(self):
        t = OutcomeTracker(entry_expiry_bars=3)
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        # bars never bracket entry (high<100) and never hit SL (low>98) → EXPIRED after 3
        t.update(_bars([(99.5, 98.5)] * 4), now=NOW + timedelta(minutes=25))
        assert t.expired == 1 and t.wins == 0 and t.losses == 0

    def test_window_scan_uses_all_bars_not_just_newest(self):
        # Regression for BUG 2: resolution happens on a MIDDLE bar; the newest bar alone
        # would (wrongly) say WIN. The scan must resolve the earlier LOSS first.
        t = OutcomeTracker()
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        t.update(_bars([(101.0, 99.5),   # fill
                        (100.0, 97.0),   # SL hit here → LOSS
                        (105.0, 101.0)]),  # a later TP the buggy newest-only check would catch
                 now=NOW + timedelta(minutes=15))
        assert t.losses == 1 and t.wins == 0

    def test_fill_one_cycle_resolve_next_cycle(self):
        t = OutcomeTracker()
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        t.update(_bars([(101.0, 99.5)], start_min=5), now=NOW + timedelta(minutes=5))   # fill only
        assert t.wins == 0 and t.losses == 0 and len(t._open) == 1
        t.update(_bars([(105.0, 101.0)], start_min=10), now=NOW + timedelta(minutes=10))  # resolve
        assert t.wins == 1

    def test_sl_checked_first_on_straddle_after_fill(self):
        t = OutcomeTracker()
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        # bar1 fills; bar2 straddles BOTH SL and TP → conservative LOSS
        t.update(_bars([(101.0, 99.5), (105.0, 97.0)]), now=NOW + timedelta(minutes=10))
        assert t.losses == 1 and t.wins == 0

    def test_filled_but_unresolved_stays_open(self):
        t = OutcomeTracker()
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        t.update(_bars([(101.0, 99.5), (101.0, 99.5)]), now=NOW + timedelta(minutes=10))
        assert t.wins == 0 and t.losses == 0 and len(t._open) == 1  # OPEN, not resolved

    def test_stale_open_dropped_and_counted(self):
        t = OutcomeTracker(max_open_hours=1.0)
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        t.update(_bars([(101.0, 99.5)]), now=NOW + timedelta(hours=2))  # filled then aged out
        assert t.wins == 0 and t.losses == 0 and len(t._open) == 0
        assert t.stale == 1

    def test_bad_signal_ignored(self):
        t = OutcomeTracker()
        t.record(_Sig("long", 100.0, 100.0, 104.0), NOW)  # risk 0 → ignored
        assert len(t._open) == 0

    def test_nullified_excluded_from_stats(self):
        t = OutcomeTracker()
        # one nullified (no trade) + one real win → stats reflect ONLY the win
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        t.update(_bars([(99.5, 97.5)]), now=NOW + timedelta(minutes=5))         # nullified
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)
        t.update(_bars([(101.0, 99.5), (105.0, 101.0)]), now=NOW + timedelta(minutes=10))  # win
        assert t.wins == 1 and t.losses == 0 and t.nullified == 1
        s = t.summary_line()
        assert "1 closed" in s and "100% win" in s and "1 nullified" in s

    def test_profit_factor_and_summary(self):
        t = OutcomeTracker()
        t.record(_Sig("long", 100.0, 98.0, 104.0), NOW)                          # +2R win
        t.update(_bars([(101.0, 99.5), (105.0, 101.0)]), now=NOW + timedelta(minutes=10))
        t.record(_Sig("short", 100.0, 102.0, 96.0), NOW)                         # -1R loss
        t.update(_bars([(100.5, 99.5), (103.0, 99.0)]), now=NOW + timedelta(minutes=10))
        assert t.profit_factor() == pytest.approx(2.0)
        s = t.summary_line()
        assert "2 closed" in s and "50% win" in s


class _CaptureSender:
    def __init__(self):
        self.msgs = []

    def send(self, text):
        self.msgs.append(text)


def _resolve_loss(t, sender=None):
    t.record(_Sig("short", 100.0, 102.0, 96.0), NOW)
    t.update(_bars([(100.5, 99.5), (103.0, 101.0)]), sender=sender, now=NOW + timedelta(minutes=10))


def _resolve_big_win(t, sender=None):
    t.record(_Sig("long", 100.0, 98.0, 110.0), NOW)  # 5R target
    t.update(_bars([(101.0, 99.5), (112.0, 101.0)]), sender=sender, now=NOW + timedelta(minutes=10))


class TestPsychNotes:
    """Low-win% / high-PF strategies fail without psychological scaffolding —
    the streak warning + recovery note keep the user in the game."""

    def test_losing_streak_triggers_warning_at_threshold(self):
        t = OutcomeTracker()
        sender = _CaptureSender()
        for _ in range(OutcomeTracker.LOSING_STREAK_WARN_AT):
            _resolve_loss(t, sender)
        warns = [m for m in sender.msgs if "Losing streak" in m]
        assert len(warns) == 1
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
        sender.msgs.clear()
        for _ in range(4):
            _resolve_loss(t, sender)
        assert any("Losing streak" in m for m in sender.msgs)

    def test_summary_shows_active_streak(self):
        t = OutcomeTracker()
        for _ in range(3):
            _resolve_loss(t)
        assert "streak: 3L" in t.summary_line()
