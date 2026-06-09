"""Tests for Signal Spacing Guard — Phase 4.7."""

import pytest
from datetime import datetime, timedelta
from core.engine.signal_spacing_guard import SignalSpacingGuard

CONFIG = {
    "signal_spacing": {
        "min_minutes_between_signals_same_direction": 30,
        "min_atr_distance_between_entries": 1.0,
    },
}


@pytest.fixture
def ssg():
    return SignalSpacingGuard(CONFIG)


class TestTimeSpacing:
    def test_first_signal_always_ok(self, ssg):
        ok, reason = ssg.can_send("long", 2340.0, datetime(2026, 3, 18, 12, 0), atr=3.0)
        assert ok is True

    def test_too_soon_same_direction(self, ssg):
        t0 = datetime(2026, 3, 18, 12, 0)
        ssg.register_signal("long", 2340.0, t0)
        ok, reason = ssg.can_send("long", 2350.0, t0 + timedelta(minutes=20), atr=3.0)
        assert ok is False
        assert "too soon" in reason

    def test_ok_after_min_time(self, ssg):
        t0 = datetime(2026, 3, 18, 12, 0)
        ssg.register_signal("long", 2340.0, t0)
        ok, _ = ssg.can_send("long", 2350.0, t0 + timedelta(minutes=35), atr=3.0)
        assert ok is True

    def test_opposite_direction_not_blocked(self, ssg):
        t0 = datetime(2026, 3, 18, 12, 0)
        ssg.register_signal("long", 2340.0, t0)
        ok, _ = ssg.can_send("short", 2340.0, t0 + timedelta(minutes=5), atr=3.0)
        assert ok is True


class TestPriceSpacing:
    def test_too_close_in_price(self, ssg):
        t0 = datetime(2026, 3, 18, 12, 0)
        ssg.register_signal("long", 2340.0, t0)
        # 35 min later (past time check), but only 2.0 away with ATR=3.0 → 0.67 ATR < 1.0
        ok, reason = ssg.can_send("long", 2342.0, t0 + timedelta(minutes=35), atr=3.0)
        assert ok is False
        assert "too close" in reason

    def test_far_enough_in_price(self, ssg):
        t0 = datetime(2026, 3, 18, 12, 0)
        ssg.register_signal("long", 2340.0, t0)
        ok, _ = ssg.can_send("long", 2344.0, t0 + timedelta(minutes=35), atr=3.0)
        assert ok is True


class TestCleanup:
    def test_cleanup_removes_old(self, ssg):
        t0 = datetime(2026, 3, 18, 8, 0)
        ssg.register_signal("long", 2340.0, t0)
        ssg.cleanup(t0 + timedelta(minutes=300))
        ok, _ = ssg.can_send("long", 2340.0, t0 + timedelta(minutes=300), atr=3.0)
        assert ok is True

    def test_reset(self, ssg):
        ssg.register_signal("long", 2340.0, datetime(2026, 3, 18, 12, 0))
        ssg.reset()
        ok, _ = ssg.can_send("long", 2340.0, datetime(2026, 3, 18, 12, 1), atr=3.0)
        assert ok is True
