"""Tests for Re-Entry Guard — Phase 4.8."""

import pytest
from datetime import datetime, timedelta
from core.engine.reentry_guard import ReentryGuard

CONFIG = {
    "re_entry": {
        "allow_after_sl": False,
        "allow_after_tp": False,
        "require_new_setup_id": True,
        "require_new_sweep": True,
        "cooldown_minutes_after_loss": 60,
    },
}


@pytest.fixture
def rg():
    return ReentryGuard(CONFIG)


class TestNoHistory:
    def test_can_enter_with_no_exits(self, rg):
        ok, reason = rg.can_enter("long", datetime.now(), "S1")
        assert ok is True
        assert reason is None


class TestAfterSL:
    def test_blocked_during_cooldown(self, rg):
        t0 = datetime(2026, 3, 18, 12, 0)
        rg.register_exit("long", "sl", t0, "S1")
        ok, reason = rg.can_enter("long", t0 + timedelta(minutes=30), "S2", has_new_sweep=True)
        assert ok is False
        assert "cooldown" in reason

    def test_blocked_same_setup_id(self, rg):
        t0 = datetime(2026, 3, 18, 12, 0)
        rg.register_exit("long", "sl", t0, "S1")
        ok, reason = rg.can_enter("long", t0 + timedelta(minutes=90), "S1", has_new_sweep=True)
        assert ok is False
        assert "new setup_id" in reason

    def test_blocked_no_new_sweep(self, rg):
        t0 = datetime(2026, 3, 18, 12, 0)
        rg.register_exit("long", "sl", t0, "S1")
        ok, reason = rg.can_enter("long", t0 + timedelta(minutes=90), "S2", has_new_sweep=False)
        assert ok is False
        assert "new sweep" in reason

    def test_allowed_with_all_conditions(self, rg):
        t0 = datetime(2026, 3, 18, 12, 0)
        rg.register_exit("long", "sl", t0, "S1")
        ok, _ = rg.can_enter("long", t0 + timedelta(minutes=90), "S2", has_new_sweep=True)
        assert ok is True

    def test_opposite_direction_not_affected(self, rg):
        t0 = datetime(2026, 3, 18, 12, 0)
        rg.register_exit("long", "sl", t0, "S1")
        ok, _ = rg.can_enter("short", t0 + timedelta(minutes=5), "S2")
        assert ok is True


class TestAfterTP:
    def test_blocked_same_setup_id(self, rg):
        t0 = datetime(2026, 3, 18, 12, 0)
        rg.register_exit("long", "tp1", t0, "S1")
        ok, reason = rg.can_enter("long", t0 + timedelta(minutes=5), "S1")
        assert ok is False
        assert "new setup_id" in reason

    def test_allowed_new_setup_id(self, rg):
        t0 = datetime(2026, 3, 18, 12, 0)
        rg.register_exit("long", "tp1", t0, "S1")
        ok, _ = rg.can_enter("long", t0 + timedelta(minutes=5), "S2")
        assert ok is True


class TestAllowAfterSL:
    def test_allow_after_sl_config(self):
        config = {
            "re_entry": {
                "allow_after_sl": True,
                "allow_after_tp": True,
                "require_new_setup_id": True,
                "require_new_sweep": True,
                "cooldown_minutes_after_loss": 60,
            },
        }
        rg = ReentryGuard(config)
        t0 = datetime(2026, 3, 18, 12, 0)
        rg.register_exit("long", "sl", t0, "S1")
        # Still needs new setup_id and new sweep
        ok, reason = rg.can_enter("long", t0 + timedelta(minutes=5), "S2", has_new_sweep=True)
        assert ok is True


class TestReset:
    def test_reset_clears(self, rg):
        rg.register_exit("long", "sl", datetime.now(), "S1")
        rg.reset()
        ok, _ = rg.can_enter("long", datetime.now(), "S1")
        assert ok is True
