"""Tests for DailyLimits — Phase 5.9."""

import pytest
from datetime import date
from core.risk.daily_limits import DailyLimits, DailyLimitResult, DayLockReason


@pytest.fixture
def default_config():
    return {"max_daily_losses": 2, "max_daily_trades": 3, "stop_after_tp": True}


@pytest.fixture
def limits(default_config):
    return DailyLimits(default_config)


TODAY = date(2026, 5, 21)
TOMORROW = date(2026, 5, 22)


class TestTradeAllowed:
    def test_first_trade_allowed(self, limits):
        r = limits.check(TODAY)
        assert r.trade_allowed
        assert r.lock_reason is None

    def test_after_one_trade(self, limits):
        limits.register_trade(TODAY)
        r = limits.check(TODAY)
        assert r.trade_allowed
        assert r.trades_today == 1


class TestMaxTrades:
    def test_locked_at_max_trades(self, limits):
        for _ in range(3):
            limits.register_trade(TODAY)
        r = limits.check(TODAY)
        assert not r.trade_allowed
        assert r.lock_reason == DayLockReason.MAX_TRADES

    def test_two_trades_still_allowed(self, limits):
        limits.register_trade(TODAY)
        limits.register_trade(TODAY)
        r = limits.check(TODAY)
        assert r.trade_allowed


class TestMaxLosses:
    def test_locked_at_max_losses(self, limits):
        limits.register_loss(TODAY)
        limits.register_loss(TODAY)
        r = limits.check(TODAY)
        assert not r.trade_allowed
        assert r.lock_reason == DayLockReason.MAX_LOSSES

    def test_one_loss_still_allowed(self, limits):
        limits.register_loss(TODAY)
        r = limits.check(TODAY)
        assert r.trade_allowed


class TestStopAfterTP:
    def test_locked_after_first_win(self, limits):
        limits.register_win(TODAY)
        r = limits.check(TODAY)
        assert not r.trade_allowed
        assert r.lock_reason == DayLockReason.AFTER_TP

    def test_no_lock_after_win_when_disabled(self):
        lim = DailyLimits({"max_daily_losses": 2, "max_daily_trades": 3, "stop_after_tp": False})
        lim.register_win(TODAY)
        r = lim.check(TODAY)
        assert r.trade_allowed


class TestDailyReset:
    def test_new_day_resets_all(self, limits):
        limits.register_trade(TODAY)
        limits.register_trade(TODAY)
        limits.register_trade(TODAY)
        assert not limits.check(TODAY).trade_allowed

        r = limits.check(TOMORROW)
        assert r.trade_allowed
        assert r.trades_today == 0
        assert r.losses_today == 0

    def test_losses_reset_on_new_day(self, limits):
        limits.register_loss(TODAY)
        limits.register_loss(TODAY)
        assert not limits.check(TODAY).trade_allowed

        r = limits.check(TOMORROW)
        assert r.trade_allowed

    def test_win_lock_reset_on_new_day(self, limits):
        limits.register_win(TODAY)
        assert not limits.check(TODAY).trade_allowed

        r = limits.check(TOMORROW)
        assert r.trade_allowed


class TestManualReset:
    def test_reset_clears_all(self, limits):
        limits.register_trade(TODAY)
        limits.register_loss(TODAY)
        limits.register_win(TODAY)
        limits.reset()
        r = limits.check(TODAY)
        assert r.trade_allowed
        assert r.trades_today == 0


class TestIsLocked:
    def test_is_locked_property(self, limits):
        assert not limits.is_locked
        limits.register_win(TODAY)
        assert limits.is_locked


class TestToDict:
    def test_result_to_dict(self, limits):
        r = limits.check(TODAY)
        d = r.to_dict()
        assert "trade_allowed" in d
        assert "lock_reason" in d
        assert d["trade_allowed"] is True
        assert d["lock_reason"] is None

    def test_locked_to_dict(self, limits):
        limits.register_win(TODAY)
        r = limits.check(TODAY)
        d = r.to_dict()
        assert d["lock_reason"] == "after_tp"


class TestEdgeCases:
    def test_default_config(self):
        lim = DailyLimits({})
        r = lim.check(TODAY)
        assert r.trade_allowed

    def test_counters_track_correctly(self, limits):
        limits.register_trade(TODAY)
        limits.register_loss(TODAY)
        r = limits.check(TODAY)
        assert r.trades_today == 1
        assert r.losses_today == 1
        assert r.wins_today == 0
