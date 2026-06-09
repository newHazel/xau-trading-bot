"""Tests for LiveSizeManager — Phase 8.2."""

import pytest
from live.live_size_manager import LiveSizeManager, LiveSizeResult


@pytest.fixture
def manager():
    return LiveSizeManager({
        "capital_allocation_min_pct": 5.0,
        "capital_allocation_max_pct": 10.0,
        "risk_per_trade_min_pct": 0.25,
        "risk_per_trade_max_pct": 0.50,
        "max_daily_trades_live": 1,
        "current_capital_allocation_pct": 5.0,
        "current_risk_per_trade_pct": 0.25,
    })


class TestGetLimits:
    def test_basic_limits(self, manager):
        r = manager.get_limits(total_balance=100000.0)
        assert r.allowed_capital == 5000.0  # 5% of 100k
        assert r.risk_per_trade_pct == 0.25
        assert r.max_risk_money == 12.50  # 0.25% of 5000
        assert r.max_daily_trades == 1

    def test_small_balance(self, manager):
        r = manager.get_limits(total_balance=1000.0)
        assert r.allowed_capital == 50.0
        assert r.max_risk_money == 0.125


class TestValidateTradeSize:
    def test_valid_trade(self, manager):
        r = manager.validate_trade_size(total_balance=100000.0, proposed_risk_money=10.0, daily_trades_so_far=0)
        assert r["allowed"]

    def test_exceeds_daily_limit(self, manager):
        r = manager.validate_trade_size(total_balance=100000.0, proposed_risk_money=10.0, daily_trades_so_far=1)
        assert not r["allowed"]
        assert "daily trade limit" in r["reason"]

    def test_exceeds_risk_limit(self, manager):
        r = manager.validate_trade_size(total_balance=100000.0, proposed_risk_money=20.0, daily_trades_so_far=0)
        assert not r["allowed"]
        assert "risk" in r["reason"]


class TestClamping:
    def test_capital_clamped_to_max(self):
        m = LiveSizeManager({"current_capital_allocation_pct": 15.0, "capital_allocation_max_pct": 10.0})
        r = m.get_limits(100000.0)
        assert r.allowed_capital == 10000.0

    def test_risk_clamped_to_max(self):
        m = LiveSizeManager({"current_risk_per_trade_pct": 1.0, "risk_per_trade_max_pct": 0.50})
        r = m.get_limits(100000.0)
        assert r.risk_per_trade_pct == 0.50


class TestToDict:
    def test_to_dict(self, manager):
        r = manager.get_limits(100000.0)
        d = r.to_dict()
        assert "allowed_capital" in d
        assert "max_risk_money" in d


class TestDefaults:
    def test_default_config(self):
        m = LiveSizeManager()
        r = m.get_limits(100000.0)
        assert r.allowed_capital > 0
        assert r.max_daily_trades >= 1
