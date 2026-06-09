"""Tests for PositionSizer — Phase 5.5."""

import pytest
from core.risk.position_sizer import PositionSizer, PositionSizeResult


@pytest.fixture
def default_configs():
    risk = {"risk_per_trade_percent": 0.5, "max_risk_per_trade_percent": 1.0}
    cost = {
        "default_spread": 0.25,
        "default_slippage": 0.10,
        "commission_per_lot": 0.0,
        "point_value_per_lot": 100.0,
    }
    return risk, cost


@pytest.fixture
def sizer(default_configs):
    return PositionSizer(*default_configs)


class TestBasicSizing:
    def test_valid_calculation(self, sizer):
        r = sizer.calculate(account_balance=10000.0, entry=2000.0, sl=1995.0)
        assert r.valid
        assert r.lot_size > 0

    def test_never_rounds_up(self, sizer):
        r = sizer.calculate(account_balance=10000.0, entry=2000.0, sl=1995.0)
        import math
        assert r.lot_size == math.floor(r.lot_size * 100) / 100

    def test_risk_within_limit(self, sizer):
        r = sizer.calculate(account_balance=10000.0, entry=2000.0, sl=1995.0)
        max_risk = 10000.0 * (0.5 / 100)
        assert r.money_at_risk_net <= max_risk + 0.01  # small float tolerance

    def test_sl_distance_correct(self, sizer):
        r = sizer.calculate(account_balance=10000.0, entry=2000.0, sl=1995.0)
        assert r.sl_distance_points == 5.0

    def test_cost_per_lot(self, sizer):
        r = sizer.calculate(account_balance=10000.0, entry=2000.0, sl=1995.0)
        expected_cost = (0.25 + 0.10) * 100.0 + 0.0  # spread + slippage * point_value + commission
        assert r.cost_per_lot == expected_cost


class TestSizeReduction:
    def test_reduce_when_exceeds_max_risk(self):
        risk = {"risk_per_trade_percent": 2.0, "max_risk_per_trade_percent": 1.0}
        cost = {"default_spread": 0.0, "default_slippage": 0.0, "commission_per_lot": 0.0, "point_value_per_lot": 100.0}
        s = PositionSizer(risk, cost)
        r = s.calculate(account_balance=10000.0, entry=2000.0, sl=1995.0)
        max_allowed = 10000.0 * (1.0 / 100)
        assert r.money_at_risk_net <= max_allowed + 0.01


class TestInvalidInputs:
    def test_zero_sl_distance(self, sizer):
        r = sizer.calculate(account_balance=10000.0, entry=2000.0, sl=2000.0)
        assert not r.valid
        assert "zero" in r.detail.lower()

    def test_tiny_balance(self, sizer):
        r = sizer.calculate(account_balance=10.0, entry=2000.0, sl=1995.0)
        assert not r.valid or r.lot_size == 0

    def test_zero_balance(self, sizer):
        r = sizer.calculate(account_balance=0.0, entry=2000.0, sl=1995.0)
        assert not r.valid


class TestSpreadOverride:
    def test_custom_spread(self, sizer):
        r1 = sizer.calculate(account_balance=10000.0, entry=2000.0, sl=1995.0)
        r2 = sizer.calculate(account_balance=10000.0, entry=2000.0, sl=1995.0, spread=1.0)
        assert r2.cost_per_lot > r1.cost_per_lot
        assert r2.lot_size <= r1.lot_size

    def test_custom_slippage(self, sizer):
        r1 = sizer.calculate(account_balance=10000.0, entry=2000.0, sl=1995.0)
        r2 = sizer.calculate(account_balance=10000.0, entry=2000.0, sl=1995.0, slippage=0.5)
        assert r2.lot_size <= r1.lot_size


class TestToDict:
    def test_result_to_dict(self, sizer):
        r = sizer.calculate(account_balance=10000.0, entry=2000.0, sl=1995.0)
        d = r.to_dict()
        assert "lot_size" in d
        assert "valid" in d
        assert d["valid"] is True


class TestEdgeCases:
    def test_default_config(self):
        s = PositionSizer({}, {})
        r = s.calculate(account_balance=10000.0, entry=2000.0, sl=1995.0)
        assert r.valid

    def test_large_balance(self, sizer):
        r = sizer.calculate(account_balance=1000000.0, entry=2000.0, sl=1995.0)
        assert r.valid
        assert r.lot_size > 0
