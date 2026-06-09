"""Tests for RRCalculator — Phase 5.4."""

import pytest
from core.risk.rr_calculator import RRCalculator, RRResult


@pytest.fixture
def default_configs():
    risk = {"rr_tiers": {"min_to_enter": 2.0}}
    cost = {
        "default_spread": 0.25,
        "default_slippage": 0.10,
        "commission_per_lot": 0.0,
        "news_slippage_multiplier": 3.0,
        "high_volatility_slippage_multiplier": 2.0,
    }
    return risk, cost


@pytest.fixture
def calc(default_configs):
    return RRCalculator(*default_configs)


class TestLongRR:
    def test_basic_rr(self, calc):
        r = calc.calculate("long", entry=2000.0, sl=1995.0, tp=2015.0)
        assert r.gross_rr == 3.0  # 15/5
        assert r.valid

    def test_costs_reduce_net_rr(self, calc):
        r = calc.calculate("long", entry=2000.0, sl=1995.0, tp=2015.0)
        assert r.net_rr < r.gross_rr

    def test_net_rr_below_min_invalid(self, calc):
        r = calc.calculate("long", entry=2000.0, sl=1995.0, tp=2007.0)
        # gross = 7/5 = 1.4, net even lower
        assert not r.valid

    def test_spread_override(self, calc):
        r1 = calc.calculate("long", entry=2000.0, sl=1995.0, tp=2015.0)
        r2 = calc.calculate("long", entry=2000.0, sl=1995.0, tp=2015.0, spread=1.0)
        assert r2.net_rr < r1.net_rr


class TestShortRR:
    def test_basic_short_rr(self, calc):
        r = calc.calculate("short", entry=2000.0, sl=2005.0, tp=1985.0)
        assert r.gross_rr == 3.0
        assert r.valid

    def test_short_costs(self, calc):
        r = calc.calculate("short", entry=2000.0, sl=2005.0, tp=1985.0)
        assert r.net_rr < r.gross_rr


class TestNewsAndVolatility:
    def test_news_time_increases_slippage(self, calc):
        r_normal = calc.calculate("long", entry=2000.0, sl=1995.0, tp=2015.0)
        r_news = calc.calculate("long", entry=2000.0, sl=1995.0, tp=2015.0, is_news_time=True)
        assert r_news.slippage_cost > r_normal.slippage_cost
        assert r_news.net_rr < r_normal.net_rr

    def test_high_vol_increases_slippage(self, calc):
        r_normal = calc.calculate("long", entry=2000.0, sl=1995.0, tp=2015.0)
        r_vol = calc.calculate("long", entry=2000.0, sl=1995.0, tp=2015.0, is_high_volatility=True)
        assert r_vol.slippage_cost > r_normal.slippage_cost

    def test_news_takes_priority_over_vol(self, calc):
        r_news = calc.calculate("long", entry=2000.0, sl=1995.0, tp=2015.0, is_news_time=True)
        r_both = calc.calculate("long", entry=2000.0, sl=1995.0, tp=2015.0, is_news_time=True, is_high_volatility=True)
        assert r_news.slippage_cost == r_both.slippage_cost


class TestInvalidInputs:
    def test_zero_risk(self, calc):
        r = calc.calculate("long", entry=2000.0, sl=2000.0, tp=2010.0)
        assert not r.valid
        assert "risk <= 0" in r.detail

    def test_sl_beyond_entry_long(self, calc):
        r = calc.calculate("long", entry=2000.0, sl=2005.0, tp=2010.0)
        assert not r.valid


class TestToDict:
    def test_result_to_dict(self, calc):
        r = calc.calculate("long", entry=2000.0, sl=1995.0, tp=2015.0)
        d = r.to_dict()
        assert "gross_rr" in d
        assert "net_rr" in d
        assert "valid" in d
        assert isinstance(d["gross_rr"], float)


class TestEdgeCases:
    def test_direction_whitespace(self, calc):
        r = calc.calculate("  SHORT  ", entry=2000.0, sl=2005.0, tp=1985.0)
        assert r.valid

    def test_commission_included(self):
        risk = {"rr_tiers": {"min_to_enter": 2.0}}
        cost = {"default_spread": 0.25, "default_slippage": 0.10, "commission_per_lot": 5.0}
        c = RRCalculator(risk, cost)
        r = c.calculate("long", entry=2000.0, sl=1995.0, tp=2015.0)
        assert r.commission_cost == 5.0
