"""Tests for TakeProfitCalculator — Phase 5.3b."""

import pytest
from core.risk.take_profit import TakeProfitCalculator, TakeProfitResult


@pytest.fixture
def default_config():
    return {"tp1_r": 2.0, "tp2_r": 3.5}


@pytest.fixture
def calc(default_config):
    return TakeProfitCalculator(default_config)


class TestLongTP:
    def test_tp1_at_2r(self, calc):
        r = calc.calculate("long", entry=2000.0, sl_distance=5.0)
        assert r.tp1 == 2010.0  # 2000 + 5*2

    def test_tp2_default_at_3_5r(self, calc):
        r = calc.calculate("long", entry=2000.0, sl_distance=5.0)
        assert r.tp2 == 2017.5  # 2000 + 5*3.5
        assert not r.tp2_from_liquidity

    def test_tp2_from_liquidity_above_tp1(self, calc):
        r = calc.calculate("long", entry=2000.0, sl_distance=5.0, liquidity_target_price=2015.0)
        assert r.tp2 == 2015.0
        assert r.tp2_from_liquidity

    def test_tp2_liquidity_below_tp1_uses_default(self, calc):
        r = calc.calculate("long", entry=2000.0, sl_distance=5.0, liquidity_target_price=2008.0)
        assert r.tp2 == 2017.5
        assert not r.tp2_from_liquidity

    def test_tp1_r_value(self, calc):
        r = calc.calculate("long", entry=2000.0, sl_distance=5.0)
        assert r.tp1_r == 2.0

    def test_tp2_r_from_liquidity(self, calc):
        r = calc.calculate("long", entry=2000.0, sl_distance=5.0, liquidity_target_price=2020.0)
        assert r.tp2_r == 4.0  # (2020-2000)/5


class TestShortTP:
    def test_tp1_at_2r_short(self, calc):
        r = calc.calculate("short", entry=2000.0, sl_distance=5.0)
        assert r.tp1 == 1990.0  # 2000 - 5*2

    def test_tp2_default_short(self, calc):
        r = calc.calculate("short", entry=2000.0, sl_distance=5.0)
        assert r.tp2 == 1982.5  # 2000 - 5*3.5

    def test_tp2_from_liquidity_below_tp1_short(self, calc):
        r = calc.calculate("short", entry=2000.0, sl_distance=5.0, liquidity_target_price=1985.0)
        assert r.tp2 == 1985.0
        assert r.tp2_from_liquidity

    def test_tp2_liquidity_above_tp1_short_uses_default(self, calc):
        r = calc.calculate("short", entry=2000.0, sl_distance=5.0, liquidity_target_price=1995.0)
        assert r.tp2 == 1982.5
        assert not r.tp2_from_liquidity


class TestCustomConfig:
    def test_custom_r_values(self):
        c = TakeProfitCalculator({"tp1_r": 1.5, "tp2_r": 4.0})
        r = c.calculate("long", entry=2000.0, sl_distance=5.0)
        assert r.tp1 == 2007.5
        assert r.tp2 == 2020.0


class TestToDict:
    def test_result_to_dict(self, calc):
        r = calc.calculate("long", entry=2000.0, sl_distance=5.0)
        d = r.to_dict()
        assert d["tp1"] == 2010.0
        assert d["tp2"] == 2017.5
        assert d["tp1_r"] == 2.0
        assert d["tp2_from_liquidity"] is False


class TestEdgeCases:
    def test_direction_whitespace(self, calc):
        r = calc.calculate("  LONG  ", entry=2000.0, sl_distance=5.0)
        assert r.tp1 == 2010.0

    def test_detail_includes_liquidity_note(self, calc):
        r = calc.calculate("long", entry=2000.0, sl_distance=5.0, liquidity_target_price=2015.0)
        assert "liquidity" in r.detail.lower()
