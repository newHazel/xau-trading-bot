"""Tests for Spread Filter — Phase 3.8."""

import pytest
from core.filters.spread_filter import SpreadFilter

CONFIG = {
    "default_spread": 0.25,
    "max_spread_atr_ratio": 0.15,
    "use_bid_ask_if_available": True,
}


@pytest.fixture
def sf():
    return SpreadFilter(CONFIG)


class TestSpreadCheck:
    def test_normal_spread_allowed(self, sf):
        result = sf.check(spread=0.20, atr=3.0)
        assert result.trade_allowed is True
        assert result.max_spread == pytest.approx(0.45)

    def test_wide_spread_blocked(self, sf):
        result = sf.check(spread=0.60, atr=3.0)
        assert result.trade_allowed is False

    def test_exact_boundary(self, sf):
        # max = 4.0 * 0.15 = 0.60, spread = 0.60 → allowed (<=)
        result = sf.check(spread=0.60, atr=4.0)
        assert result.trade_allowed is True

    def test_default_spread_when_none(self, sf):
        result = sf.check(spread=None, atr=3.0)
        assert result.current_spread == 0.25
        assert result.trade_allowed is True

    def test_no_atr_uses_fallback_max(self, sf):
        # max = default_spread * 3 = 0.75
        result = sf.check(spread=0.50, atr=None)
        assert result.trade_allowed is True
        assert result.max_spread == pytest.approx(0.75)

    def test_no_atr_wide_spread_blocked(self, sf):
        result = sf.check(spread=1.0, atr=None)
        assert result.trade_allowed is False


class TestIsTradeAllowed:
    def test_shortcut_allowed(self, sf):
        assert sf.is_trade_allowed(spread=0.20, atr=3.0) is True

    def test_shortcut_blocked(self, sf):
        assert sf.is_trade_allowed(spread=0.60, atr=3.0) is False


class TestResultDict:
    def test_to_dict(self, sf):
        d = sf.check(spread=0.20, atr=3.0).to_dict()
        assert d["trade_allowed"] is True
        assert isinstance(d["current_spread"], float)
        assert isinstance(d["max_spread"], float)
