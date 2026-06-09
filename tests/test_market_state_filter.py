"""Tests for Market State Filter — Phase 3.7."""

import pytest
import numpy as np
from core.filters.market_state_filter import MarketStateFilter, MarketState

CONFIG = {
    "state_lookback_candles": 20,
    "choppy_wick_ratio": 0.6,
    "ranging_cross_ratio": 0.4,
    "news_spike_atr_mult": 3.0,
}


@pytest.fixture
def msf():
    return MarketStateFilter(CONFIG)


def _trending_up(n=25):
    """Strong uptrend with small wicks."""
    opens = [2300 + i * 1.0 for i in range(n)]
    closes = [o + 0.8 for o in opens]
    highs = [c + 0.1 for c in closes]
    lows = [o - 0.1 for o in opens]
    return highs, lows, closes, opens


def _choppy(n=25):
    """Large wicks, tiny bodies → choppy."""
    opens = [2300.0] * n
    closes = [2300.1] * n
    highs = [2305.0] * n
    lows = [2295.0] * n
    return highs, lows, closes, opens


def _ranging(n=25):
    """Oscillating rapidly around midpoint with decent bodies."""
    mid = 2300.0
    amp = 3.0
    opens, closes, highs, lows = [], [], [], []
    for i in range(n):
        val = mid + amp * np.sin(i * 1.5)
        o = val
        c = val + 0.5 * (1 if i % 2 == 0 else -1)
        opens.append(o)
        closes.append(c)
        highs.append(max(o, c) + 0.2)
        lows.append(min(o, c) - 0.2)
    return highs, lows, closes, opens


class TestTrending:
    def test_uptrend(self, msf):
        h, l, c, o = _trending_up()
        result = msf.check(h, l, c, o)
        assert result.state == MarketState.TRENDING
        assert result.trade_allowed is True


class TestChoppy:
    def test_choppy_market(self, msf):
        h, l, c, o = _choppy()
        result = msf.check(h, l, c, o)
        assert result.state == MarketState.CHOPPY
        assert result.trade_allowed is False


class TestRanging:
    def test_ranging_market(self, msf):
        h, l, c, o = _ranging()
        result = msf.check(h, l, c, o)
        assert result.state == MarketState.RANGING
        assert result.trade_allowed is True


class TestNewsSpike:
    def test_spike_detected(self, msf):
        h, l, c, o = _trending_up()
        # Last candle has huge range
        h[-1] = 2340.0
        l[-1] = 2310.0
        result = msf.check(h, l, c, o, atr=3.0)
        assert result.state == MarketState.NEWS_SPIKE
        assert result.trade_allowed is False

    def test_no_spike_without_atr(self, msf):
        h, l, c, o = _trending_up()
        h[-1] = 2340.0
        l[-1] = 2310.0
        result = msf.check(h, l, c, o, atr=None)
        # Without ATR, spike detection skipped
        assert result.state != MarketState.NEWS_SPIKE


class TestNoData:
    def test_insufficient_data(self, msf):
        result = msf.check([1, 2], [1, 2], [1, 2], [1, 2])
        assert result.state == MarketState.NO_DATA
        assert result.trade_allowed is False

    def test_none_input(self, msf):
        result = msf.check(None, None, None, None)
        assert result.state == MarketState.NO_DATA


class TestResultDict:
    def test_to_dict(self, msf):
        h, l, c, o = _trending_up()
        d = msf.check(h, l, c, o).to_dict()
        assert d["state"] == "trending"
        assert d["trade_allowed"] is True
