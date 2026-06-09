"""Tests for Volatility / ATR Regime Filter — Phase 3.6."""

import pytest
from core.filters.volatility_filter import VolatilityFilter, VolatilityRegime

CONFIG = {
    "low_atr_ratio": 0.5,
    "high_atr_ratio": 1.5,
    "extreme_atr_ratio": 2.5,
    "atr_median_lookback": 100,
}


@pytest.fixture
def vf():
    return VolatilityFilter(CONFIG)


def _stable_atr(value=3.0, n=100):
    return [value] * n


class TestRegimeClassification:
    def test_normal_regime(self, vf):
        atrs = _stable_atr(3.0)
        result = vf.check(atrs)
        assert result.regime == VolatilityRegime.NORMAL
        assert result.trade_allowed is True
        assert result.atr_ratio == pytest.approx(1.0)

    def test_low_regime(self, vf):
        atrs = _stable_atr(3.0, 99) + [1.0]
        result = vf.check(atrs)
        assert result.regime == VolatilityRegime.LOW
        assert result.trade_allowed is False

    def test_high_regime(self, vf):
        atrs = _stable_atr(3.0, 99) + [5.0]
        result = vf.check(atrs)
        assert result.regime == VolatilityRegime.HIGH
        assert result.trade_allowed is True

    def test_extreme_regime(self, vf):
        atrs = _stable_atr(3.0, 99) + [9.0]
        result = vf.check(atrs)
        assert result.regime == VolatilityRegime.EXTREME
        assert result.trade_allowed is False


class TestNoData:
    def test_none(self, vf):
        result = vf.check(None)
        assert result.regime == VolatilityRegime.NO_DATA
        assert result.trade_allowed is False

    def test_too_few(self, vf):
        result = vf.check([3.0] * 5)
        assert result.regime == VolatilityRegime.NO_DATA

    def test_zero_median(self, vf):
        result = vf.check([0.0] * 20)
        assert result.regime == VolatilityRegime.NO_DATA


class TestResultDict:
    def test_to_dict(self, vf):
        d = vf.check(_stable_atr()).to_dict()
        assert d["regime"] == "normal"
        assert d["trade_allowed"] is True
        assert isinstance(d["atr_ratio"], float)
