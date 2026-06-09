"""Tests for DXY Confluence Filter — Phase 3.4."""

import pytest
from core.filters.dxy_filter import DXYFilter, DXYState, DXYAlignment

CONFIG = {
    "dxy_required": False,
    "dxy_lookback_candles": 20,
    "dxy_weak_threshold_pct": -0.05,
    "dxy_strong_threshold_pct": 0.05,
}


@pytest.fixture
def df():
    return DXYFilter(CONFIG)


def _falling_dxy(start=104.0, drop_pct=0.15, n=25):
    """Generate a declining DXY series."""
    step = (start * drop_pct / 100) / (n - 1)
    return [start - step * i for i in range(n)]


def _rising_dxy(start=104.0, rise_pct=0.15, n=25):
    step = (start * rise_pct / 100) / (n - 1)
    return [start + step * i for i in range(n)]


def _flat_dxy(price=104.0, n=25):
    return [price] * n


class TestDXYState:
    def test_weak_dxy(self, df):
        closes = _falling_dxy()
        result = df.check(closes, "long")
        assert result.state == DXYState.WEAK

    def test_strong_dxy(self, df):
        closes = _rising_dxy()
        result = df.check(closes, "short")
        assert result.state == DXYState.STRONG

    def test_neutral_dxy(self, df):
        closes = _flat_dxy()
        result = df.check(closes, "long")
        assert result.state == DXYState.NEUTRAL


class TestAlignment:
    def test_long_xau_weak_dxy_aligned(self, df):
        closes = _falling_dxy()
        result = df.check(closes, "long")
        assert result.alignment == DXYAlignment.ALIGNED

    def test_short_xau_strong_dxy_aligned(self, df):
        closes = _rising_dxy()
        result = df.check(closes, "short")
        assert result.alignment == DXYAlignment.ALIGNED

    def test_long_xau_strong_dxy_not_aligned(self, df):
        closes = _rising_dxy()
        result = df.check(closes, "long")
        assert result.alignment == DXYAlignment.NOT_ALIGNED

    def test_short_xau_weak_dxy_not_aligned(self, df):
        closes = _falling_dxy()
        result = df.check(closes, "short")
        assert result.alignment == DXYAlignment.NOT_ALIGNED

    def test_neutral_dxy_neutral_alignment(self, df):
        closes = _flat_dxy()
        result = df.check(closes, "long")
        assert result.alignment == DXYAlignment.NEUTRAL


class TestNoData:
    def test_none_input(self, df):
        result = df.check(None, "long")
        assert result.state == DXYState.NO_DATA
        assert result.alignment == DXYAlignment.NO_DATA

    def test_empty_list(self, df):
        result = df.check([], "long")
        assert result.state == DXYState.NO_DATA

    def test_single_value(self, df):
        result = df.check([104.0], "long")
        assert result.state == DXYState.NO_DATA

    def test_zero_start(self, df):
        result = df.check([0, 104.0, 104.5], "long")
        assert result.state == DXYState.NO_DATA


class TestIsAligned:
    def test_is_aligned_true(self, df):
        assert df.is_aligned(_falling_dxy(), "long") is True

    def test_is_aligned_false(self, df):
        assert df.is_aligned(_rising_dxy(), "long") is False


class TestLookback:
    def test_uses_last_n_candles(self):
        config = {**CONFIG, "dxy_lookback_candles": 5}
        df = DXYFilter(config)
        # First 20 candles rising, last 5 falling sharply
        rising = _rising_dxy(n=20)
        falling_end = [rising[-1] - i * 0.05 for i in range(5)]
        closes = rising + falling_end
        result = df.check(closes, "long")
        assert result.state == DXYState.WEAK


class TestResultDict:
    def test_to_dict(self, df):
        result = df.check(_falling_dxy(), "long")
        d = result.to_dict()
        assert d["state"] == "weak"
        assert d["alignment"] == "aligned"
        assert isinstance(d["change_percent"], float)
        assert "detail" in d


class TestRequired:
    def test_default_not_required(self, df):
        assert df.required is False

    def test_required_when_configured(self):
        config = {**CONFIG, "dxy_required": True}
        df = DXYFilter(config)
        assert df.required is True
