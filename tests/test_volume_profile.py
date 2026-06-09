"""Tests for VolumeProfile — Phase 11.4."""

import pytest
from datetime import datetime, timezone, timedelta
from core.indicators.volume_profile import VolumeProfile, PriceLevel


def _c(ts, h, l, c, v=1000):
    return {"timestamp": ts, "open": c, "high": h, "low": l, "close": c, "volume": v}


NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def vp():
    return VolumeProfile({"window_candles": 100, "num_bins": 20})


class TestWarmup:
    def test_returns_none_before_warmup(self, vp):
        for i in range(5):
            r = vp.update(_c(NOW + timedelta(minutes=i), 2655, 2645, 2650), atr=5.0)
            assert r is None

    def test_returns_reading_after_warmup(self, vp):
        for i in range(25):
            r = vp.update(_c(NOW + timedelta(minutes=i), 2655, 2645, 2650), atr=5.0)
        assert r is not None
        assert r.poc > 0


class TestPOC:
    def test_poc_at_high_volume_zone(self, vp):
        # Most candles at ~2650, a few outliers — POC should be near 2650
        for i in range(30):
            vp.update(_c(NOW + timedelta(minutes=i), 2655, 2645, 2650, v=10000), atr=5.0)
        # Add lower-volume outlier candles
        for i in range(30, 35):
            vp.update(_c(NOW + timedelta(minutes=i), 2705, 2695, 2700, v=100), atr=5.0)
        r = vp.update(_c(NOW + timedelta(minutes=35), 2655, 2645, 2650, v=10000), atr=5.0)
        assert abs(r.poc - 2650) < 5


class TestHVNLVN:
    def test_finds_hvn(self, vp):
        for i in range(50):
            vp.update(_c(NOW + timedelta(minutes=i), 2655, 2645, 2650, v=10000), atr=5.0)
        r = vp.update(_c(NOW + timedelta(minutes=50), 2700, 2690, 2695, v=100), atr=5.0)
        assert len(r.hvn_levels) > 0

    def test_value_area_bounds(self, vp):
        for i in range(60):
            close = 2650 + (i % 10 - 5)
            vp.update(_c(NOW + timedelta(minutes=i), close + 2, close - 2, close, v=1000), atr=5.0)
        r = vp.update(_c(NOW + timedelta(minutes=60), 2650, 2645, 2648, v=1000), atr=5.0)
        assert r.value_area_high >= r.value_area_low
        assert r.value_area_low <= r.poc <= r.value_area_high


class TestClassification:
    def test_at_poc(self, vp):
        for i in range(50):
            vp.update(_c(NOW + timedelta(minutes=i), 2655, 2645, 2650, v=10000), atr=5.0)
        r = vp.update(_c(NOW + timedelta(minutes=50), 2655, 2645, 2650, v=1000), atr=5.0)
        assert r.current_level in (PriceLevel.POC, PriceLevel.HVN)


class TestReset:
    def test_reset(self, vp):
        for i in range(30):
            vp.update(_c(NOW + timedelta(minutes=i), 2655, 2645, 2650), atr=5.0)
        vp.reset()
        assert vp.candle_count == 0


class TestWindow:
    def test_window_caps(self):
        vp = VolumeProfile({"window_candles": 10, "num_bins": 5})
        for i in range(20):
            vp.update(_c(NOW + timedelta(minutes=i), 2655, 2645, 2650), atr=5.0)
        assert vp.candle_count == 10


class TestToDict:
    def test_to_dict(self, vp):
        for i in range(30):
            r = vp.update(_c(NOW + timedelta(minutes=i), 2655, 2645, 2650), atr=5.0)
        d = r.to_dict()
        assert "poc" in d
        assert "hvn_levels" in d
        assert "current_level" in d
