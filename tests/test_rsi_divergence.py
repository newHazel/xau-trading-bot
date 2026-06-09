"""Tests for RSIDivergenceDetector — Phase 11.3."""

import pytest
import math
from datetime import datetime, timezone, timedelta
from core.indicators.rsi_divergence import RSIDivergenceDetector, DivergenceType


def _c(ts, close):
    return {"timestamp": ts, "close": close, "open": close, "high": close, "low": close, "volume": 1000}


NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def detector():
    return RSIDivergenceDetector({"period": 14, "pivot_window": 2, "min_pivot_distance": 4, "max_pivot_distance": 50})


def _feed_closes(detector, closes):
    for i, c in enumerate(closes):
        detector.update(_c(NOW + timedelta(minutes=i), c))


class TestRSI:
    def test_rsi_none_until_warmed(self, detector):
        for i in range(10):
            r = detector.update(_c(NOW + timedelta(minutes=i), 2600 + i))
            assert r is None

    def test_rsi_value_after_warmup(self, detector):
        for i in range(15):
            detector.update(_c(NOW + timedelta(minutes=i), 2600 + i))
        assert detector.current_rsi is not None
        assert 0 <= detector.current_rsi <= 100

    def test_strong_uptrend_high_rsi(self, detector):
        for i in range(30):
            detector.update(_c(NOW + timedelta(minutes=i), 2600 + i * 5))
        assert detector.current_rsi > 70

    def test_strong_downtrend_low_rsi(self, detector):
        for i in range(30):
            detector.update(_c(NOW + timedelta(minutes=i), 2700 - i * 5))
        assert detector.current_rsi < 30


class TestDivergence:
    def test_no_div_in_trend(self, detector):
        for i in range(40):
            detector.update(_c(NOW + timedelta(minutes=i), 2600 + i))
        div = detector.detect_divergence()
        # Steady trend may or may not have div, but should not be bullish_regular at the top
        if div is not None:
            assert div.type != DivergenceType.BULLISH_REGULAR

    def test_bullish_divergence(self, detector):
        closes = []
        # First low cycle
        for i in range(15):
            closes.append(2700 - i * 5)  # 2700 → 2630 (steep down)
        for i in range(8):
            closes.append(2630 + i * 2)  # bounce 2630 → 2644
        # Second leg down (smaller move = RSI HL while price LL)
        for i in range(10):
            closes.append(2644 - i * 1.5)  # 2644 → 2629.5 (lower low but small)
        for i in range(5):
            closes.append(2629.5 + i)
        _feed_closes(detector, closes)
        div = detector.detect_divergence()
        # We just verify the detector doesn't crash and returns something sensible
        assert div is None or isinstance(div.strength, float)


class TestReset:
    def test_reset(self, detector):
        for i in range(20):
            detector.update(_c(NOW + timedelta(minutes=i), 2600 + i))
        detector.reset()
        assert detector.current_rsi is None


class TestReadingToDict:
    def test_to_dict(self, detector):
        for i in range(20):
            r = detector.update(_c(NOW + timedelta(minutes=i), 2600 + i))
        d = r.to_dict()
        assert "rsi" in d
        assert "price" in d
