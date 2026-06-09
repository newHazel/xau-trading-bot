"""Tests for EMACalculator — Phase 11.2."""

import pytest
from datetime import datetime, timezone, timedelta
from core.indicators.ema import EMACalculator, CrossoverType


def _c(ts, close):
    return {"timestamp": ts, "close": close, "open": close, "high": close, "low": close, "volume": 1000}


NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def ema():
    return EMACalculator({"fast_period": 5, "slow_period": 20})


class TestBasic:
    def test_first_candle_initializes(self, ema):
        r = ema.update(_c(NOW, 2650))
        assert r.ema_fast == 2650
        assert r.ema_slow == 2650

    def test_ema_converges(self, ema):
        for i in range(50):
            ema.update(_c(NOW + timedelta(minutes=i), 2700))
        assert abs(ema.ema_fast - 2700) < 1
        assert abs(ema.ema_slow - 2700) < 10


class TestBias:
    def test_long_bias(self, ema):
        for i in range(30):
            ema.update(_c(NOW + timedelta(minutes=i), 2600))
        # ramp up sharply
        for i in range(30, 80):
            ema.update(_c(NOW + timedelta(minutes=i), 2750))
        r = ema.update(_c(NOW + timedelta(minutes=80), 2750))
        assert r.bias == "long"

    def test_short_bias(self, ema):
        for i in range(30):
            ema.update(_c(NOW + timedelta(minutes=i), 2700))
        for i in range(30, 80):
            ema.update(_c(NOW + timedelta(minutes=i), 2550))
        r = ema.update(_c(NOW + timedelta(minutes=80), 2550))
        assert r.bias == "short"


class TestCrossover:
    def test_golden_cross(self, ema):
        for i in range(30):
            ema.update(_c(NOW + timedelta(minutes=i), 2600))
        for i in range(30, 80):
            ema.update(_c(NOW + timedelta(minutes=i), 2750))
        assert ema.last_crossover is not None
        assert ema.last_crossover.type == CrossoverType.GOLDEN

    def test_death_cross(self, ema):
        # establish fast > slow with an uptrend first
        for i in range(60):
            ema.update(_c(NOW + timedelta(minutes=i), 2600 + i * 2))
        # then sharp drop to flip fast below slow
        for i in range(60, 140):
            ema.update(_c(NOW + timedelta(minutes=i), 2500))
        assert ema.last_crossover is not None
        assert ema.last_crossover.type == CrossoverType.DEATH


class TestPermissions:
    def test_long_allowed_above_slow(self, ema):
        for i in range(40):
            ema.update(_c(NOW + timedelta(minutes=i), 2600))
        assert ema.is_long_allowed(2650)
        assert not ema.is_short_allowed(2650)

    def test_short_allowed_below_slow(self, ema):
        for i in range(40):
            ema.update(_c(NOW + timedelta(minutes=i), 2700))
        assert ema.is_short_allowed(2650)
        assert not ema.is_long_allowed(2650)


class TestReset:
    def test_reset(self, ema):
        ema.update(_c(NOW, 2650))
        ema.reset()
        assert ema.ema_fast is None
        assert ema.ema_slow is None
