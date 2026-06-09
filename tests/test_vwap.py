"""Tests for SessionalVWAP — Phase 11.1."""

import pytest
from datetime import datetime, timezone, timedelta
from core.indicators.vwap import SessionalVWAP, VWAPBias


def _c(ts, o, h, l, c, v=1000):
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


@pytest.fixture
def vwap():
    return SessionalVWAP()


# Israel time 10:00 = UTC 08:00 (winter) — use UTC for predictable test data
LONDON_OPEN_UTC = datetime(2026, 1, 21, 8, 0, tzinfo=timezone.utc)


class TestBasic:
    def test_first_candle(self, vwap):
        r = vwap.update(_c(LONDON_OPEN_UTC, 2650, 2655, 2645, 2650), atr=5.0)
        assert r.vwap == 2650.0
        assert r.bias == VWAPBias.NEUTRAL

    def test_above_bias(self, vwap):
        vwap.update(_c(LONDON_OPEN_UTC, 2650, 2655, 2645, 2650), atr=5.0)
        r = vwap.update(_c(LONDON_OPEN_UTC + timedelta(minutes=5), 2650, 2670, 2650, 2665), atr=5.0)
        assert r.bias == VWAPBias.ABOVE
        assert r.price > r.vwap

    def test_below_bias(self, vwap):
        vwap.update(_c(LONDON_OPEN_UTC, 2650, 2655, 2645, 2650), atr=5.0)
        r = vwap.update(_c(LONDON_OPEN_UTC + timedelta(minutes=5), 2650, 2650, 2630, 2635), atr=5.0)
        assert r.bias == VWAPBias.BELOW


class TestSessionReset:
    def test_resets_at_new_session(self, vwap):
        london = LONDON_OPEN_UTC
        vwap.update(_c(london, 2650, 2660, 2640, 2655, v=10000), atr=5.0)
        vwap.update(_c(london + timedelta(hours=1), 2655, 2700, 2655, 2695, v=10000), atr=5.0)
        vwap_london = vwap.current_vwap

        ny = datetime(2026, 1, 21, 14, 30, tzinfo=timezone.utc)
        r_ny = vwap.update(_c(ny, 2695, 2700, 2690, 2693, v=10000), atr=5.0)
        assert r_ny.vwap != vwap_london
        # New session — VWAP reflects only the NY candle's typical price
        expected = (2700 + 2690 + 2693) / 3
        assert abs(r_ny.vwap - expected) < 0.01


class TestDistanceATR:
    def test_distance_positive(self, vwap):
        vwap.update(_c(LONDON_OPEN_UTC, 2650, 2655, 2645, 2650), atr=5.0)
        r = vwap.update(_c(LONDON_OPEN_UTC + timedelta(minutes=5), 2650, 2680, 2650, 2680), atr=5.0)
        assert r.distance_atr > 0

    def test_distance_negative(self, vwap):
        vwap.update(_c(LONDON_OPEN_UTC, 2650, 2655, 2645, 2650), atr=5.0)
        r = vwap.update(_c(LONDON_OPEN_UTC + timedelta(minutes=5), 2650, 2650, 2620, 2620), atr=5.0)
        assert r.distance_atr < 0


class TestReset:
    def test_explicit_reset(self, vwap):
        vwap.update(_c(LONDON_OPEN_UTC, 2650, 2660, 2640, 2655), atr=5.0)
        vwap.reset()
        assert vwap.current_vwap is None
        assert vwap.current_session is None


class TestToDict:
    def test_to_dict(self, vwap):
        r = vwap.update(_c(LONDON_OPEN_UTC, 2650, 2655, 2645, 2650), atr=5.0)
        d = r.to_dict()
        assert "vwap" in d
        assert d["bias"] == "neutral"
