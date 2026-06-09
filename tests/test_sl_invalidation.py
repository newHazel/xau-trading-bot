"""Tests for SLInvalidationChecker — Phase 5.2."""

import pytest
from core.risk.sl_invalidation import SLInvalidationChecker, SLInvalidationResult


@pytest.fixture
def close_checker():
    return SLInvalidationChecker({"sl_require_close_for_invalidation": True})


@pytest.fixture
def wick_checker():
    return SLInvalidationChecker({"sl_require_close_for_invalidation": False})


class TestCloseBasedInvalidation:
    def test_long_close_below_sl(self, close_checker):
        r = close_checker.check("long", sl_price=1990.0, candle_close=1989.0)
        assert r.invalidated
        assert r.method == "close"

    def test_long_close_above_sl(self, close_checker):
        r = close_checker.check("long", sl_price=1990.0, candle_close=1991.0)
        assert not r.invalidated
        assert r.method == "none"

    def test_long_close_equals_sl(self, close_checker):
        r = close_checker.check("long", sl_price=1990.0, candle_close=1990.0)
        assert not r.invalidated

    def test_short_close_above_sl(self, close_checker):
        r = close_checker.check("short", sl_price=2010.0, candle_close=2011.0)
        assert r.invalidated
        assert r.method == "close"

    def test_short_close_below_sl(self, close_checker):
        r = close_checker.check("short", sl_price=2010.0, candle_close=2009.0)
        assert not r.invalidated

    def test_short_close_equals_sl(self, close_checker):
        r = close_checker.check("short", sl_price=2010.0, candle_close=2010.0)
        assert not r.invalidated


class TestWickBasedInvalidation:
    def test_long_wick_below_sl(self, wick_checker):
        r = wick_checker.check("long", sl_price=1990.0, candle_close=1992.0, candle_low=1989.0)
        assert r.invalidated
        assert r.method == "wick"

    def test_long_wick_above_sl(self, wick_checker):
        r = wick_checker.check("long", sl_price=1990.0, candle_close=1992.0, candle_low=1991.0)
        assert not r.invalidated

    def test_short_wick_above_sl(self, wick_checker):
        r = wick_checker.check("short", sl_price=2010.0, candle_close=2008.0, candle_high=2011.0)
        assert r.invalidated
        assert r.method == "wick"

    def test_short_wick_below_sl(self, wick_checker):
        r = wick_checker.check("short", sl_price=2010.0, candle_close=2008.0, candle_high=2009.0)
        assert not r.invalidated


class TestDefaults:
    def test_default_requires_close(self):
        c = SLInvalidationChecker()
        r = c.check("long", sl_price=1990.0, candle_close=1989.0)
        assert r.invalidated
        assert r.method == "close"

    def test_none_config_defaults(self):
        c = SLInvalidationChecker(None)
        r = c.check("long", sl_price=1990.0, candle_close=1991.0)
        assert not r.invalidated


class TestToDict:
    def test_result_to_dict(self, close_checker):
        r = close_checker.check("long", sl_price=1990.0, candle_close=1989.0)
        d = r.to_dict()
        assert d["invalidated"] is True
        assert d["method"] == "close"
        assert "detail" in d


class TestEdgeCases:
    def test_direction_whitespace(self, close_checker):
        r = close_checker.check("  SHORT  ", sl_price=2010.0, candle_close=2011.0)
        assert r.invalidated

    def test_wick_mode_no_low_provided(self, wick_checker):
        r = wick_checker.check("long", sl_price=1990.0, candle_close=1992.0)
        assert not r.invalidated
