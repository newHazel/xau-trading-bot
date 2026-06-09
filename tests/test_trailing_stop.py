"""Tests for TrailingStopManager — Phase 5.8."""

import pytest
from core.risk.trailing_stop import TrailingStopManager, TrailingResult, TrailingPhase


@pytest.fixture
def default_config():
    return {
        "tp1_r": 2.0,
        "trailing_start_r": 3.0,
        "sl_buffer_atr_ratio": 0.20,
        "atr_trailing_multiplier": 1.5,
    }


@pytest.fixture
def mgr(default_config):
    return TrailingStopManager(default_config)


class TestInitialPhase:
    def test_below_tp1_no_move(self, mgr):
        r = mgr.update("long", entry=2000.0, current_sl=1995.0, current_price=2005.0,
                        sl_distance=5.0, atr=5.0)
        assert r.phase == TrailingPhase.INITIAL
        assert not r.moved
        assert r.new_sl == 1995.0

    def test_short_below_tp1_no_move(self, mgr):
        r = mgr.update("short", entry=2000.0, current_sl=2005.0, current_price=1997.0,
                        sl_distance=5.0, atr=5.0)
        assert r.phase == TrailingPhase.INITIAL
        assert not r.moved


class TestBreakevenPhase:
    def test_long_at_tp1_moves_to_be(self, mgr):
        r = mgr.update("long", entry=2000.0, current_sl=1995.0, current_price=2012.0,
                        sl_distance=5.0, atr=5.0, costs=0.35)
        assert r.phase == TrailingPhase.BREAKEVEN
        assert r.moved
        assert r.new_sl == 2000.35  # entry + costs

    def test_long_be_already_set(self, mgr):
        r = mgr.update("long", entry=2000.0, current_sl=2001.0, current_price=2012.0,
                        sl_distance=5.0, atr=5.0, costs=0.35)
        assert r.phase == TrailingPhase.BREAKEVEN
        assert not r.moved

    def test_short_at_tp1_moves_to_be(self, mgr):
        r = mgr.update("short", entry=2000.0, current_sl=2005.0, current_price=1988.0,
                        sl_distance=5.0, atr=5.0, costs=0.35)
        assert r.phase == TrailingPhase.BREAKEVEN
        assert r.moved
        assert r.new_sl == 1999.65  # entry - costs

    def test_short_be_already_set(self, mgr):
        r = mgr.update("short", entry=2000.0, current_sl=1999.0, current_price=1988.0,
                        sl_distance=5.0, atr=5.0, costs=0.35)
        assert r.phase == TrailingPhase.BREAKEVEN
        assert not r.moved


class TestTrailingPhase:
    def test_long_trailing_with_structure(self, mgr):
        r = mgr.update("long", entry=2000.0, current_sl=2000.35, current_price=2018.0,
                        sl_distance=5.0, atr=5.0, costs=0.35, swing_level=2012.0)
        assert r.phase == TrailingPhase.TRAILING
        # trail = swing_level - buffer = 2012 - (5*0.2) = 2011.0
        assert r.new_sl >= 2000.35  # at least BE

    def test_long_trailing_with_atr(self, mgr):
        r = mgr.update("long", entry=2000.0, current_sl=2000.35, current_price=2018.0,
                        sl_distance=5.0, atr=5.0, costs=0.35)
        assert r.phase == TrailingPhase.TRAILING
        # trail = price - atr*1.5 = 2018 - 7.5 = 2010.5
        assert r.new_sl >= 2000.35

    def test_trailing_never_moves_backward_long(self, mgr):
        r = mgr.update("long", entry=2000.0, current_sl=2015.0, current_price=2018.0,
                        sl_distance=5.0, atr=5.0, costs=0.35, swing_level=2012.0)
        assert r.new_sl >= 2015.0

    def test_short_trailing_with_atr(self, mgr):
        r = mgr.update("short", entry=2000.0, current_sl=1999.65, current_price=1982.0,
                        sl_distance=5.0, atr=5.0, costs=0.35)
        assert r.phase == TrailingPhase.TRAILING
        # trail = price + atr*1.5 = 1982 + 7.5 = 1989.5
        assert r.new_sl <= 1999.65

    def test_short_trailing_never_moves_upward(self, mgr):
        r = mgr.update("short", entry=2000.0, current_sl=1985.0, current_price=1982.0,
                        sl_distance=5.0, atr=5.0, costs=0.35)
        assert r.new_sl <= 1985.0


class TestDetailAndToDict:
    def test_detail_mentions_structure(self, mgr):
        r = mgr.update("long", entry=2000.0, current_sl=2000.0, current_price=2018.0,
                        sl_distance=5.0, atr=5.0, swing_level=2012.0)
        assert "structure" in r.detail.lower()

    def test_detail_mentions_atr(self, mgr):
        r = mgr.update("long", entry=2000.0, current_sl=2000.0, current_price=2018.0,
                        sl_distance=5.0, atr=5.0)
        assert "atr" in r.detail.lower()

    def test_to_dict(self, mgr):
        r = mgr.update("long", entry=2000.0, current_sl=1995.0, current_price=2005.0,
                        sl_distance=5.0, atr=5.0)
        d = r.to_dict()
        assert "phase" in d
        assert "new_sl" in d
        assert "moved" in d


class TestEdgeCases:
    def test_zero_sl_distance(self, mgr):
        r = mgr.update("long", entry=2000.0, current_sl=1995.0, current_price=2005.0,
                        sl_distance=0, atr=5.0)
        assert r.phase == TrailingPhase.INITIAL

    def test_direction_whitespace(self, mgr):
        r = mgr.update("  LONG  ", entry=2000.0, current_sl=1995.0, current_price=2005.0,
                        sl_distance=5.0, atr=5.0)
        assert r.phase == TrailingPhase.INITIAL
