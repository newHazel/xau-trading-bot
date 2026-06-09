"""Tests for LiquidityTargetFinder — Phase 5.3."""

import pytest
from core.risk.liquidity_target_finder import (
    LiquidityTargetFinder, LiquidityTargetResult, LiquidityTarget,
)


@pytest.fixture
def default_config():
    return {"require_liquidity_target_for_tp2": True, "tp1_r": 2.0, "tp2_r": 3.5}


@pytest.fixture
def finder(default_config):
    return LiquidityTargetFinder(default_config)


class TestLongTargets:
    def test_finds_targets_above_entry(self, finder):
        levels = [
            {"price": 2010.0, "type": "eqh"},
            {"price": 2020.0, "type": "swing_high"},
            {"price": 1990.0, "type": "eql"},  # below entry, should be filtered
        ]
        r = finder.find("long", entry=2000.0, sl_distance=5.0, levels=levels)
        assert len(r.targets) == 2
        assert all(t.price > 2000.0 for t in r.targets)

    def test_targets_sorted_by_distance(self, finder):
        levels = [
            {"price": 2020.0, "type": "swing_high"},
            {"price": 2010.0, "type": "eqh"},
        ]
        r = finder.find("long", entry=2000.0, sl_distance=5.0, levels=levels)
        assert r.targets[0].price == 2010.0
        assert r.targets[1].price == 2020.0

    def test_r_multiple_calculated(self, finder):
        levels = [{"price": 2010.0, "type": "eqh"}]
        r = finder.find("long", entry=2000.0, sl_distance=5.0, levels=levels)
        assert r.targets[0].r_multiple == 2.0

    def test_tp2_target_found(self, finder):
        levels = [
            {"price": 2012.0, "type": "eqh"},  # 2.4R
            {"price": 2020.0, "type": "swing_high"},  # 4R
        ]
        r = finder.find("long", entry=2000.0, sl_distance=5.0, levels=levels)
        assert r.tp2_target is not None
        assert r.tp2_target.price == 2012.0

    def test_has_target_before_2r(self, finder):
        levels = [
            {"price": 2008.0, "type": "eqh"},  # 1.6R
            {"price": 2015.0, "type": "swing_high"},  # 3R
        ]
        r = finder.find("long", entry=2000.0, sl_distance=5.0, levels=levels)
        assert r.has_target_before_2r

    def test_has_target_between_2r_5r(self, finder):
        levels = [{"price": 2015.0, "type": "eqh"}]  # 3R
        r = finder.find("long", entry=2000.0, sl_distance=5.0, levels=levels)
        assert r.has_target_between_2r_5r


class TestShortTargets:
    def test_finds_targets_below_entry(self, finder):
        levels = [
            {"price": 1990.0, "type": "eql"},
            {"price": 1980.0, "type": "swing_low"},
            {"price": 2010.0, "type": "eqh"},  # above entry, filtered
        ]
        r = finder.find("short", entry=2000.0, sl_distance=5.0, levels=levels)
        assert len(r.targets) == 2
        assert all(t.price < 2000.0 for t in r.targets)

    def test_tp2_target_short(self, finder):
        levels = [
            {"price": 1988.0, "type": "eql"},  # 2.4R
        ]
        r = finder.find("short", entry=2000.0, sl_distance=5.0, levels=levels)
        assert r.tp2_target is not None
        assert r.tp2_target.price == 1988.0


class TestEdgeCases:
    def test_no_levels(self, finder):
        r = finder.find("long", entry=2000.0, sl_distance=5.0, levels=[])
        assert len(r.targets) == 0
        assert r.tp2_target is None

    def test_zero_sl_distance(self, finder):
        r = finder.find("long", entry=2000.0, sl_distance=0, levels=[])
        assert not r.valid
        assert "invalid SL distance" in r.detail

    def test_no_tp2_candidate(self, finder):
        levels = [{"price": 2005.0, "type": "eqh"}]  # 1R, below tp1_r
        r = finder.find("long", entry=2000.0, sl_distance=5.0, levels=levels)
        assert r.tp2_target is None

    def test_direction_whitespace(self, finder):
        levels = [{"price": 2012.0, "type": "eqh"}]
        r = finder.find("  Long  ", entry=2000.0, sl_distance=5.0, levels=levels)
        assert len(r.targets) == 1


class TestToDict:
    def test_result_to_dict_with_tp2(self, finder):
        levels = [{"price": 2015.0, "type": "eqh"}]
        r = finder.find("long", entry=2000.0, sl_distance=5.0, levels=levels)
        d = r.to_dict()
        assert d["tp2_price"] == 2015.0
        assert d["tp2_type"] == "eqh"

    def test_result_to_dict_no_tp2(self, finder):
        r = finder.find("long", entry=2000.0, sl_distance=5.0, levels=[])
        d = r.to_dict()
        assert d["tp2_price"] is None
