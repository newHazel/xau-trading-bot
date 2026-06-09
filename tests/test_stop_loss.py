"""Tests for StopLossCalculator — Phase 5.1."""

import pytest
from core.risk.stop_loss import StopLossCalculator, StopLossResult


@pytest.fixture
def default_config():
    return {
        "sl_invalidation_mode": "min_of_sweep_and_fvg",
        "sl_buffer_atr_ratio": 0.20,
        "atr_sl_multiplier": 1.5,
    }


@pytest.fixture
def calc(default_config):
    return StopLossCalculator(default_config)


class TestLongSL:
    def test_sweep_low_with_buffer(self, calc):
        r = calc.calculate("long", entry=2000.0, atr=5.0, sweep_low=1995.0)
        assert r.valid
        assert r.sl_price == 1995.0 - 5.0 * 0.20  # 1994.0
        assert r.sl_distance == 2000.0 - r.sl_price

    def test_fvg_bottom_with_buffer(self, calc):
        r = calc.calculate("long", entry=2000.0, atr=5.0, fvg_bottom=1996.0)
        assert r.valid
        assert r.sl_price == 1996.0 - 1.0  # buffer = 5*0.2 = 1.0
        assert r.structural_level == 1996.0

    def test_min_of_sweep_and_fvg_picks_tighter(self, calc):
        r = calc.calculate("long", entry=2000.0, atr=5.0, sweep_low=1994.0, fvg_bottom=1996.0)
        assert r.valid
        assert r.structural_level == 1994.0

    def test_swing_low_used(self):
        c = StopLossCalculator({"sl_invalidation_mode": "swing_low", "sl_buffer_atr_ratio": 0.20, "atr_sl_multiplier": 1.5})
        r = c.calculate("long", entry=2000.0, atr=5.0, swing_low=1997.0)
        assert r.valid
        assert r.structural_level == 1997.0

    def test_no_structural_level(self, calc):
        r = calc.calculate("long", entry=2000.0, atr=5.0)
        assert not r.valid
        assert "no structural level" in r.rejection_reason

    def test_sl_too_wide(self, calc):
        r = calc.calculate("long", entry=2000.0, atr=5.0, sweep_low=1990.0)
        assert not r.valid
        assert "too wide" in r.rejection_reason
        assert r.sl_price > 0

    def test_sl_beyond_entry(self, calc):
        r = calc.calculate("long", entry=2000.0, atr=5.0, sweep_low=2002.0)
        assert not r.valid
        assert "beyond entry" in r.rejection_reason


class TestShortSL:
    def test_sweep_high_with_buffer(self, calc):
        r = calc.calculate("short", entry=2000.0, atr=5.0, sweep_high=2005.0)
        assert r.valid
        buffer = 5.0 * 0.20
        assert r.sl_price == 2005.0 + buffer

    def test_fvg_top_with_buffer(self, calc):
        r = calc.calculate("short", entry=2000.0, atr=5.0, fvg_top=2004.0)
        assert r.valid
        assert r.structural_level == 2004.0

    def test_no_structural_short(self, calc):
        r = calc.calculate("short", entry=2000.0, atr=5.0)
        assert not r.valid

    def test_sl_too_wide_short(self, calc):
        r = calc.calculate("short", entry=2000.0, atr=5.0, sweep_high=2010.0)
        assert not r.valid
        assert "too wide" in r.rejection_reason


class TestModes:
    def test_sweep_low_mode(self):
        c = StopLossCalculator({"sl_invalidation_mode": "sweep_low", "sl_buffer_atr_ratio": 0.20, "atr_sl_multiplier": 1.5})
        r = c.calculate("long", entry=2000.0, atr=5.0, sweep_low=1996.0, fvg_bottom=1998.0)
        assert r.structural_level == 1996.0

    def test_fvg_bottom_mode(self):
        c = StopLossCalculator({"sl_invalidation_mode": "fvg_bottom", "sl_buffer_atr_ratio": 0.20, "atr_sl_multiplier": 1.5})
        r = c.calculate("long", entry=2000.0, atr=5.0, sweep_low=1996.0, fvg_bottom=1998.0)
        assert r.structural_level == 1998.0

    def test_sweep_high_mode_short(self):
        c = StopLossCalculator({"sl_invalidation_mode": "sweep_high", "sl_buffer_atr_ratio": 0.20, "atr_sl_multiplier": 1.5})
        r = c.calculate("short", entry=2000.0, atr=5.0, sweep_high=2004.0, fvg_top=2002.0)
        assert r.structural_level == 2004.0


class TestToDict:
    def test_result_to_dict(self, calc):
        r = calc.calculate("long", entry=2000.0, atr=5.0, sweep_low=1996.0)
        d = r.to_dict()
        assert "sl_price" in d
        assert "valid" in d
        assert d["valid"] is True


class TestEdgeCases:
    def test_whitespace_direction(self, calc):
        r = calc.calculate("  Long  ", entry=2000.0, atr=5.0, sweep_low=1996.0)
        assert r.valid

    def test_default_config(self):
        c = StopLossCalculator({})
        r = c.calculate("long", entry=2000.0, atr=5.0, sweep_low=1996.0)
        assert r.valid
