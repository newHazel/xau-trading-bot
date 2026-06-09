"""Tests for FillEngine — Phase 6.2."""

import pytest
from backtesting.fill_engine import FillEngine, FillType, OpenPosition


def _long_position(**overrides) -> OpenPosition:
    defaults = dict(
        direction="long", entry_price=2000.0, sl_price=1995.0,
        tp1_price=2010.0, tp2_price=2017.5, lot_size=0.10,
        remaining_lots=0.10, entry_bar=0, setup_id="BT-0001",
        sl_distance=5.0, costs_per_lot=0.35,
    )
    defaults.update(overrides)
    return OpenPosition(**defaults)


def _short_position(**overrides) -> OpenPosition:
    defaults = dict(
        direction="short", entry_price=2000.0, sl_price=2005.0,
        tp1_price=1990.0, tp2_price=1982.5, lot_size=0.10,
        remaining_lots=0.10, entry_bar=0, setup_id="BT-0002",
        sl_distance=5.0, costs_per_lot=0.35,
    )
    defaults.update(overrides)
    return OpenPosition(**defaults)


@pytest.fixture
def engine():
    return FillEngine({"conservative_backtest": True, "default_spread": 0.25, "default_slippage": 0.10})


class TestConservativeFills:
    def test_sl_and_tp_same_candle_fills_sl(self, engine):
        pos = _long_position()
        fills = engine.check_fills(pos, bar_high=2015.0, bar_low=1993.0, bar_close=2005.0, bar_index=5)
        assert len(fills) == 1
        assert fills[0].fill_type == FillType.SL_HIT

    def test_sl_and_tp_same_candle_short(self, engine):
        pos = _short_position()
        fills = engine.check_fills(pos, bar_high=2006.0, bar_low=1988.0, bar_close=1995.0, bar_index=5)
        assert len(fills) == 1
        assert fills[0].fill_type == FillType.SL_HIT


class TestSLHit:
    def test_long_sl_hit(self, engine):
        pos = _long_position()
        fills = engine.check_fills(pos, bar_high=2002.0, bar_low=1994.0, bar_close=1996.0, bar_index=3)
        assert len(fills) == 1
        assert fills[0].fill_type == FillType.SL_HIT
        assert fills[0].r_multiple < 0
        assert not pos.is_open

    def test_short_sl_hit(self, engine):
        pos = _short_position()
        fills = engine.check_fills(pos, bar_high=2006.0, bar_low=2001.0, bar_close=2004.0, bar_index=3)
        assert len(fills) == 1
        assert fills[0].fill_type == FillType.SL_HIT


class TestTP1Hit:
    def test_long_tp1_partial_close(self, engine):
        pos = _long_position()
        fills = engine.check_fills(pos, bar_high=2011.0, bar_low=1998.0, bar_close=2009.0, bar_index=4)
        assert len(fills) == 1
        assert fills[0].fill_type == FillType.TP1_HIT
        assert pos.tp1_hit
        assert pos.is_open  # still open with remaining lots

    def test_short_tp1_partial_close(self, engine):
        pos = _short_position()
        fills = engine.check_fills(pos, bar_high=1998.0, bar_low=1989.0, bar_close=1991.0, bar_index=4)
        assert len(fills) == 1
        assert fills[0].fill_type == FillType.TP1_HIT
        assert pos.tp1_hit


class TestTP2Hit:
    def test_long_tp2_after_tp1(self, engine):
        pos = _long_position(tp1_hit=True, remaining_lots=0.05)
        fills = engine.check_fills(pos, bar_high=2018.0, bar_low=2008.0, bar_close=2017.0, bar_index=8)
        assert len(fills) == 1
        assert fills[0].fill_type == FillType.TP2_HIT
        assert not pos.is_open

    def test_short_tp2_after_tp1(self, engine):
        pos = _short_position(tp1_hit=True, remaining_lots=0.05)
        fills = engine.check_fills(pos, bar_high=1988.0, bar_low=1981.0, bar_close=1983.0, bar_index=8)
        assert len(fills) == 1
        assert fills[0].fill_type == FillType.TP2_HIT


class TestNoFill:
    def test_no_fill_when_price_between_sl_tp(self, engine):
        pos = _long_position()
        fills = engine.check_fills(pos, bar_high=2005.0, bar_low=1997.0, bar_close=2003.0, bar_index=2)
        assert len(fills) == 0
        assert pos.is_open

    def test_closed_position_no_fill(self, engine):
        pos = _long_position(remaining_lots=0)
        fills = engine.check_fills(pos, bar_high=2020.0, bar_low=1990.0, bar_close=2000.0, bar_index=5)
        assert len(fills) == 0


class TestIntrabar:
    def test_intrabar_sl_before_tp(self, engine):
        pos = _long_position()
        candles = [
            {"high": 2002.0, "low": 1998.0},
            {"high": 2001.0, "low": 1994.0},  # SL hit here
            {"high": 2012.0, "low": 2008.0},  # TP would hit but SL already filled
        ]
        fills = engine.check_fills(pos, 2012.0, 1994.0, 2008.0, 5, intrabar_candles=candles)
        assert len(fills) == 1
        assert fills[0].fill_type == FillType.SL_HIT

    def test_intrabar_tp_before_sl(self, engine):
        pos = _long_position()
        candles = [
            {"high": 2011.0, "low": 1998.0},  # TP1 hit first
            {"high": 2005.0, "low": 1994.0},  # SL would hit
        ]
        fills = engine.check_fills(pos, 2011.0, 1994.0, 2000.0, 5, intrabar_candles=candles)
        assert fills[0].fill_type == FillType.TP1_HIT


class TestFillToDict:
    def test_fill_result_to_dict(self, engine):
        pos = _long_position()
        fills = engine.check_fills(pos, bar_high=2002.0, bar_low=1993.0, bar_close=1996.0, bar_index=3)
        d = fills[0].to_dict()
        assert "fill_type" in d
        assert "fill_price" in d
        assert "net_pnl" in d


class TestNonConservative:
    def test_non_conservative_tp_wins(self):
        eng = FillEngine({"conservative_backtest": False})
        pos = _long_position()
        fills = eng.check_fills(pos, bar_high=2011.0, bar_low=1994.0, bar_close=2008.0, bar_index=5)
        assert fills[0].fill_type == FillType.SL_HIT
