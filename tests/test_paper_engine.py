"""Tests for PaperEngine — Phase 7.2."""

import pytest
from datetime import datetime, timezone
from paper_trading.paper_engine import PaperEngine, PaperSignal, PaperTradeResult


def _make_signal(**overrides) -> PaperSignal:
    defaults = dict(
        setup_id="XAU-20260121-1030-LONG-FVG01",
        direction="long",
        entry_price=2000.0,
        sl_price=1995.0,
        tp1_price=2010.0,
        tp2_price=2017.5,
        grade="A+",
        timestamp=datetime(2026, 1, 21, 10, 30, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return PaperSignal(**defaults)


@pytest.fixture
def engine():
    return PaperEngine({
        "allowed_grades": ["A+", "A"],
        "max_daily_trades": 3,
        "max_daily_losses": 2,
        "stop_after_tp": True,
        "default_spread": 0.25,
        "default_slippage": 0.10,
    })


class TestAcceptSignal:
    def test_accept_valid_signal(self, engine):
        sig = _make_signal()
        r = engine.accept_signal(sig)
        assert r["accepted"]
        assert engine.has_open_position

    def test_reject_bad_grade(self, engine):
        sig = _make_signal(grade="B")
        r = engine.accept_signal(sig)
        assert not r["accepted"]
        assert "grade" in r["reason"]

    def test_reject_when_position_open(self, engine):
        engine.accept_signal(_make_signal())
        r = engine.accept_signal(_make_signal(setup_id="XAU-002"))
        assert not r["accepted"]
        assert "position already open" in r["reason"]

    def test_reject_daily_trade_limit(self):
        eng = PaperEngine({
            "allowed_grades": ["A+", "A"],
            "max_daily_trades": 3,
            "max_daily_losses": 10,  # high so trade limit triggers first
            "stop_after_tp": False,
            "default_spread": 0.25,
            "default_slippage": 0.10,
        })
        ts = datetime(2026, 1, 21, 10, 30, tzinfo=timezone.utc)
        for i in range(3):
            sig = _make_signal(setup_id=f"XAU-{i:03d}", timestamp=ts)
            eng.accept_signal(sig)
            eng.update_price(1993.0, 1993.0, 1994.0, ts)  # SL hit
        sig4 = _make_signal(setup_id="XAU-003", timestamp=ts)
        r = eng.accept_signal(sig4)
        assert not r["accepted"]
        assert "daily trade limit" in r["reason"]


class TestSLHit:
    def test_long_sl_hit(self, engine):
        engine.accept_signal(_make_signal())
        ts = datetime(2026, 1, 21, 11, 0, tzinfo=timezone.utc)
        result = engine.update_price(high=2002.0, low=1994.0, close=1996.0, timestamp=ts)
        assert result is not None
        assert result.exit_type == "sl_hit"
        assert result.gross_r < 0
        assert not engine.has_open_position

    def test_short_sl_hit(self, engine):
        sig = _make_signal(direction="short", sl_price=2005.0, tp1_price=1990.0, tp2_price=1982.5)
        engine.accept_signal(sig)
        ts = datetime(2026, 1, 21, 11, 0, tzinfo=timezone.utc)
        result = engine.update_price(high=2006.0, low=1998.0, close=2004.0, timestamp=ts)
        assert result is not None
        assert result.exit_type == "sl_hit"


class TestTP1AndTP2:
    def test_tp1_moves_sl_to_be(self, engine):
        engine.accept_signal(_make_signal())
        ts = datetime(2026, 1, 21, 11, 0, tzinfo=timezone.utc)
        result = engine.update_price(high=2011.0, low=1998.0, close=2009.0, timestamp=ts)
        assert result is None  # TP1 hit but not closed yet
        assert engine.has_open_position
        assert engine._position.tp1_hit

    def test_tp2_after_tp1(self, engine):
        engine.accept_signal(_make_signal())
        ts1 = datetime(2026, 1, 21, 11, 0, tzinfo=timezone.utc)
        engine.update_price(high=2011.0, low=1998.0, close=2009.0, timestamp=ts1)

        ts2 = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)
        result = engine.update_price(high=2018.0, low=2008.0, close=2017.0, timestamp=ts2)
        assert result is not None
        assert result.exit_type == "tp2_hit"
        assert result.gross_r > 0


class TestConservativeFill:
    def test_sl_wins_when_both_hit(self, engine):
        engine.accept_signal(_make_signal())
        ts = datetime(2026, 1, 21, 11, 0, tzinfo=timezone.utc)
        result = engine.update_price(high=2011.0, low=1994.0, close=2005.0, timestamp=ts)
        assert result is not None
        assert result.exit_type == "sl_hit"


class TestStopAfterTP:
    def test_stop_after_first_win(self, engine):
        sig1 = _make_signal(setup_id="XAU-001")
        engine.accept_signal(sig1)
        ts1 = datetime(2026, 1, 21, 11, 0, tzinfo=timezone.utc)
        engine.update_price(high=2011.0, low=1998.0, close=2009.0, timestamp=ts1)
        ts2 = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)
        engine.update_price(high=2018.0, low=2008.0, close=2017.0, timestamp=ts2)  # TP2 win

        sig2 = _make_signal(setup_id="XAU-002")
        r = engine.accept_signal(sig2)
        assert not r["accepted"]
        assert "stop after TP" in r["reason"]


class TestNoUpdate:
    def test_no_position_returns_none(self, engine):
        ts = datetime(2026, 1, 21, 11, 0, tzinfo=timezone.utc)
        result = engine.update_price(high=2005.0, low=1998.0, close=2003.0, timestamp=ts)
        assert result is None

    def test_price_between_sl_tp_no_fill(self, engine):
        engine.accept_signal(_make_signal())
        ts = datetime(2026, 1, 21, 11, 0, tzinfo=timezone.utc)
        result = engine.update_price(high=2005.0, low=1997.0, close=2003.0, timestamp=ts)
        assert result is None


class TestReset:
    def test_reset_clears_all(self, engine):
        engine.accept_signal(_make_signal())
        engine.reset()
        assert not engine.has_open_position
        assert len(engine.results) == 0


class TestToDict:
    def test_signal_to_dict(self):
        sig = _make_signal()
        d = sig.to_dict()
        assert d["setup_id"] == "XAU-20260121-1030-LONG-FVG01"
        assert d["grade"] == "A+"

    def test_result_to_dict(self, engine):
        engine.accept_signal(_make_signal())
        ts = datetime(2026, 1, 21, 11, 0, tzinfo=timezone.utc)
        result = engine.update_price(high=2002.0, low=1994.0, close=1996.0, timestamp=ts)
        d = result.to_dict()
        assert "net_r" in d
        assert "exit_type" in d
