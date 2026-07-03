"""Entry-bar fill evaluation — a bar that traverses the limit entry AND the stop
must book the SL loss on that same bar (falling-knife case). Same-bar TP stays
un-granted: the TP print may predate the limit fill, so the conservative engine
never awards it.
"""

import pandas as pd
from datetime import datetime, timezone

from backtesting.backtest_runner import BacktestRunner, BacktestConfig
from backtesting.fill_engine import FillEngine, OpenPosition


def _df(rows):
    """rows: list of (open, high, low, close)."""
    idx = pd.date_range(datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc),
                        periods=len(rows), freq="5min")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [100.0] * len(rows)},
        index=idx)


def _cfg():
    return BacktestConfig(initial_balance=10000.0, conservative_fills=True,
                          costs_inclusive=True, max_daily_trades=999, max_daily_losses=999)


def _long_signal(bar_index=1, entry=2000.0, sl=1995.0, tp1=2010.0, tp2=2017.5):
    return {"bar_index": bar_index, "direction": "long", "entry": entry, "sl": sl,
            "tp1": tp1, "tp2": tp2, "lot_size": 0.10}


class TestEntryBarSL:
    def test_falling_knife_books_sl_on_entry_bar(self):
        """Bar 2 sweeps through the limit (2000) AND the stop (1995) → SL loss ON bar 2,
        even though price then recovers (the pre-fix runner survived to book a win)."""
        df = _df([
            (2005, 2006, 2004, 2005),   # 0: quiet
            (2005, 2006, 2004, 2005),   # 1: signal bar (armed, no fill)
            (2004, 2005, 1990, 1992),   # 2: knife through entry AND SL
            (1992, 2012, 1991, 2011),   # 3: full recovery through TP1
            (2011, 2020, 2010, 2018),   # 4: and beyond
        ])
        res = BacktestRunner(_cfg()).run(df, signals=[_long_signal()])
        assert len(res.trades) == 1
        t = res.trades[0]
        assert t.exit_type == "sl_hit"
        assert t.bar_exit == 2
        assert t.r_multiple < 0

    def test_no_same_bar_tp_granted(self):
        """Bar 2 brackets entry AND TP1 but not SL → no same-bar win; TP1 may only
        fill from bar 3 onward (here bar 3 hits TP1 then bar 4 hits TP2)."""
        df = _df([
            (2005, 2006, 2004, 2005),   # 0
            (2005, 2006, 2004, 2005),   # 1: signal bar
            (2005, 2011, 1999, 2009),   # 2: entry + TP1 in range — must NOT book TP1 here
            (2009, 2012, 2008, 2011),   # 3: TP1
            (2011, 2020, 2010, 2019),   # 4: TP2
        ])
        res = BacktestRunner(_cfg()).run(df, signals=[_long_signal()])
        assert len(res.trades) == 1
        t = res.trades[0]
        assert t.exit_type == "tp2_hit"
        assert t.bar_exit == 4          # TP2 on bar 4 — nothing terminal on bar 2
        assert t.r_multiple > 0

    def test_short_entry_bar_sl(self):
        """Mirror case for shorts: bar sweeps up through entry AND stop."""
        df = _df([
            (1995, 1996, 1994, 1995),   # 0
            (1995, 1996, 1994, 1995),   # 1: signal bar
            (1996, 2007, 1995, 2006),   # 2: spike through entry (2000) and SL (2005)
            (2006, 2007, 1980, 1985),   # 3: would have been a big winner
        ])
        sig = {"bar_index": 1, "direction": "short", "entry": 2000.0, "sl": 2005.0,
               "tp1": 1990.0, "tp2": 1982.5, "lot_size": 0.10}
        res = BacktestRunner(_cfg()).run(df, signals=[sig])
        assert len(res.trades) == 1
        assert res.trades[0].exit_type == "sl_hit"
        assert res.trades[0].bar_exit == 2

    def test_clean_entry_bar_leaves_position_open(self):
        """Entry fills, neither SL nor TP touched on the entry bar → position remains
        open and resolves later exactly as before the fix."""
        df = _df([
            (2005, 2006, 2004, 2005),   # 0
            (2005, 2006, 2004, 2005),   # 1: signal bar
            (2004, 2005, 1999, 2001),   # 2: entry only
            (2001, 2002, 1993, 1994),   # 3: SL
        ])
        res = BacktestRunner(_cfg()).run(df, signals=[_long_signal()])
        assert len(res.trades) == 1
        assert res.trades[0].exit_type == "sl_hit"
        assert res.trades[0].bar_exit == 3


class TestFillEngineEntryBarAPI:
    def test_sl_only_no_tp(self):
        fe = FillEngine({"default_spread": 0.0, "default_slippage": 0.0})
        pos = OpenPosition(direction="long", entry_price=2000.0, sl_price=1995.0,
                           tp1_price=2010.0, tp2_price=2017.5, lot_size=0.1,
                           remaining_lots=0.1, entry_bar=5, setup_id="t",
                           sl_distance=5.0)
        # TP-side range only → nothing
        assert fe.check_entry_bar_fills(pos, 2012.0, 1999.0, 5) == []
        assert pos.is_open
        # SL-side touch → SL fill
        fills = fe.check_entry_bar_fills(pos, 2001.0, 1994.0, 5)
        assert len(fills) == 1 and fills[0].fill_type.value == "sl_hit"
        assert not pos.is_open
