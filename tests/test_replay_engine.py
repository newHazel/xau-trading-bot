"""Tests for ReplayEngine — Phase 6.1."""

import pytest
import pandas as pd
from datetime import datetime, timezone

from backtesting.replay_engine import ReplayEngine, ReplayBar, ReplayState


def _make_df(n: int = 10) -> pd.DataFrame:
    base = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    dates = pd.date_range(base, periods=n, freq="1min")
    data = {
        "open": [2000 + i for i in range(n)],
        "high": [2001 + i for i in range(n)],
        "low": [1999 + i for i in range(n)],
        "close": [2000.5 + i for i in range(n)],
        "volume": [100] * n,
    }
    return pd.DataFrame(data, index=dates)


class TestReplayBasic:
    def test_runs_all_bars(self):
        df = _make_df(20)
        engine = ReplayEngine()
        bars_seen = []
        engine.run(df, lambda bar, eng: bars_seen.append(bar))
        assert len(bars_seen) == 20

    def test_state_tracking(self):
        df = _make_df(15)
        engine = ReplayEngine()
        state = engine.run(df, lambda b, e: None)
        assert state.total_bars == 15
        assert state.bars_processed == 15
        assert state.started
        assert state.finished

    def test_bar_index_increments(self):
        df = _make_df(5)
        engine = ReplayEngine()
        indices = []
        engine.run(df, lambda bar, eng: indices.append(bar.bar_index))
        assert indices == [0, 1, 2, 3, 4]

    def test_bar_data_correct(self):
        df = _make_df(3)
        engine = ReplayEngine()
        bars = []
        engine.run(df, lambda bar, eng: bars.append(bar))
        assert bars[0].open == 2000
        assert bars[0].high == 2001
        assert bars[0].low == 1999

    def test_timeframe_default(self):
        df = _make_df(2)
        engine = ReplayEngine()
        bars = []
        engine.run(df, lambda bar, eng: bars.append(bar))
        assert bars[0].timeframe == "1m"

    def test_custom_timeframe(self):
        df = _make_df(2)
        engine = ReplayEngine({"base_timeframe": "5m"})
        bars = []
        engine.run(df, lambda bar, eng: bars.append(bar))
        assert bars[0].timeframe == "5m"


class TestHistory:
    def test_history_available_during_replay(self):
        df = _make_df(10)
        engine = ReplayEngine()
        history_lengths = []
        def on_bar(bar, eng):
            history_lengths.append(len(eng.history))
        engine.run(df, on_bar)
        assert history_lengths == list(range(1, 11))

    def test_history_capped_at_max(self):
        df = _make_df(20)
        engine = ReplayEngine({"max_history_bars": 5})
        engine.run(df, lambda b, e: None)
        assert len(engine.history) == 5

    def test_get_history_df(self):
        df = _make_df(5)
        engine = ReplayEngine()
        engine.run(df, lambda b, e: None)
        hdf = engine.get_history_df()
        assert len(hdf) == 5
        assert "open" in hdf.columns

    def test_get_history_df_last_n(self):
        df = _make_df(10)
        engine = ReplayEngine()
        engine.run(df, lambda b, e: None)
        hdf = engine.get_history_df(last_n=3)
        assert len(hdf) == 3


class TestNoLookAhead:
    def test_callback_only_sees_past(self):
        df = _make_df(10)
        engine = ReplayEngine()
        violations = []
        def on_bar(bar, eng):
            if bar.bar_index < 9:
                h = eng.history
                future_bars = [b for b in h if b.bar_index > bar.bar_index]
                if future_bars:
                    violations.append(bar.bar_index)
        engine.run(df, on_bar)
        assert violations == []


class TestValidation:
    def test_empty_df_raises(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close"])
        engine = ReplayEngine()
        with pytest.raises(ValueError, match="empty"):
            engine.run(df, lambda b, e: None)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"open": [1], "high": [2]})
        engine = ReplayEngine()
        with pytest.raises(ValueError, match="Missing"):
            engine.run(df, lambda b, e: None)


class TestToDict:
    def test_bar_to_dict(self):
        df = _make_df(1)
        engine = ReplayEngine()
        bars = []
        engine.run(df, lambda bar, eng: bars.append(bar))
        d = bars[0].to_dict()
        assert "timestamp" in d
        assert "open" in d
        assert d["bar_index"] == 0
