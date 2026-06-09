"""Tests for WalkForwardRunner + Splitter — Phase 6.7/6.8."""

import pytest
import pandas as pd
from datetime import datetime, timezone

from backtesting.walk_forward import (
    WalkForwardRunner, WalkForwardResult, split_in_out_of_sample, WalkForwardFold,
)


def _make_df(n: int = 300) -> pd.DataFrame:
    base = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    dates = pd.date_range(base, periods=n, freq="1min")
    data = {
        "open": [2000 + i * 0.1 for i in range(n)],
        "high": [2001 + i * 0.1 for i in range(n)],
        "low": [1999 + i * 0.1 for i in range(n)],
        "close": [2000.5 + i * 0.1 for i in range(n)],
        "volume": [100] * n,
    }
    return pd.DataFrame(data, index=dates)


class TestSplitter:
    def test_single_fold(self):
        splits = split_in_out_of_sample(100, is_ratio=0.70, num_folds=1)
        assert len(splits) == 1
        is_start, is_end, oos_start, oos_end = splits[0]
        assert is_start == 0
        assert is_end == 70
        assert oos_start == 70
        assert oos_end == 100

    def test_three_folds(self):
        splits = split_in_out_of_sample(300, is_ratio=0.70, num_folds=3)
        assert len(splits) == 3
        for is_start, is_end, oos_start, oos_end in splits:
            assert is_start < is_end
            assert is_end == oos_start
            assert oos_start < oos_end

    def test_no_overlap_between_is_oos(self):
        splits = split_in_out_of_sample(200, is_ratio=0.70, num_folds=2)
        for _, is_end, oos_start, _ in splits:
            assert is_end == oos_start

    def test_too_few_bars_raises(self):
        with pytest.raises(ValueError, match="Too few"):
            split_in_out_of_sample(5)

    def test_zero_folds_raises(self):
        with pytest.raises(ValueError):
            split_in_out_of_sample(100, num_folds=0)


class TestWalkForwardRunner:
    def test_basic_run(self):
        df = _make_df(300)
        signals = [
            {"bar_index": i, "r_multiple": 1.5, "net_pnl": 50, "direction": "long", "grade": "A"}
            for i in range(0, 300, 5)
        ]
        def mock_backtest(bt_df, bt_signals):
            return [s for s in bt_signals]

        runner = WalkForwardRunner({"num_folds": 3, "min_oos_trades": 1, "min_oos_win_rate": 0.0})
        result = runner.run(df, signals, mock_backtest)
        assert len(result.folds) == 3

    def test_passed_with_good_data(self):
        df = _make_df(300)
        signals = [
            {"bar_index": i, "r_multiple": 2.0, "net_pnl": 100, "direction": "long", "grade": "A"}
            for i in range(0, 300, 3)
        ]
        def mock_backtest(bt_df, bt_signals):
            return bt_signals

        runner = WalkForwardRunner({
            "num_folds": 3,
            "min_oos_trades": 5,
            "min_oos_win_rate": 0.40,
            "max_oos_drawdown_r": 50.0,
        })
        result = runner.run(df, signals, mock_backtest)
        assert result.passed

    def test_fails_insufficient_trades(self):
        df = _make_df(300)
        signals = [{"bar_index": 5, "r_multiple": 2.0, "net_pnl": 100}]
        def mock_backtest(bt_df, bt_signals):
            return bt_signals

        runner = WalkForwardRunner({"num_folds": 3, "min_oos_trades": 50})
        result = runner.run(df, signals, mock_backtest)
        assert not result.passed

    def test_aggregate_metrics(self):
        df = _make_df(300)
        signals = [
            {"bar_index": i, "r_multiple": 1.0, "net_pnl": 50}
            for i in range(0, 300, 10)
        ]
        def mock_backtest(bt_df, bt_signals):
            return bt_signals

        runner = WalkForwardRunner({"num_folds": 3, "min_oos_trades": 1, "min_oos_win_rate": 0.0})
        result = runner.run(df, signals, mock_backtest)
        assert result.aggregate_oos_metrics is not None

    def test_config_hash_propagated(self):
        df = _make_df(100)
        runner = WalkForwardRunner({"num_folds": 1, "min_oos_trades": 0, "min_oos_win_rate": 0.0})
        result = runner.run(df, [], lambda d, s: s, config_hash="abc123")
        assert result.config_hash == "abc123"


class TestFoldToDict:
    def test_fold_to_dict(self):
        fold = WalkForwardFold(fold_index=0, is_start=0, is_end=70, oos_start=70, oos_end=100)
        d = fold.to_dict()
        assert d["fold_index"] == 0
        assert d["is_range"] == [0, 70]
        assert d["oos_range"] == [70, 100]

    def test_result_to_dict(self):
        result = WalkForwardResult(passed=True, config_hash="test123", detail="ok")
        d = result.to_dict()
        assert d["passed"] is True
        assert d["config_hash"] == "test123"
