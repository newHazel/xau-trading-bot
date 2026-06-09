"""Tests for Backtest Metrics — Phase 6.6."""

import pytest
from backtesting.metrics import compute_metrics, MetricsResult


def _winning_trades(n: int = 5) -> list:
    return [
        {"r_multiple": 2.0, "net_pnl": 100.0, "direction": "long", "grade": "A", "bar_entry": i * 10, "bar_exit": i * 10 + 5}
        for i in range(n)
    ]


def _losing_trades(n: int = 3) -> list:
    return [
        {"r_multiple": -1.0, "net_pnl": -50.0, "direction": "long", "grade": "A", "bar_entry": i * 10, "bar_exit": i * 10 + 3}
        for i in range(n)
    ]


def _mixed_trades() -> list:
    return _winning_trades(5) + _losing_trades(3)


class TestBasicMetrics:
    def test_total_trades(self):
        m = compute_metrics(_mixed_trades())
        assert m.total_trades == 8

    def test_wins_and_losses(self):
        m = compute_metrics(_mixed_trades())
        assert m.wins == 5
        assert m.losses == 3

    def test_win_rate(self):
        m = compute_metrics(_mixed_trades())
        assert abs(m.win_rate - 5 / 8) < 0.001

    def test_avg_r(self):
        trades = _mixed_trades()
        expected = (5 * 2.0 + 3 * (-1.0)) / 8
        m = compute_metrics(trades)
        assert abs(m.avg_r - expected) < 0.001

    def test_avg_win_r(self):
        m = compute_metrics(_mixed_trades())
        assert abs(m.avg_win_r - 2.0) < 0.001

    def test_avg_loss_r(self):
        m = compute_metrics(_mixed_trades())
        assert abs(m.avg_loss_r - (-1.0)) < 0.001


class TestProfitFactor:
    def test_profit_factor(self):
        m = compute_metrics(_mixed_trades())
        # gross_wins = 5*2 = 10, gross_losses = 3*1 = 3
        assert abs(m.profit_factor - 10 / 3) < 0.01

    def test_all_wins_inf(self):
        m = compute_metrics(_winning_trades(3))
        assert m.profit_factor == float("inf")

    def test_all_losses_zero(self):
        m = compute_metrics(_losing_trades(3))
        assert m.profit_factor == 0


class TestDrawdown:
    def test_max_drawdown_r(self):
        trades = [
            {"r_multiple": 2.0, "net_pnl": 100},
            {"r_multiple": -1.0, "net_pnl": -50},
            {"r_multiple": -1.0, "net_pnl": -50},
            {"r_multiple": 2.0, "net_pnl": 100},
        ]
        m = compute_metrics(trades)
        assert m.max_drawdown_r == 2.0

    def test_no_drawdown(self):
        m = compute_metrics(_winning_trades(5))
        assert m.max_drawdown_r == 0


class TestSharpe:
    def test_sharpe_positive(self):
        m = compute_metrics(_mixed_trades())
        assert m.sharpe_like > 0

    def test_sharpe_all_same(self):
        trades = [{"r_multiple": 1.0, "net_pnl": 50}] * 5
        m = compute_metrics(trades)
        # all same → std = 0 → sharpe = 0
        assert m.sharpe_like == 0


class TestExpectancy:
    def test_expectancy(self):
        m = compute_metrics(_mixed_trades())
        expected = (m.win_rate * m.avg_win_r) + ((1 - m.win_rate) * m.avg_loss_r)
        assert abs(m.expectancy - expected) < 0.001


class TestBreakdowns:
    def test_by_direction(self):
        trades = [
            {"r_multiple": 2.0, "net_pnl": 100, "direction": "long", "grade": "A"},
            {"r_multiple": -1.0, "net_pnl": -50, "direction": "short", "grade": "B"},
        ]
        m = compute_metrics(trades)
        assert "long" in m.breakdowns["by_direction"]
        assert "short" in m.breakdowns["by_direction"]

    def test_by_grade(self):
        trades = [
            {"r_multiple": 2.0, "net_pnl": 100, "direction": "long", "grade": "A+"},
            {"r_multiple": 1.5, "net_pnl": 75, "direction": "long", "grade": "A"},
        ]
        m = compute_metrics(trades)
        assert "A+" in m.breakdowns["by_grade"]
        assert "A" in m.breakdowns["by_grade"]


class TestEmpty:
    def test_empty_trades(self):
        m = compute_metrics([])
        assert m.total_trades == 0
        assert m.win_rate == 0
        assert m.profit_factor == 0


class TestToDict:
    def test_to_dict_keys(self):
        m = compute_metrics(_mixed_trades())
        d = m.to_dict()
        assert "total_trades" in d
        assert "win_rate" in d
        assert "profit_factor" in d
        assert "breakdowns" in d


class TestEdgeCases:
    def test_single_trade(self):
        m = compute_metrics([{"r_multiple": 2.0, "net_pnl": 100}])
        assert m.total_trades == 1
        assert m.win_rate == 1.0

    def test_bars_in_trade(self):
        trades = [{"r_multiple": 1.0, "net_pnl": 50, "bar_entry": 0, "bar_exit": 10}]
        m = compute_metrics(trades)
        assert m.avg_bars_in_trade == 10.0
