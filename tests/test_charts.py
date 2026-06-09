"""Tests for chart components — Phase 10.2."""

import pytest
from dashboard.components.charts import plot_equity_curve, plot_drawdown_curve, plot_trade_scatter


class TestEquityCurve:
    def test_empty(self):
        result = plot_equity_curve([])
        assert result["points"] == []
        assert result["final_r"] == 0.0

    def test_basic(self):
        result = plot_equity_curve([1.5, -1.0, 2.0, -0.5])
        assert result["chart"] == "equity_curve"
        assert len(result["points"]) == 4
        assert result["final_r"] == 2.0

    def test_cumulative(self):
        result = plot_equity_curve([1.0, 1.0, 1.0])
        points = result["points"]
        assert points[0]["cumulative_r"] == 1.0
        assert points[1]["cumulative_r"] == 2.0
        assert points[2]["cumulative_r"] == 3.0

    def test_all_losses(self):
        result = plot_equity_curve([-1.0, -1.0])
        assert result["final_r"] == -2.0


class TestDrawdownCurve:
    def test_empty(self):
        result = plot_drawdown_curve([])
        assert result["max_drawdown_r"] == 0.0

    def test_basic(self):
        result = plot_drawdown_curve([2.0, -1.0, -0.5, 1.0])
        assert result["chart"] == "drawdown"
        assert len(result["points"]) == 4
        assert result["max_drawdown_r"] == 1.5

    def test_no_drawdown(self):
        result = plot_drawdown_curve([1.0, 1.0, 1.0])
        assert result["max_drawdown_r"] == 0.0

    def test_all_losses(self):
        result = plot_drawdown_curve([-1.0, -1.0])
        assert result["max_drawdown_r"] == 2.0


class TestTradeScatter:
    def test_empty(self):
        result = plot_trade_scatter([])
        assert result["total"] == 0

    def test_basic(self):
        trades = [
            {"net_r": 1.5, "direction": "LONG", "grade": "A+"},
            {"net_r": -1.0, "direction": "SHORT", "grade": "A"},
            {"net_r": 2.0, "direction": "LONG", "grade": "A+"},
        ]
        result = plot_trade_scatter(trades)
        assert result["total"] == 3
        assert result["winners"] == 2
        assert result["losers"] == 1

    def test_missing_fields(self):
        trades = [{"net_r": 1.0}]
        result = plot_trade_scatter(trades)
        assert result["points"][0]["direction"] == "unknown"
