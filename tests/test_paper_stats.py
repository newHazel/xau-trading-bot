"""Tests for PaperStats — Phase 7.4."""

import pytest
from paper_trading.paper_stats import PaperStats, PaperStatsResult, ComparisonResult


@pytest.fixture
def stats():
    return PaperStats({"min_performance_ratio": 0.70, "min_paper_trades": 20})


@pytest.fixture
def bt_metrics():
    return {"win_rate": 0.60, "avg_r": 1.5, "profit_factor": 2.5, "expectancy": 0.9}


class TestPassed:
    def test_good_paper_performance(self, stats, bt_metrics):
        paper = {"win_rate": 0.55, "avg_r": 1.2, "profit_factor": 2.0, "expectancy": 0.8}
        r = stats.compare(bt_metrics, paper, total_paper_trades=25)
        assert r.overall_passed
        assert len(r.degraded_metrics) == 0

    def test_equal_performance(self, stats, bt_metrics):
        r = stats.compare(bt_metrics, bt_metrics, total_paper_trades=25)
        assert r.overall_passed


class TestDegraded:
    def test_low_win_rate(self, stats, bt_metrics):
        paper = {"win_rate": 0.30, "avg_r": 1.2, "profit_factor": 2.0, "expectancy": 0.8}
        r = stats.compare(bt_metrics, paper, total_paper_trades=25)
        assert not r.overall_passed
        assert "win_rate" in r.degraded_metrics

    def test_low_profit_factor(self, stats, bt_metrics):
        paper = {"win_rate": 0.55, "avg_r": 1.2, "profit_factor": 1.0, "expectancy": 0.8}
        r = stats.compare(bt_metrics, paper, total_paper_trades=25)
        assert "profit_factor" in r.degraded_metrics

    def test_multiple_degraded(self, stats, bt_metrics):
        paper = {"win_rate": 0.20, "avg_r": 0.3, "profit_factor": 0.5, "expectancy": 0.1}
        r = stats.compare(bt_metrics, paper, total_paper_trades=25)
        assert len(r.degraded_metrics) >= 3

    def test_insufficient_trades(self, stats, bt_metrics):
        paper = {"win_rate": 0.60, "avg_r": 1.5, "profit_factor": 2.5, "expectancy": 0.9}
        r = stats.compare(bt_metrics, paper, total_paper_trades=10)
        assert not r.overall_passed
        assert "insufficient" in r.detail


class TestEdgeCases:
    def test_zero_backtest_metric(self, stats):
        bt = {"win_rate": 0, "avg_r": 0, "profit_factor": 0, "expectancy": 0}
        paper = {"win_rate": 0.5, "avg_r": 1.0, "profit_factor": 2.0, "expectancy": 0.5}
        r = stats.compare(bt, paper, total_paper_trades=25)
        assert r.overall_passed

    def test_default_config(self):
        s = PaperStats()
        bt = {"win_rate": 0.60, "avg_r": 1.5, "profit_factor": 2.5, "expectancy": 0.9}
        paper = {"win_rate": 0.55, "avg_r": 1.2, "profit_factor": 2.0, "expectancy": 0.8}
        r = s.compare(bt, paper, total_paper_trades=25)
        assert r.overall_passed


class TestToDict:
    def test_comparison_to_dict(self):
        c = ComparisonResult("win_rate", 0.60, 0.55, 0.917, True, "ok")
        d = c.to_dict()
        assert d["metric_name"] == "win_rate"
        assert d["passed"]

    def test_result_to_dict(self, stats, bt_metrics):
        paper = {"win_rate": 0.55, "avg_r": 1.2, "profit_factor": 2.0, "expectancy": 0.8}
        r = stats.compare(bt_metrics, paper, total_paper_trades=25)
        d = r.to_dict()
        assert "comparisons" in d
        assert "overall_passed" in d
        assert d["total_paper_trades"] == 25
