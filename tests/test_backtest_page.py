"""Tests for BacktestPage — Phase 10.1."""

import pytest
from dashboard.pages.backtest_page import BacktestResult, BacktestPageData, render_backtest


def _make_result(i=0, win_rate=0.55, total_r=10.0) -> BacktestResult:
    return BacktestResult(
        experiment_id=f"exp-{i:03d}",
        config_hash=f"hash-{i}",
        total_trades=100 + i * 10,
        win_rate=win_rate,
        avg_r=0.3 + i * 0.01,
        profit_factor=1.5 + i * 0.1,
        max_drawdown_pct=5.0 + i,
        sharpe=1.2 + i * 0.05,
        expectancy=0.2 + i * 0.01,
        total_r=total_r,
    )


@pytest.fixture
def page_data():
    data = BacktestPageData()
    for i in range(4):
        data.add_result(_make_result(i, total_r=10.0 + i * 5))
    return data


class TestBacktestResult:
    def test_to_dict(self):
        r = _make_result(0)
        d = r.to_dict()
        assert d["experiment_id"] == "exp-000"
        assert isinstance(d["win_rate"], float)


class TestBacktestPageData:
    def test_total_experiments(self, page_data):
        assert page_data.total_experiments == 4

    def test_get_best_by_total_r(self, page_data):
        best = page_data.get_best_by("total_r")
        assert best.total_r == 25.0

    def test_get_best_by_win_rate(self, page_data):
        best = page_data.get_best_by("win_rate")
        assert best is not None

    def test_get_best_empty(self):
        assert BacktestPageData().get_best_by("total_r") is None

    def test_filter_by_min_trades(self, page_data):
        result = page_data.filter_by_min_trades(120)
        assert len(result) == 2

    def test_filter_by_min_win_rate(self, page_data):
        page_data.add_result(_make_result(10, win_rate=0.70))
        result = page_data.filter_by_min_win_rate(0.60)
        assert len(result) == 1

    def test_get_summary(self, page_data):
        s = page_data.get_summary()
        assert s["total_experiments"] == 4
        assert "avg_win_rate" in s
        assert s["best_total_r"] == 25.0

    def test_empty_summary(self):
        s = BacktestPageData().get_summary()
        assert s["total_experiments"] == 0

    def test_to_records(self, page_data):
        recs = page_data.to_records()
        assert len(recs) == 4


class TestRender:
    def test_render_empty(self):
        result = render_backtest()
        assert result["page"] == "Backtest"

    def test_render_with_data(self, page_data):
        result = render_backtest(page_data)
        assert result["summary"]["total_experiments"] == 4
