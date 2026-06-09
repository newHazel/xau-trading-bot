"""Tests for WalkForwardPage — Phase 10.1."""

import pytest
from dashboard.pages.walk_forward_page import FoldResult, WalkForwardPageData, render_walk_forward


def _make_fold(i=0, passed=True, oos_wr=0.55, oos_r=5.0) -> FoldResult:
    return FoldResult(
        fold_index=i,
        is_start="2025-01-01",
        is_end="2025-06-30",
        oos_start="2025-07-01",
        oos_end="2025-09-30",
        is_trades=80 + i,
        oos_trades=30 + i,
        is_win_rate=0.58,
        oos_win_rate=oos_wr,
        oos_total_r=oos_r,
        passed=passed,
    )


@pytest.fixture
def page_data():
    data = WalkForwardPageData()
    for i in range(3):
        data.add_fold(_make_fold(i, passed=True, oos_r=5.0 + i))
    data.overall_passed = True
    return data


class TestFoldResult:
    def test_to_dict(self):
        f = _make_fold(0)
        d = f.to_dict()
        assert d["fold_index"] == 0
        assert "→" in d["is_period"]
        assert d["passed"] is True


class TestWalkForwardPageData:
    def test_total_folds(self, page_data):
        assert page_data.total_folds == 3

    def test_passed_folds(self, page_data):
        assert page_data.passed_folds == 3

    def test_failed_folds(self, page_data):
        page_data.add_fold(_make_fold(10, passed=False))
        assert page_data.failed_folds == 1

    def test_oos_metrics(self, page_data):
        m = page_data.get_oos_metrics()
        assert "avg_oos_win_rate" in m
        assert "avg_oos_total_r" in m
        assert m["max_oos_total_r"] == 7.0

    def test_oos_metrics_empty(self):
        assert WalkForwardPageData().get_oos_metrics() == {}

    def test_get_summary(self, page_data):
        s = page_data.get_summary()
        assert s["total_folds"] == 3
        assert s["overall_passed"] is True

    def test_to_records(self, page_data):
        recs = page_data.to_records()
        assert len(recs) == 3


class TestRender:
    def test_render_empty(self):
        result = render_walk_forward()
        assert result["page"] == "Walk-Forward"
        assert result["summary"]["total_folds"] == 0

    def test_render_with_data(self, page_data):
        result = render_walk_forward(page_data)
        assert result["summary"]["overall_passed"] is True
