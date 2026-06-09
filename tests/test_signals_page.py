"""Tests for SignalsPage — Phase 10.1."""

import pytest
from datetime import datetime, timezone, timedelta
from dashboard.pages.signals_page import SignalRow, SignalsPageData, render_signals


NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)


def _make_signal(i=0, grade="A+", direction="LONG", status="closed") -> SignalRow:
    return SignalRow(
        setup_id=f"XAU-{i:03d}",
        timestamp=NOW + timedelta(hours=i),
        direction=direction,
        grade=grade,
        entry=2650.0 + i,
        sl=2640.0 + i,
        tp1=2670.0 + i,
        tp2=2685.0 + i,
        status=status,
        net_r=1.5 if status == "closed" else None,
    )


@pytest.fixture
def page_data():
    data = SignalsPageData()
    for i in range(5):
        data.add_signal(_make_signal(i))
    return data


class TestSignalRow:
    def test_to_dict(self):
        s = _make_signal(0)
        d = s.to_dict()
        assert d["setup_id"] == "XAU-000"
        assert d["grade"] == "A+"
        assert "timestamp" in d

    def test_none_net_r(self):
        s = _make_signal(0, status="open")
        assert s.to_dict()["net_r"] is None


class TestSignalsPageData:
    def test_total(self, page_data):
        assert page_data.total == 5

    def test_filter_by_grade(self, page_data):
        page_data.add_signal(_make_signal(10, grade="B"))
        assert len(page_data.filter_by_grade(["A+"])) == 5
        assert len(page_data.filter_by_grade(["B"])) == 1

    def test_filter_by_direction(self, page_data):
        page_data.add_signal(_make_signal(10, direction="SHORT"))
        assert len(page_data.filter_by_direction("LONG")) == 5
        assert len(page_data.filter_by_direction("SHORT")) == 1

    def test_filter_by_status(self, page_data):
        page_data.add_signal(_make_signal(10, status="open"))
        assert len(page_data.filter_by_status("closed")) == 5

    def test_filter_by_date_range(self, page_data):
        start = NOW + timedelta(hours=1)
        end = NOW + timedelta(hours=3)
        result = page_data.filter_by_date_range(start, end)
        assert len(result) == 3

    def test_get_summary(self, page_data):
        s = page_data.get_summary()
        assert s["total"] == 5
        assert "A+" in s["by_grade"]
        assert "LONG" in s["by_direction"]

    def test_empty_summary(self):
        s = SignalsPageData().get_summary()
        assert s["total"] == 0

    def test_to_records(self, page_data):
        recs = page_data.to_records()
        assert len(recs) == 5
        assert all(isinstance(r, dict) for r in recs)


class TestRender:
    def test_render_empty(self):
        result = render_signals()
        assert result["page"] == "Signals"
        assert result["summary"]["total"] == 0

    def test_render_with_data(self, page_data):
        result = render_signals(page_data)
        assert result["summary"]["total"] == 5
        assert len(result["records"]) == 5
