"""Tests for JournalPage — Phase 10.1."""

import pytest
from datetime import datetime, timezone, timedelta
from dashboard.pages.journal_page import JournalEntry, JournalPageData, render_journal


NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)


def _make_entry(i=0, result="win", grade="A+", direction="LONG", net_r=1.5) -> JournalEntry:
    return JournalEntry(
        setup_id=f"XAU-{i:03d}",
        timestamp=NOW + timedelta(hours=i),
        direction=direction,
        grade=grade,
        entry_price=2650.0,
        sl_price=2640.0,
        tp1_price=2670.0,
        tp2_price=2685.0,
        result=result,
        net_r=net_r,
    )


@pytest.fixture
def page_data():
    data = JournalPageData()
    data.add_entry(_make_entry(0, result="win", net_r=2.0))
    data.add_entry(_make_entry(1, result="win", net_r=1.5))
    data.add_entry(_make_entry(2, result="loss", net_r=-1.0))
    data.add_entry(_make_entry(3, result="breakeven", net_r=0.0))
    return data


class TestJournalEntry:
    def test_to_dict(self):
        e = _make_entry(0)
        d = e.to_dict()
        assert d["setup_id"] == "XAU-000"
        assert d["result"] == "win"
        assert d["violations"] == []

    def test_with_violations(self):
        e = _make_entry(0)
        e.violations = ["wrong_session"]
        d = e.to_dict()
        assert "wrong_session" in d["violations"]


class TestJournalPageData:
    def test_total(self, page_data):
        assert page_data.total == 4

    def test_filter_by_result(self, page_data):
        assert len(page_data.filter_by_result("win")) == 2
        assert len(page_data.filter_by_result("loss")) == 1

    def test_filter_by_grade(self, page_data):
        page_data.add_entry(_make_entry(10, grade="B"))
        assert len(page_data.filter_by_grade(["A+"])) == 4

    def test_filter_by_direction(self, page_data):
        page_data.add_entry(_make_entry(10, direction="SHORT"))
        assert len(page_data.filter_by_direction("LONG")) == 4

    def test_filter_by_date_range(self, page_data):
        start = NOW + timedelta(hours=1)
        end = NOW + timedelta(hours=2)
        result = page_data.filter_by_date_range(start, end)
        assert len(result) == 2

    def test_get_with_violations(self, page_data):
        assert len(page_data.get_with_violations()) == 0
        e = _make_entry(10)
        e.violations = ["bad_session"]
        page_data.add_entry(e)
        assert len(page_data.get_with_violations()) == 1

    def test_get_summary(self, page_data):
        s = page_data.get_summary()
        assert s["total"] == 4
        assert s["by_result"]["win"] == 2
        assert s["avg_net_r"] == 0.625
        assert s["violation_count"] == 0

    def test_empty_summary(self):
        s = JournalPageData().get_summary()
        assert s["total"] == 0

    def test_to_records(self, page_data):
        recs = page_data.to_records()
        assert len(recs) == 4


class TestRender:
    def test_render_empty(self):
        result = render_journal()
        assert result["page"] == "Journal"

    def test_render_with_data(self, page_data):
        result = render_journal(page_data)
        assert result["summary"]["total"] == 4
