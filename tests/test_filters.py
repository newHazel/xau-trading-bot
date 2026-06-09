"""Tests for filter components — Phase 10.2."""

import pytest
from datetime import datetime, timezone
from dashboard.components.filters import FilterConfig, apply_filters


NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)

RECORDS = [
    {"setup_id": "XAU-001", "grade": "A+", "direction": "LONG", "status": "closed", "net_r": 2.0, "timestamp": "2026-01-21T10:00:00+00:00"},
    {"setup_id": "XAU-002", "grade": "A", "direction": "SHORT", "status": "closed", "net_r": -1.0, "timestamp": "2026-01-21T11:00:00+00:00"},
    {"setup_id": "XAU-003", "grade": "A+", "direction": "LONG", "status": "open", "net_r": None, "timestamp": "2026-01-21T12:00:00+00:00"},
    {"setup_id": "XAU-004", "grade": "B", "direction": "SHORT", "status": "closed", "net_r": 1.5, "timestamp": "2026-01-21T14:00:00+00:00"},
]


class TestFilterConfig:
    def test_empty(self):
        f = FilterConfig()
        assert f.is_empty()

    def test_not_empty(self):
        f = FilterConfig(grades=["A+"])
        assert not f.is_empty()

    def test_to_dict(self):
        f = FilterConfig(grades=["A+"], min_r=1.0)
        d = f.to_dict()
        assert d["grades"] == ["A+"]
        assert d["min_r"] == 1.0
        assert "directions" not in d


class TestApplyFilters:
    def test_no_filters(self):
        result = apply_filters(RECORDS, FilterConfig())
        assert len(result) == 4

    def test_filter_by_grade(self):
        result = apply_filters(RECORDS, FilterConfig(grades=["A+"]))
        assert len(result) == 2

    def test_filter_by_direction(self):
        result = apply_filters(RECORDS, FilterConfig(directions=["LONG"]))
        assert len(result) == 2

    def test_filter_by_status(self):
        result = apply_filters(RECORDS, FilterConfig(statuses=["closed"]))
        assert len(result) == 3

    def test_filter_by_min_r(self):
        result = apply_filters(RECORDS, FilterConfig(min_r=1.0))
        assert len(result) == 2

    def test_filter_by_max_r(self):
        result = apply_filters(RECORDS, FilterConfig(max_r=0.0))
        assert len(result) == 2  # -1.0 and None (treated as 0)

    def test_filter_by_date_range(self):
        start = datetime(2026, 1, 21, 11, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 21, 13, 0, tzinfo=timezone.utc)
        result = apply_filters(RECORDS, FilterConfig(date_start=start, date_end=end))
        assert len(result) == 2

    def test_combined_filters(self):
        result = apply_filters(RECORDS, FilterConfig(grades=["A+"], directions=["LONG"]))
        assert len(result) == 2

    def test_empty_records(self):
        result = apply_filters([], FilterConfig(grades=["A+"]))
        assert len(result) == 0
