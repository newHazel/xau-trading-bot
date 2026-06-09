"""Tests for HealthPage — Phase 10.1."""

import pytest
from datetime import datetime, timezone
from dashboard.pages.health_page import HealthCheckRow, AlertRow, HealthPageData, render_health


NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def page_data():
    data = HealthPageData(system_state="healthy", uptime_minutes=120.5, last_heartbeat=NOW)
    data.add_check(HealthCheckRow("data_freshness", "pass", "2 min ago", NOW))
    data.add_check(HealthCheckRow("db_writable", "pass", "ok", NOW))
    data.add_check(HealthCheckRow("memory_usage_ok", "warn", "420 MB", NOW))
    data.add_alert(AlertRow(1, "critical", "data_feed", "connection lost", NOW, False, True))
    data.add_alert(AlertRow(2, "warning", "health", "degraded", NOW, True, True))
    return data


class TestHealthCheckRow:
    def test_to_dict(self):
        c = HealthCheckRow("test", "pass", "ok", NOW)
        d = c.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "pass"


class TestAlertRow:
    def test_to_dict(self):
        a = AlertRow(1, "critical", "data", "msg", NOW, False, True)
        d = a.to_dict()
        assert d["alert_id"] == 1
        assert d["resolved"] is False


class TestHealthPageData:
    def test_failed_checks(self, page_data):
        assert len(page_data.failed_checks) == 0
        page_data.add_check(HealthCheckRow("test", "fail", "bad", NOW))
        assert len(page_data.failed_checks) == 1

    def test_warned_checks(self, page_data):
        assert len(page_data.warned_checks) == 1

    def test_unresolved_alerts(self, page_data):
        assert len(page_data.unresolved_alerts) == 1

    def test_get_summary(self, page_data):
        s = page_data.get_summary()
        assert s["system_state"] == "healthy"
        assert s["uptime_minutes"] == 120.5
        assert s["total_checks"] == 3
        assert s["warned_checks"] == 1
        assert s["unresolved_alerts"] == 1
        assert s["last_heartbeat"] is not None

    def test_checks_to_records(self, page_data):
        recs = page_data.checks_to_records()
        assert len(recs) == 3

    def test_alerts_to_records(self, page_data):
        recs = page_data.alerts_to_records()
        assert len(recs) == 2

    def test_active_issues(self, page_data):
        page_data.active_issues = ["slow feed", "high memory"]
        s = page_data.get_summary()
        assert len(s["active_issues"]) == 2


class TestRender:
    def test_render_empty(self):
        result = render_health()
        assert result["page"] == "Health"
        assert result["summary"]["total_checks"] == 0

    def test_render_with_data(self, page_data):
        result = render_health(page_data)
        assert result["summary"]["system_state"] == "healthy"
        assert len(result["checks"]) == 3
        assert len(result["alerts"]) == 2
