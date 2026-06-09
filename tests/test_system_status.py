"""Tests for SystemStatusManager — Phase 9.3."""

import pytest
from datetime import datetime, timezone, timedelta
from core.monitoring.system_status import SystemStatusManager, SystemState, SystemStatusReport


NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def mgr():
    m = SystemStatusManager()
    m.start(NOW)
    return m


class TestInitialState:
    def test_starts_healthy(self, mgr):
        assert mgr.state == SystemState.HEALTHY

    def test_no_issues(self, mgr):
        assert len(mgr.issues) == 0


class TestUpdateFromHealth:
    def test_stays_healthy(self, mgr):
        mgr.update_from_health(failed_count=0, warned_count=0, now=NOW)
        assert mgr.state == SystemState.HEALTHY

    def test_degrades_on_1_failure(self, mgr):
        mgr.update_from_health(failed_count=1, warned_count=0, now=NOW)
        assert mgr.state == SystemState.DEGRADED

    def test_degrades_on_2_warnings(self, mgr):
        mgr.update_from_health(failed_count=0, warned_count=2, now=NOW)
        assert mgr.state == SystemState.DEGRADED

    def test_error_on_3_failures(self, mgr):
        mgr.update_from_health(failed_count=3, warned_count=0, now=NOW)
        assert mgr.state == SystemState.ERROR

    def test_recovers_to_healthy(self, mgr):
        mgr.update_from_health(failed_count=2, warned_count=0, now=NOW)
        assert mgr.state == SystemState.DEGRADED
        mgr.update_from_health(failed_count=0, warned_count=0, now=NOW + timedelta(minutes=5))
        assert mgr.state == SystemState.HEALTHY


class TestMaintenance:
    def test_set_maintenance(self, mgr):
        mgr.set_maintenance("deploying update", NOW)
        assert mgr.state == SystemState.MAINTENANCE

    def test_maintenance_not_overridden_by_health(self, mgr):
        mgr.set_maintenance("deploying", NOW)
        mgr.update_from_health(failed_count=0, warned_count=0, now=NOW)
        assert mgr.state == SystemState.MAINTENANCE

    def test_clear_maintenance(self, mgr):
        mgr.set_maintenance("deploying", NOW)
        mgr.clear_maintenance(NOW + timedelta(minutes=10))
        assert mgr.state == SystemState.HEALTHY


class TestIssues:
    def test_add_issue(self, mgr):
        mgr.add_issue("data feed slow")
        assert "data feed slow" in mgr.issues

    def test_no_duplicate_issues(self, mgr):
        mgr.add_issue("slow")
        mgr.add_issue("slow")
        assert len(mgr.issues) == 1

    def test_clear_issue(self, mgr):
        mgr.add_issue("slow")
        mgr.clear_issue("slow")
        assert len(mgr.issues) == 0

    def test_clear_all(self, mgr):
        mgr.add_issue("a")
        mgr.add_issue("b")
        mgr.clear_all_issues()
        assert len(mgr.issues) == 0


class TestReport:
    def test_get_report(self, mgr):
        report = mgr.get_report(NOW + timedelta(minutes=30))
        assert report.state == SystemState.HEALTHY
        assert report.uptime_minutes == 30.0

    def test_report_to_dict(self, mgr):
        d = mgr.get_report(NOW).to_dict()
        assert d["state"] == "healthy"
        assert "active_issues" in d


class TestTransitions:
    def test_transition_recorded(self, mgr):
        mgr.update_from_health(failed_count=2, warned_count=0, now=NOW)
        report = mgr.get_report(NOW)
        assert report.last_transition is not None
        assert report.last_transition.to_state == SystemState.DEGRADED

    def test_no_transition_on_same_state(self, mgr):
        mgr.update_from_health(failed_count=0, warned_count=0, now=NOW)
        report = mgr.get_report(NOW)
        assert report.last_transition is None
