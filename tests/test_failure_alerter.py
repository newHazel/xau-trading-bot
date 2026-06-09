"""Tests for FailureAlerter — Phase 9.5."""

import pytest
from datetime import datetime, timezone
from core.monitoring.failure_alerter import FailureAlerter, AlertRecord


NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def alerter():
    return FailureAlerter({"retry_count": 3, "send_to_telegram": True})


class TestAlert:
    def test_create_alert(self, alerter):
        a = alerter.alert("critical", "data_feed", "connection lost", NOW)
        assert a.alert_id == 1
        assert a.severity == "critical"
        assert a.source == "data_feed"
        assert not a.resolved

    def test_ids_increment(self, alerter):
        a1 = alerter.alert("warning", "health", "degraded", NOW)
        a2 = alerter.alert("critical", "db", "write failed", NOW)
        assert a2.alert_id == 2

    def test_alert_stored(self, alerter):
        alerter.alert("critical", "test", "msg", NOW)
        assert len(alerter.alerts) == 1


class TestSendFunction:
    def test_send_on_alert(self, alerter):
        sent = []
        alerter.set_send_function(lambda msg: (sent.append(msg), True)[1])
        alerter.alert("critical", "data", "connection lost", NOW)
        assert len(sent) == 1
        assert "CRITICAL" in sent[0]

    def test_send_failure(self, alerter):
        alerter.set_send_function(lambda msg: False)
        a = alerter.alert("critical", "data", "connection lost", NOW)
        assert not a.sent
        assert a.retries == 1

    def test_send_exception(self, alerter):
        def fail(msg):
            raise ConnectionError("no internet")
        alerter.set_send_function(fail)
        a = alerter.alert("critical", "data", "test", NOW)
        assert not a.sent


class TestRetry:
    def test_retry_unsent(self, alerter):
        alerter.set_send_function(lambda msg: False)
        alerter.alert("critical", "data", "test", NOW)

        alerter.set_send_function(lambda msg: True)
        retried = alerter.retry_unsent()
        assert len(retried) == 1
        assert retried[0].sent

    def test_no_retry_after_max(self, alerter):
        call_count = [0]
        def fail(msg):
            call_count[0] += 1
            return False
        alerter.set_send_function(fail)
        alerter.alert("critical", "data", "test", NOW)
        alerter.retry_unsent()
        alerter.retry_unsent()
        # After 3 retries (1 initial + 2 retry_unsent), should stop
        alerter.retry_unsent()
        assert call_count[0] == 3  # max_retries = 3


class TestResolve:
    def test_resolve_alert(self, alerter):
        a = alerter.alert("warning", "test", "msg", NOW)
        resolved = alerter.resolve(a.alert_id)
        assert resolved.resolved

    def test_resolve_nonexistent(self, alerter):
        assert alerter.resolve(999) is None


class TestQueries:
    def test_unresolved(self, alerter):
        alerter.alert("critical", "data", "msg1", NOW)
        a2 = alerter.alert("warning", "health", "msg2", NOW)
        alerter.resolve(a2.alert_id)
        assert len(alerter.unresolved) == 1

    def test_by_severity(self, alerter):
        alerter.alert("critical", "data", "msg1", NOW)
        alerter.alert("warning", "health", "msg2", NOW)
        alerter.alert("critical", "db", "msg3", NOW)
        assert len(alerter.get_by_severity("critical")) == 2
        assert len(alerter.get_by_severity("warning")) == 1

    def test_get_recent(self, alerter):
        for i in range(5):
            alerter.alert("info", "test", f"msg-{i}", NOW)
        recent = alerter.get_recent(3)
        assert len(recent) == 3
        assert recent[0].message == "msg-4"


class TestClear:
    def test_clear(self, alerter):
        alerter.alert("critical", "test", "msg", NOW)
        alerter.clear()
        assert len(alerter.alerts) == 0


class TestToDict:
    def test_alert_to_dict(self, alerter):
        a = alerter.alert("critical", "data", "connection lost", NOW)
        d = a.to_dict()
        assert d["severity"] == "critical"
        assert d["source"] == "data"
        assert "timestamp" in d
