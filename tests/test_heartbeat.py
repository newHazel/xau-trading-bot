"""Tests for HeartbeatManager — Phase 9.2."""

import pytest
from datetime import datetime, timezone, timedelta
from core.monitoring.heartbeat import HeartbeatManager, HeartbeatMessage


NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def hb():
    m = HeartbeatManager({"interval_minutes": 60, "enabled": True})
    m.start(NOW)
    return m


class TestIsDue:
    def test_first_heartbeat_always_due(self, hb):
        assert hb.is_due(NOW)

    def test_not_due_after_recent(self, hb):
        hb.generate("WAITING", "healthy", 0, NOW)
        assert not hb.is_due(NOW + timedelta(minutes=30))

    def test_due_after_interval(self, hb):
        hb.generate("WAITING", "healthy", 0, NOW)
        assert hb.is_due(NOW + timedelta(minutes=61))

    def test_disabled(self):
        m = HeartbeatManager({"enabled": False})
        assert not m.is_due(NOW)


class TestGenerate:
    def test_basic_message(self, hb):
        msg = hb.generate("SCANNING", "all healthy", 2, NOW + timedelta(minutes=30))
        assert msg.uptime_minutes == 30.0
        assert msg.current_state == "SCANNING"
        assert msg.trades_today == 2

    def test_with_signal(self, hb):
        sig_time = NOW + timedelta(minutes=10)
        hb.update_signal("XAU-001", sig_time)
        msg = hb.generate("SCANNING", "healthy", 1, NOW + timedelta(minutes=30))
        assert msg.last_signal_id == "XAU-001"
        assert msg.last_signal_time == sig_time

    def test_no_signal(self, hb):
        msg = hb.generate("WAITING", "healthy", 0, NOW)
        assert msg.last_signal_id is None


class TestFormat:
    def test_telegram_format(self, hb):
        msg = hb.generate("SCANNING", "healthy", 1, NOW + timedelta(minutes=60))
        text = msg.format_telegram()
        assert "Heartbeat" in text
        assert "SCANNING" in text
        assert "healthy" in text

    def test_telegram_format_with_signal(self, hb):
        hb.update_signal("XAU-001", NOW)
        msg = hb.generate("SCANNING", "healthy", 1, NOW + timedelta(minutes=60))
        text = msg.format_telegram()
        assert "XAU-001" in text


class TestToDict:
    def test_to_dict(self, hb):
        msg = hb.generate("WAITING", "healthy", 0, NOW)
        d = msg.to_dict()
        assert "uptime_minutes" in d
        assert "current_state" in d
        assert d["current_state"] == "WAITING"


class TestProperties:
    def test_enabled(self, hb):
        assert hb.enabled

    def test_interval(self, hb):
        assert hb.interval_minutes == 60
