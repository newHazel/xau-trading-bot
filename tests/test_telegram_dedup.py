"""Tests for TelegramDedup — Phase 9.4."""

import pytest
from datetime import datetime, timezone
from core.monitoring.telegram_dedup import TelegramDedup


NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def dedup():
    return TelegramDedup({"max_dedup_history": 100})


class TestShouldSend:
    def test_first_message_allowed(self, dedup):
        assert dedup.should_send("signal alert", "XAU-001", NOW)

    def test_duplicate_blocked(self, dedup):
        dedup.should_send("signal alert", "XAU-001", NOW)
        assert not dedup.should_send("signal alert", "XAU-001", NOW)

    def test_different_content_allowed(self, dedup):
        dedup.should_send("signal alert", "XAU-001", NOW)
        assert dedup.should_send("different alert", "XAU-001", NOW)

    def test_different_setup_id_allowed(self, dedup):
        dedup.should_send("signal alert", "XAU-001", NOW)
        assert dedup.should_send("signal alert", "XAU-002", NOW)

    def test_different_minute_allowed(self, dedup):
        from datetime import timedelta
        dedup.should_send("signal alert", "XAU-001", NOW)
        assert dedup.should_send("signal alert", "XAU-001", NOW + timedelta(minutes=1))


class TestIsDuplicate:
    def test_not_duplicate_initially(self, dedup):
        assert not dedup.is_duplicate("test", "XAU-001", NOW)

    def test_is_duplicate_after_send(self, dedup):
        dedup.should_send("test", "XAU-001", NOW)
        assert dedup.is_duplicate("test", "XAU-001", NOW)


class TestMarkSent:
    def test_mark_sent(self, dedup):
        h = dedup.mark_sent("alert", "XAU-001", NOW)
        assert len(h) == 16
        assert dedup.is_duplicate("alert", "XAU-001", NOW)


class TestHistoryLimit:
    def test_trim_old_entries(self):
        d = TelegramDedup({"max_dedup_history": 5})
        for i in range(10):
            d.should_send(f"msg-{i}", f"XAU-{i:03d}", NOW)
        assert d.history_size == 5

    def test_oldest_evicted(self):
        d = TelegramDedup({"max_dedup_history": 3})
        d.should_send("a", "001", NOW)
        d.should_send("b", "002", NOW)
        d.should_send("c", "003", NOW)
        d.should_send("d", "004", NOW)
        assert not d.is_duplicate("a", "001", NOW)  # evicted
        assert d.is_duplicate("d", "004", NOW)


class TestClear:
    def test_clear(self, dedup):
        dedup.should_send("test", "XAU-001", NOW)
        dedup.clear()
        assert dedup.history_size == 0
        assert dedup.should_send("test", "XAU-001", NOW)


class TestNoneValues:
    def test_no_setup_id(self, dedup):
        assert dedup.should_send("alert", None, NOW)
        assert not dedup.should_send("alert", None, NOW)

    def test_no_timestamp(self, dedup):
        assert dedup.should_send("alert", "XAU-001")
        assert not dedup.should_send("alert", "XAU-001")
