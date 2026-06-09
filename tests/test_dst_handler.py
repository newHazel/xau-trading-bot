"""Tests for DST Handler."""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from core.utils.dst_handler import DSTHandler

ISR = ZoneInfo("Asia/Jerusalem")
UTC = ZoneInfo("Etc/UTC")

CONFIG = {
    "timezone": "Asia/Jerusalem",
    "dst": {
        "use_broker_timezone": True,
        "handle_dst_transitions": True,
        "dst_transition_buffer_days": 3,
    },
}


@pytest.fixture
def handler():
    return DSTHandler(CONFIG)


class TestDSTDetection:
    def test_summer_is_dst(self, handler):
        dt = datetime(2026, 7, 15, 12, 0, tzinfo=ISR)
        assert handler.is_dst_active(dt) is True

    def test_winter_is_not_dst(self, handler):
        dt = datetime(2026, 1, 15, 12, 0, tzinfo=ISR)
        assert handler.is_dst_active(dt) is False


class TestUTCOffset:
    def test_winter_offset(self, handler):
        dt = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
        assert handler.utc_offset_hours(dt) == 2.0

    def test_summer_offset(self, handler):
        dt = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
        assert handler.utc_offset_hours(dt) == 3.0


class TestNearTransition:
    def test_near_spring_transition(self, handler):
        # Israel DST 2026 starts last Friday of March → 2026-03-27
        dt = datetime(2026, 3, 25, 12, 0, tzinfo=ISR)
        assert handler.is_near_dst_transition(dt) is True

    def test_far_from_transition(self, handler):
        dt = datetime(2026, 6, 15, 12, 0, tzinfo=ISR)
        assert handler.is_near_dst_transition(dt) is False

    def test_disabled_transitions(self):
        config = {
            "timezone": "Asia/Jerusalem",
            "dst": {"handle_dst_transitions": False},
        }
        h = DSTHandler(config)
        dt = datetime(2026, 3, 25, 12, 0, tzinfo=ISR)
        assert h.is_near_dst_transition(dt) is False


class TestAdjustSessionTimes:
    def test_returns_unchanged(self, handler):
        dt = datetime(2026, 3, 18, 12, 0, tzinfo=ISR)
        start, end = handler.adjust_session_times("10:00", "13:00", dt)
        assert start == "10:00"
        assert end == "13:00"
