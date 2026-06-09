"""Tests for Session / Kill Zone Filter — Phase 3.1."""

import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from core.filters.session_filter import SessionFilter, SessionName, SessionResult

ISR = ZoneInfo("Asia/Jerusalem")
UTC = ZoneInfo("Etc/UTC")

SESSIONS_CONFIG = {
    "timezone": "Asia/Jerusalem",
    "asia_range": {
        "enabled": True,
        "start": "02:00",
        "end": "07:00",
        "trade_allowed": False,
        "mark_high_low": True,
    },
    "london_kill_zone": {
        "enabled": True,
        "start": "10:00",
        "end": "13:00",
        "trade_allowed": True,
    },
    "ny_kill_zone": {
        "enabled": True,
        "start": "15:30",
        "end": "18:00",
        "trade_allowed": True,
    },
    "overlap": {
        "enabled": True,
        "start": "15:30",
        "end": "17:00",
        "trade_allowed": True,
        "priority": "highest",
    },
    "dst": {
        "use_broker_timezone": True,
        "convert_to_local_for_display": True,
        "handle_dst_transitions": True,
        "dst_transition_buffer_days": 3,
    },
}


@pytest.fixture
def sf():
    return SessionFilter(SESSIONS_CONFIG)


class TestSessionClassification:
    """Test that timestamps are classified into the correct session."""

    def _utc(self, hour: int, minute: int = 0) -> datetime:
        """Helper: create a UTC datetime on a known Wednesday."""
        return datetime(2026, 3, 18, hour, minute, tzinfo=UTC)

    def _local(self, hour: int, minute: int = 0) -> datetime:
        """Helper: create an Israel-time datetime on a known Wednesday."""
        return datetime(2026, 3, 18, hour, minute, tzinfo=ISR)

    def test_asia_range(self, sf):
        # 03:00 Israel = inside Asia Range
        dt = self._local(3, 0)
        result = sf.check(dt)
        assert result.session == SessionName.ASIA
        assert result.trade_allowed is False
        assert result.is_kill_zone is False

    def test_london_kill_zone(self, sf):
        # 11:00 Israel = inside London KZ
        dt = self._local(11, 0)
        result = sf.check(dt)
        assert result.session == SessionName.LONDON
        assert result.trade_allowed is True
        assert result.is_kill_zone is True

    def test_ny_kill_zone_outside_overlap(self, sf):
        # 17:30 Israel = NY but past overlap end (17:00)
        dt = self._local(17, 30)
        result = sf.check(dt)
        assert result.session == SessionName.NY
        assert result.trade_allowed is True

    def test_overlap_takes_priority(self, sf):
        # 16:00 Israel = both NY and Overlap -> Overlap wins
        dt = self._local(16, 0)
        result = sf.check(dt)
        assert result.session == SessionName.OVERLAP
        assert result.trade_allowed is True
        assert result.priority == "highest"

    def test_overlap_start_boundary(self, sf):
        # 15:30 Israel = start of overlap
        dt = self._local(15, 30)
        result = sf.check(dt)
        assert result.session == SessionName.OVERLAP

    def test_off_session(self, sf):
        # 08:00 Israel = between Asia end (07:00) and London start (10:00)
        dt = self._local(8, 0)
        result = sf.check(dt)
        assert result.session == SessionName.OFF_SESSION
        assert result.trade_allowed is False

    def test_late_night_off_session(self, sf):
        # 23:00 Israel = no session
        dt = self._local(23, 0)
        result = sf.check(dt)
        assert result.session == SessionName.OFF_SESSION

    def test_between_london_and_ny(self, sf):
        # 14:00 Israel = gap between London (ends 13:00) and NY (starts 15:30)
        dt = self._local(14, 0)
        result = sf.check(dt)
        assert result.session == SessionName.OFF_SESSION


class TestTradeAllowed:
    def _local(self, hour: int, minute: int = 0) -> datetime:
        return datetime(2026, 3, 18, hour, minute, tzinfo=ISR)

    def test_trade_allowed_london(self, sf):
        assert sf.is_trade_allowed(self._local(11, 0)) is True

    def test_trade_blocked_asia(self, sf):
        assert sf.is_trade_allowed(self._local(4, 0)) is False

    def test_trade_blocked_off_session(self, sf):
        assert sf.is_trade_allowed(self._local(8, 0)) is False


class TestActiveSessions:
    def _local(self, hour: int, minute: int = 0) -> datetime:
        return datetime(2026, 3, 18, hour, minute, tzinfo=ISR)

    def test_overlap_returns_multiple(self, sf):
        active = sf.get_active_sessions(self._local(16, 0))
        assert SessionName.NY in active
        assert SessionName.OVERLAP in active

    def test_asia_only(self, sf):
        active = sf.get_active_sessions(self._local(3, 0))
        assert active == [SessionName.ASIA]


class TestDisabledSession:
    def test_disabled_london(self):
        config = dict(SESSIONS_CONFIG)
        config["london_kill_zone"] = {
            **config["london_kill_zone"],
            "enabled": False,
        }
        sf = SessionFilter(config)
        dt = datetime(2026, 3, 18, 11, 0, tzinfo=ISR)
        result = sf.check(dt)
        assert result.session == SessionName.OFF_SESSION


class TestUTCInput:
    """Verify that UTC timestamps are correctly converted to local time."""

    def test_utc_london_equivalent(self, sf):
        # Israel is UTC+2 in winter (March 18, 2026 is still winter time)
        # 09:00 UTC = 11:00 Israel → London KZ
        dt = datetime(2026, 3, 18, 9, 0, tzinfo=UTC)
        result = sf.check(dt)
        assert result.session == SessionName.LONDON
        assert result.trade_allowed is True


class TestSessionResultDict:
    def test_to_dict(self, sf):
        dt = datetime(2026, 3, 18, 11, 0, tzinfo=ISR)
        result = sf.check(dt)
        d = result.to_dict()
        assert d["session"] == "london"
        assert d["trade_allowed"] is True
        assert "local_time_str" in d


class TestDSTTransition:
    def test_near_dst_flag(self, sf):
        # Israel DST starts last Friday of March. 2026-03-27 is the transition.
        # 2026-03-25 should be within 3-day buffer.
        dt = datetime(2026, 3, 25, 11, 0, tzinfo=ISR)
        result = sf.check(dt)
        assert result.near_dst_transition is True

    def test_far_from_dst(self, sf):
        # 2026-06-15 — well into summer, no transition nearby
        dt = datetime(2026, 6, 15, 11, 0, tzinfo=ISR)
        result = sf.check(dt)
        assert result.near_dst_transition is False


class TestEdgeCases:
    def test_exact_session_end_is_excluded(self, sf):
        # 13:00 Israel = end of London (exclusive)
        dt = datetime(2026, 3, 18, 13, 0, tzinfo=ISR)
        result = sf.check(dt)
        assert result.session != SessionName.LONDON

    def test_exact_session_start_is_included(self, sf):
        # 10:00 Israel = start of London (inclusive)
        dt = datetime(2026, 3, 18, 10, 0, tzinfo=ISR)
        result = sf.check(dt)
        assert result.session == SessionName.LONDON

    def test_naive_datetime_treated_as_utc(self, sf):
        # Naive datetime → to_local assumes UTC
        dt = datetime(2026, 3, 18, 9, 0)
        result = sf.check(dt)
        # 09:00 UTC = 11:00 Israel = London
        assert result.session == SessionName.LONDON
