"""Tests for Market Calendar Filter — Phase 3.2."""

import pytest
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from core.filters.market_calendar_filter import (
    CalendarBlockReason,
    MarketCalendarFilter,
)

ISR = ZoneInfo("Asia/Jerusalem")
UTC = ZoneInfo("Etc/UTC")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CALENDAR_CONFIG = {
    "enabled": True,
    "timezone": "Asia/Jerusalem",
    "block_weekends": True,
    "monday_open_cooldown_minutes": 120,
    "friday_late_close_block": True,
    "friday_block_after": "21:00",
    "use_manual_holidays_csv": True,
    "manual_holidays_path": "data/calendar/manual_holidays.csv",
    "gap_detection": {
        "enabled": True,
        "cooldown_after_gap_minutes": 60,
        "weekend_gap_cooldown_minutes": 120,
    },
}


@pytest.fixture
def mcf():
    return MarketCalendarFilter(CALENDAR_CONFIG, project_root=PROJECT_ROOT)


class TestWeekendBlocking:
    def test_saturday_blocked(self, mcf):
        # 2026-03-21 is Saturday
        dt = datetime(2026, 3, 21, 12, 0, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is False
        assert result.block_reason == CalendarBlockReason.WEEKEND

    def test_sunday_blocked(self, mcf):
        # 2026-03-22 is Sunday
        dt = datetime(2026, 3, 22, 12, 0, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is False
        assert result.block_reason == CalendarBlockReason.WEEKEND

    def test_wednesday_allowed(self, mcf):
        # 2026-03-18 is Wednesday
        dt = datetime(2026, 3, 18, 12, 0, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is True

    def test_weekend_blocking_disabled(self):
        config = {**CALENDAR_CONFIG, "block_weekends": False}
        mcf = MarketCalendarFilter(config, project_root=PROJECT_ROOT)
        dt = datetime(2026, 3, 21, 12, 0, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is True


class TestHolidays:
    def test_new_years_blocked(self, mcf):
        # 2026-01-01 is in manual_holidays.csv
        dt = datetime(2026, 1, 1, 12, 0, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is False
        assert result.block_reason == CalendarBlockReason.HOLIDAY
        assert "New Year" in result.detail

    def test_christmas_blocked(self, mcf):
        dt = datetime(2026, 12, 25, 12, 0, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is False
        assert result.block_reason == CalendarBlockReason.HOLIDAY

    def test_regular_day_not_holiday(self, mcf):
        dt = datetime(2026, 3, 18, 12, 0, tzinfo=ISR)
        assert mcf.is_holiday(date(2026, 3, 18)) is False

    def test_is_holiday_api(self, mcf):
        assert mcf.is_holiday(date(2026, 1, 1)) is True

    def test_get_holidays_dict(self, mcf):
        holidays = mcf.get_holidays()
        assert date(2026, 1, 1) in holidays
        assert len(holidays) >= 9


class TestMondayCooldown:
    def test_monday_early_morning_blocked(self, mcf):
        # 2026-03-23 is Monday, 01:00 local → within 120min cooldown
        dt = datetime(2026, 3, 23, 1, 0, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is False
        assert result.block_reason == CalendarBlockReason.MONDAY_COOLDOWN

    def test_monday_after_cooldown(self, mcf):
        # 2026-03-23 Monday 03:00 → 180min since midnight > 120min
        dt = datetime(2026, 3, 23, 3, 0, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is True

    def test_monday_exact_cooldown_boundary(self, mcf):
        # Exactly at 02:00 (120min) → cooldown expired
        dt = datetime(2026, 3, 23, 2, 0, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is True

    def test_tuesday_no_cooldown(self, mcf):
        # 2026-03-24 is Tuesday
        dt = datetime(2026, 3, 24, 0, 30, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is True


class TestFridayLateBlock:
    def test_friday_before_cutoff(self, mcf):
        # 2026-03-20 is Friday, 20:00 → before 21:00
        dt = datetime(2026, 3, 20, 20, 0, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is True

    def test_friday_after_cutoff(self, mcf):
        # 2026-03-20 Friday, 21:30 → after 21:00
        dt = datetime(2026, 3, 20, 21, 30, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is False
        assert result.block_reason == CalendarBlockReason.FRIDAY_LATE

    def test_friday_exact_cutoff(self, mcf):
        # Exactly 21:00 → blocked (>=)
        dt = datetime(2026, 3, 20, 21, 0, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is False

    def test_friday_late_block_disabled(self):
        config = {**CALENDAR_CONFIG, "friday_late_close_block": False}
        mcf = MarketCalendarFilter(config, project_root=PROJECT_ROOT)
        dt = datetime(2026, 3, 20, 22, 0, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is True


class TestGapCooldown:
    def test_gap_cooldown_active(self, mcf):
        gap_time = datetime(2026, 3, 18, 10, 0, tzinfo=ISR)
        mcf.register_gap(gap_time, is_weekend_gap=False)
        # 30 minutes later → still in 60min cooldown
        check_time = datetime(2026, 3, 18, 10, 30, tzinfo=ISR)
        result = mcf.check(check_time)
        assert result.trade_allowed is False
        assert result.block_reason == CalendarBlockReason.GAP_COOLDOWN

    def test_gap_cooldown_expired(self, mcf):
        gap_time = datetime(2026, 3, 18, 10, 0, tzinfo=ISR)
        mcf.register_gap(gap_time, is_weekend_gap=False)
        # 65 minutes later → past 60min cooldown
        check_time = datetime(2026, 3, 18, 11, 5, tzinfo=ISR)
        result = mcf.check(check_time)
        assert result.trade_allowed is True

    def test_weekend_gap_longer_cooldown(self, mcf):
        gap_time = datetime(2026, 3, 23, 0, 30, tzinfo=ISR)  # Monday
        mcf.register_gap(gap_time, is_weekend_gap=True)
        # 90 minutes later → still in 120min weekend cooldown
        check_time = datetime(2026, 3, 23, 2, 0, tzinfo=ISR)
        # Note: also blocked by monday cooldown, but let's check gap too
        mcf._monday_cooldown_min = 0  # disable monday for this test
        result = mcf.check(check_time)
        assert result.trade_allowed is False
        assert result.block_reason == CalendarBlockReason.GAP_COOLDOWN

    def test_weekend_gap_cooldown_expired(self, mcf):
        gap_time = datetime(2026, 3, 23, 0, 30, tzinfo=ISR)
        mcf.register_gap(gap_time, is_weekend_gap=True)
        mcf._monday_cooldown_min = 0
        # 130 minutes later → past 120min
        check_time = datetime(2026, 3, 23, 2, 40, tzinfo=ISR)
        result = mcf.check(check_time)
        assert result.trade_allowed is True

    def test_clear_gap(self, mcf):
        gap_time = datetime(2026, 3, 18, 10, 0, tzinfo=ISR)
        mcf.register_gap(gap_time)
        mcf.clear_gap()
        check_time = datetime(2026, 3, 18, 10, 5, tzinfo=ISR)
        result = mcf.check(check_time)
        assert result.trade_allowed is True


class TestCalendarDisabled:
    def test_everything_passes_when_disabled(self):
        config = {**CALENDAR_CONFIG, "enabled": False}
        mcf = MarketCalendarFilter(config, project_root=PROJECT_ROOT)
        # Saturday should pass when disabled
        dt = datetime(2026, 3, 21, 12, 0, tzinfo=ISR)
        result = mcf.check(dt)
        assert result.trade_allowed is True


class TestMissingHolidayCSV:
    def test_no_crash_on_missing_csv(self):
        config = {
            **CALENDAR_CONFIG,
            "manual_holidays_path": "data/calendar/nonexistent.csv",
        }
        mcf = MarketCalendarFilter(config, project_root=PROJECT_ROOT)
        dt = datetime(2026, 1, 1, 12, 0, tzinfo=ISR)
        # No holidays loaded → day is allowed
        result = mcf.check(dt)
        assert result.trade_allowed is True


class TestResultDict:
    def test_blocked_to_dict(self, mcf):
        dt = datetime(2026, 3, 21, 12, 0, tzinfo=ISR)
        d = mcf.check(dt).to_dict()
        assert d["trade_allowed"] is False
        assert d["block_reason"] == "weekend"

    def test_allowed_to_dict(self, mcf):
        dt = datetime(2026, 3, 18, 12, 0, tzinfo=ISR)
        d = mcf.check(dt).to_dict()
        assert d["trade_allowed"] is True
        assert d["block_reason"] is None


class TestUTCInput:
    def test_utc_saturday_blocked(self, mcf):
        # Saturday in UTC is also Saturday in Israel
        dt = datetime(2026, 3, 21, 10, 0, tzinfo=UTC)
        result = mcf.check(dt)
        assert result.trade_allowed is False
        assert result.block_reason == CalendarBlockReason.WEEKEND
