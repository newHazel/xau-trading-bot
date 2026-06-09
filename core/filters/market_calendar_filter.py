"""
Market Calendar Filter — Phase 3.2.

Blocks trading during periods the market is closed or unreliable:
  - Weekends (Saturday–Sunday)
  - Holidays from manual_holidays.csv
  - Monday open cooldown (first N minutes after market opens)
  - Friday late block (after configurable cutoff)
  - Gap cooldown (after a price/weekend gap is detected)

All thresholds come from config/market_calendar.yaml.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from zoneinfo import ZoneInfo

from core.utils.time_utils import parse_time, to_local

logger = logging.getLogger(__name__)


class CalendarBlockReason(str, Enum):
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    MONDAY_COOLDOWN = "monday_cooldown"
    FRIDAY_LATE = "friday_late"
    GAP_COOLDOWN = "gap_cooldown"
    CALENDAR_DISABLED = "calendar_disabled"


@dataclass(frozen=True)
class CalendarResult:
    trade_allowed: bool
    block_reason: Optional[CalendarBlockReason]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_allowed": self.trade_allowed,
            "block_reason": self.block_reason.value if self.block_reason else None,
            "detail": self.detail,
        }


class MarketCalendarFilter:
    """Checks whether the market is open and trading is permitted."""

    def __init__(
        self,
        config: Dict[str, Any],
        project_root: Optional[Path] = None,
    ) -> None:
        self._enabled = config.get("enabled", True)
        self._tz = ZoneInfo(config.get("timezone", "Asia/Jerusalem"))
        self._block_weekends = config.get("block_weekends", True)
        self._monday_cooldown_min = config.get("monday_open_cooldown_minutes", 120)
        self._friday_late_block = config.get("friday_late_close_block", True)
        self._friday_block_after = parse_time(config.get("friday_block_after", "21:00"))

        gap_cfg = config.get("gap_detection", {})
        self._gap_cooldown_min = gap_cfg.get("cooldown_after_gap_minutes", 60)
        self._weekend_gap_cooldown_min = gap_cfg.get("weekend_gap_cooldown_minutes", 120)

        self._holidays: Set[date] = set()
        self._holiday_names: Dict[date, str] = {}
        if config.get("use_manual_holidays_csv", False):
            csv_path = config.get("manual_holidays_path", "data/calendar/manual_holidays.csv")
            if project_root:
                csv_path = str(project_root / csv_path)
            self._load_holidays(csv_path)

        self._last_gap_time: Optional[datetime] = None
        self._last_gap_is_weekend: bool = False

    def check(self, dt: datetime) -> CalendarResult:
        if not self._enabled:
            return CalendarResult(True, None, "calendar filter disabled")

        local_dt = to_local(dt, self._tz)

        if self._block_weekends and local_dt.weekday() in (5, 6):
            return CalendarResult(
                False,
                CalendarBlockReason.WEEKEND,
                f"weekend ({local_dt.strftime('%A')})",
            )

        if local_dt.date() in self._holidays:
            name = self._holiday_names.get(local_dt.date(), "unknown holiday")
            return CalendarResult(
                False,
                CalendarBlockReason.HOLIDAY,
                f"holiday: {name}",
            )

        if local_dt.weekday() == 0:
            monday_start = local_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            minutes_since = (local_dt - monday_start).total_seconds() / 60
            if minutes_since < self._monday_cooldown_min:
                remaining = int(self._monday_cooldown_min - minutes_since)
                return CalendarResult(
                    False,
                    CalendarBlockReason.MONDAY_COOLDOWN,
                    f"monday cooldown, {remaining}min remaining",
                )

        if self._friday_late_block and local_dt.weekday() == 4:
            if local_dt.time() >= self._friday_block_after:
                return CalendarResult(
                    False,
                    CalendarBlockReason.FRIDAY_LATE,
                    f"friday late block after {self._friday_block_after}",
                )

        if self._last_gap_time is not None:
            cooldown = (
                self._weekend_gap_cooldown_min
                if self._last_gap_is_weekend
                else self._gap_cooldown_min
            )
            gap_local = to_local(self._last_gap_time, self._tz)
            elapsed = (local_dt - gap_local).total_seconds() / 60
            if elapsed < cooldown:
                remaining = int(cooldown - elapsed)
                return CalendarResult(
                    False,
                    CalendarBlockReason.GAP_COOLDOWN,
                    f"gap cooldown, {remaining}min remaining",
                )

        return CalendarResult(True, None, "market open")

    def is_trade_allowed(self, dt: datetime) -> bool:
        return self.check(dt).trade_allowed

    def register_gap(self, gap_time: datetime, is_weekend_gap: bool = False) -> None:
        self._last_gap_time = gap_time
        self._last_gap_is_weekend = is_weekend_gap

    def clear_gap(self) -> None:
        self._last_gap_time = None
        self._last_gap_is_weekend = False

    def is_holiday(self, d: date) -> bool:
        return d in self._holidays

    def get_holidays(self) -> Dict[date, str]:
        return dict(self._holiday_names)

    def _load_holidays(self, csv_path: str) -> None:
        path = Path(csv_path)
        if not path.exists():
            logger.warning("[MarketCalendar] Holiday CSV not found: %s", csv_path)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    closed = row.get("market_closed", "true").strip().lower()
                    if closed == "true":
                        d = date.fromisoformat(row["date"].strip())
                        self._holidays.add(d)
                        self._holiday_names[d] = row.get("name", "").strip()
            logger.info(
                "[MarketCalendar] Loaded %d holidays from %s",
                len(self._holidays), csv_path,
            )
        except Exception as e:
            logger.error("[MarketCalendar] Failed to load holidays: %s", e)
