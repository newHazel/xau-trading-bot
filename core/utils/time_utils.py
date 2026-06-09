"""Time utilities — timezone conversion and timestamp helpers."""

from __future__ import annotations

from datetime import datetime, time
from typing import Optional

import pytz
from zoneinfo import ZoneInfo

BROKER_TZ = ZoneInfo("Etc/UTC")
LOCAL_TZ = ZoneInfo("Asia/Jerusalem")


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=BROKER_TZ)
    return dt.astimezone(BROKER_TZ)


def to_local(dt: datetime, tz: ZoneInfo = LOCAL_TZ) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=BROKER_TZ)
    return dt.astimezone(tz)


def make_aware(dt: datetime, tz: ZoneInfo = BROKER_TZ) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt


def time_in_range(t: time, start: time, end: time) -> bool:
    """Check if time t falls within [start, end). Handles overnight ranges."""
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def parse_time(s: str) -> time:
    parts = s.strip().split(":")
    return time(int(parts[0]), int(parts[1]))
