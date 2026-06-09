"""DST Handler — detects DST transitions and applies buffer logic."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any

from zoneinfo import ZoneInfo


class DSTHandler:
    """
    Tracks DST transitions for a target timezone and flags dates
    that fall within the configured buffer window around a transition.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        dst_cfg = config.get("dst", {})
        self._local_tz = ZoneInfo(config.get("timezone", "Asia/Jerusalem"))
        self._buffer_days = dst_cfg.get("dst_transition_buffer_days", 3)
        self._handle_transitions = dst_cfg.get("handle_dst_transitions", True)

    @property
    def local_tz(self) -> ZoneInfo:
        return self._local_tz

    def utc_offset_hours(self, dt: datetime) -> float:
        local_dt = dt.astimezone(self._local_tz)
        return local_dt.utcoffset().total_seconds() / 3600

    def is_dst_active(self, dt: datetime) -> bool:
        local_dt = dt.astimezone(self._local_tz)
        return bool(local_dt.dst())

    def is_near_dst_transition(self, dt: datetime) -> bool:
        if not self._handle_transitions:
            return False
        local_dt = dt.astimezone(self._local_tz)
        for delta_days in range(-self._buffer_days, self._buffer_days + 1):
            check = local_dt + timedelta(days=delta_days)
            if self._dst_differs(local_dt, check):
                return True
        return False

    def _dst_differs(self, dt1: datetime, dt2: datetime) -> bool:
        d1 = dt1.astimezone(self._local_tz)
        d2 = dt2.astimezone(self._local_tz)
        return bool(d1.dst()) != bool(d2.dst())

    def adjust_session_times(
        self, session_start_str: str, session_end_str: str, reference_dt: datetime
    ) -> tuple[str, str]:
        """
        Return session start/end as-is — times in sessions.yaml are already
        in local timezone. This method exists as a hook for future adjustments
        if broker times diverge from local during DST transitions.
        """
        return session_start_str, session_end_str
