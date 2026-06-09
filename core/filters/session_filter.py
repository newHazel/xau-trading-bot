"""
Session / Kill Zone Filter — Phase 3.1.

Determines which trading session a given timestamp falls into and whether
trading is allowed at that time. Sessions are defined in config/sessions.yaml
with times in Asia/Jerusalem timezone.

Sessions (Israel time):
  - Asia Range:      02:00–07:00  (mark only, no trade)
  - London Kill Zone: 10:00–13:00
  - NY Kill Zone:     15:30–18:00
  - Overlap:          15:30–17:00  (highest priority)

Rules:
  - Trade only inside enabled Kill Zones where trade_allowed is True.
  - Overlap takes priority over NY when both match.
  - Asia range is marking only (high/low) — never trade.
  - DST transitions: buffer days flag caution but don't block.
  - Outside all sessions → no trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

from core.utils.dst_handler import DSTHandler
from core.utils.time_utils import parse_time, time_in_range, to_local


class SessionName(str, Enum):
    ASIA = "asia"
    LONDON = "london"
    NY = "ny"
    OVERLAP = "overlap"
    OFF_SESSION = "off_session"


@dataclass(frozen=True)
class SessionResult:
    session: SessionName
    trade_allowed: bool
    is_kill_zone: bool
    near_dst_transition: bool
    priority: str
    local_time_str: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session": self.session.value,
            "trade_allowed": self.trade_allowed,
            "is_kill_zone": self.is_kill_zone,
            "near_dst_transition": self.near_dst_transition,
            "priority": self.priority,
            "local_time_str": self.local_time_str,
        }


@dataclass
class _SessionDef:
    name: SessionName
    start: time
    end: time
    enabled: bool
    trade_allowed: bool
    priority: str


class SessionFilter:
    """Classifies timestamps into trading sessions and checks trade permission."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._tz = ZoneInfo(config.get("timezone", "Asia/Jerusalem"))
        self._dst = DSTHandler(config)
        self._sessions = self._build_sessions(config)

    def check(self, dt: datetime) -> SessionResult:
        local_dt = to_local(dt, self._tz)
        local_t = local_dt.time()
        near_dst = self._dst.is_near_dst_transition(dt)
        local_time_str = local_dt.strftime("%H:%M:%S")

        matched = self._match_sessions(local_t)

        if not matched:
            return SessionResult(
                session=SessionName.OFF_SESSION,
                trade_allowed=False,
                is_kill_zone=False,
                near_dst_transition=near_dst,
                priority="none",
                local_time_str=local_time_str,
            )

        best = self._pick_highest_priority(matched)

        return SessionResult(
            session=best.name,
            trade_allowed=best.trade_allowed,
            is_kill_zone=best.trade_allowed,
            near_dst_transition=near_dst,
            priority=best.priority,
            local_time_str=local_time_str,
        )

    def is_trade_allowed(self, dt: datetime) -> bool:
        return self.check(dt).trade_allowed

    def get_active_sessions(self, dt: datetime) -> List[SessionName]:
        local_dt = to_local(dt, self._tz)
        local_t = local_dt.time()
        return [s.name for s in self._match_sessions(local_t)]

    def _match_sessions(self, local_t: time) -> List[_SessionDef]:
        return [
            s for s in self._sessions
            if s.enabled and time_in_range(local_t, s.start, s.end)
        ]

    @staticmethod
    def _pick_highest_priority(sessions: List[_SessionDef]) -> _SessionDef:
        priority_order = {"highest": 0, "high": 1, "normal": 2, "low": 3}
        return min(sessions, key=lambda s: priority_order.get(s.priority, 99))

    def _build_sessions(self, config: Dict[str, Any]) -> List[_SessionDef]:
        defs: List[_SessionDef] = []

        mapping = [
            ("asia_range", SessionName.ASIA, "low"),
            ("london_kill_zone", SessionName.LONDON, "normal"),
            ("ny_kill_zone", SessionName.NY, "normal"),
            ("overlap", SessionName.OVERLAP, "highest"),
        ]

        for key, name, default_priority in mapping:
            section = config.get(key, {})
            if not section:
                continue
            defs.append(_SessionDef(
                name=name,
                start=parse_time(section["start"]),
                end=parse_time(section["end"]),
                enabled=section.get("enabled", True),
                trade_allowed=section.get("trade_allowed", False),
                priority=section.get("priority", default_priority),
            ))

        return defs
