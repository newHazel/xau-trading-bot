"""
System Status — Phase 9.3.

Tracks overall system status:
  HEALTHY → DEGRADED → ERROR → MAINTENANCE
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SystemState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass(frozen=True)
class StatusTransition:
    from_state: SystemState
    to_state: SystemState
    reason: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class SystemStatusReport:
    state: SystemState
    active_issues: List[str]
    last_transition: Optional[StatusTransition]
    uptime_minutes: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "active_issues": self.active_issues,
            "last_transition": self.last_transition.to_dict() if self.last_transition else None,
            "uptime_minutes": round(self.uptime_minutes, 1),
        }


class SystemStatusManager:
    """Tracks and manages system-wide status."""

    def __init__(self) -> None:
        self._state = SystemState.HEALTHY
        self._issues: List[str] = []
        self._transitions: List[StatusTransition] = []
        self._start_time: Optional[datetime] = None

    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def issues(self) -> List[str]:
        return list(self._issues)

    def start(self, now: Optional[datetime] = None) -> None:
        self._start_time = now or datetime.utcnow()

    def update_from_health(self, failed_count: int, warned_count: int, now: Optional[datetime] = None) -> SystemState:
        now = now or datetime.utcnow()

        if failed_count >= 3:
            self._set_state(SystemState.ERROR, f"{failed_count} health checks failed", now)
        elif failed_count >= 1:
            self._set_state(SystemState.DEGRADED, f"{failed_count} health checks failed", now)
        elif warned_count >= 2:
            self._set_state(SystemState.DEGRADED, f"{warned_count} health warnings", now)
        else:
            if self._state != SystemState.MAINTENANCE:
                self._set_state(SystemState.HEALTHY, "all checks passed", now)

        return self._state

    def set_maintenance(self, reason: str, now: Optional[datetime] = None) -> None:
        now = now or datetime.utcnow()
        self._set_state(SystemState.MAINTENANCE, reason, now)

    def clear_maintenance(self, now: Optional[datetime] = None) -> None:
        now = now or datetime.utcnow()
        self._set_state(SystemState.HEALTHY, "maintenance cleared", now)

    def add_issue(self, issue: str) -> None:
        if issue not in self._issues:
            self._issues.append(issue)

    def clear_issue(self, issue: str) -> None:
        if issue in self._issues:
            self._issues.remove(issue)

    def clear_all_issues(self) -> None:
        self._issues.clear()

    def get_report(self, now: Optional[datetime] = None) -> SystemStatusReport:
        now = now or datetime.utcnow()
        uptime = 0.0
        if self._start_time:
            uptime = (now - self._start_time).total_seconds() / 60

        last_transition = self._transitions[-1] if self._transitions else None

        return SystemStatusReport(
            state=self._state,
            active_issues=list(self._issues),
            last_transition=last_transition,
            uptime_minutes=uptime,
        )

    def _set_state(self, new_state: SystemState, reason: str, now: datetime) -> None:
        if new_state == self._state:
            return
        transition = StatusTransition(self._state, new_state, reason, now)
        self._transitions.append(transition)
        self._state = new_state
