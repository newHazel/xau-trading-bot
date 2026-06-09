"""
Heartbeat — Phase 9.2.

Sends periodic status to Telegram:
  - Uptime
  - Last signal time + setup_id
  - Current state machine state
  - Quick health summary
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class HeartbeatMessage:
    uptime_minutes: float
    last_signal_time: Optional[datetime]
    last_signal_id: Optional[str]
    current_state: str
    health_summary: str
    trades_today: int
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uptime_minutes": round(self.uptime_minutes, 1),
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
            "last_signal_id": self.last_signal_id,
            "current_state": self.current_state,
            "health_summary": self.health_summary,
            "trades_today": self.trades_today,
            "timestamp": self.timestamp.isoformat(),
        }

    def format_telegram(self) -> str:
        lines = [
            f"Heartbeat | {self.timestamp.strftime('%H:%M UTC')}",
            f"Uptime: {self.uptime_minutes:.0f}min",
            f"State: {self.current_state}",
            f"Trades today: {self.trades_today}",
            f"Health: {self.health_summary}",
        ]
        if self.last_signal_id:
            lines.append(f"Last signal: {self.last_signal_id}")
        return "\n".join(lines)


class HeartbeatManager:
    """Manages periodic heartbeat generation."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._interval_minutes = config.get("interval_minutes", 60)
        self._enabled = config.get("enabled", True)
        self._start_time: Optional[datetime] = None
        self._last_heartbeat: Optional[datetime] = None
        self._last_signal_time: Optional[datetime] = None
        self._last_signal_id: Optional[str] = None

    def start(self, now: Optional[datetime] = None) -> None:
        self._start_time = now or datetime.utcnow()

    def update_signal(self, signal_id: str, timestamp: datetime) -> None:
        self._last_signal_id = signal_id
        self._last_signal_time = timestamp

    def is_due(self, now: Optional[datetime] = None) -> bool:
        if not self._enabled:
            return False
        now = now or datetime.utcnow()
        if self._last_heartbeat is None:
            return True
        elapsed = (now - self._last_heartbeat).total_seconds() / 60
        return elapsed >= self._interval_minutes

    def generate(
        self,
        current_state: str,
        health_summary: str,
        trades_today: int,
        now: Optional[datetime] = None,
    ) -> HeartbeatMessage:
        now = now or datetime.utcnow()
        uptime = 0.0
        if self._start_time:
            uptime = (now - self._start_time).total_seconds() / 60

        self._last_heartbeat = now

        return HeartbeatMessage(
            uptime_minutes=uptime,
            last_signal_time=self._last_signal_time,
            last_signal_id=self._last_signal_id,
            current_state=current_state,
            health_summary=health_summary,
            trades_today=trades_today,
            timestamp=now,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def interval_minutes(self) -> int:
        return self._interval_minutes
