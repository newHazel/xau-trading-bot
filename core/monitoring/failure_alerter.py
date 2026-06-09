"""
Failure Alerter — Phase 9.5.

Alerts on failures with retry logic:
  - Retry 3x before giving up
  - Log all errors
  - Send to Telegram when configured
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AlertRecord:
    alert_id: int
    severity: str  # critical / warning / info
    source: str
    message: str
    timestamp: datetime
    retries: int = 0
    max_retries: int = 3
    resolved: bool = False
    sent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity,
            "source": self.source,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "retries": self.retries,
            "resolved": self.resolved,
            "sent": self.sent,
        }


SendFn = Callable[[str], bool]


class FailureAlerter:
    """Manages failure alerts with retry logic."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._max_retries = config.get("retry_count", 3)
        self._send_to_telegram = config.get("send_to_telegram", True)
        self._alerts: List[AlertRecord] = []
        self._counter = 0
        self._send_fn: Optional[SendFn] = None

    def set_send_function(self, fn: SendFn) -> None:
        self._send_fn = fn

    @property
    def alerts(self) -> List[AlertRecord]:
        return list(self._alerts)

    @property
    def unresolved(self) -> List[AlertRecord]:
        return [a for a in self._alerts if not a.resolved]

    def alert(
        self,
        severity: str,
        source: str,
        message: str,
        now: Optional[datetime] = None,
    ) -> AlertRecord:
        now = now or datetime.utcnow()
        self._counter += 1

        record = AlertRecord(
            alert_id=self._counter,
            severity=severity,
            source=source,
            message=message,
            timestamp=now,
            max_retries=self._max_retries,
        )
        self._alerts.append(record)

        if self._send_to_telegram and self._send_fn:
            self._try_send(record)

        return record

    def retry_unsent(self) -> List[AlertRecord]:
        retried: List[AlertRecord] = []
        for a in self._alerts:
            if not a.sent and not a.resolved and a.retries < a.max_retries:
                if self._send_fn:
                    self._try_send(a)
                    retried.append(a)
        return retried

    def resolve(self, alert_id: int) -> Optional[AlertRecord]:
        for a in self._alerts:
            if a.alert_id == alert_id:
                a.resolved = True
                return a
        return None

    def get_by_severity(self, severity: str) -> List[AlertRecord]:
        return [a for a in self._alerts if a.severity == severity]

    def get_recent(self, count: int = 10) -> List[AlertRecord]:
        return list(reversed(self._alerts[-count:]))

    def clear(self) -> None:
        self._alerts.clear()
        self._counter = 0

    def _try_send(self, record: AlertRecord) -> None:
        record.retries += 1
        try:
            text = f"[{record.severity.upper()}] {record.source}: {record.message}"
            success = self._send_fn(text)
            if success:
                record.sent = True
        except Exception:
            pass
