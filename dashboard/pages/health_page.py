"""Health Page — Phase 10.1: System health, heartbeat, alerts overview."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class HealthCheckRow:
    name: str
    status: str  # "pass" | "fail" | "warn" | "skip"
    message: str
    checked_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class AlertRow:
    alert_id: int
    severity: str
    source: str
    message: str
    timestamp: datetime
    resolved: bool
    sent: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity,
            "source": self.source,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "sent": self.sent,
        }


@dataclass
class HealthPageData:
    system_state: str = "healthy"
    uptime_minutes: float = 0.0
    checks: List[HealthCheckRow] = field(default_factory=list)
    alerts: List[AlertRow] = field(default_factory=list)
    last_heartbeat: Optional[datetime] = None
    active_issues: List[str] = field(default_factory=list)

    def add_check(self, check: HealthCheckRow) -> None:
        self.checks.append(check)

    def add_alert(self, alert: AlertRow) -> None:
        self.alerts.append(alert)

    @property
    def failed_checks(self) -> List[HealthCheckRow]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warned_checks(self) -> List[HealthCheckRow]:
        return [c for c in self.checks if c.status == "warn"]

    @property
    def unresolved_alerts(self) -> List[AlertRow]:
        return [a for a in self.alerts if not a.resolved]

    def get_summary(self) -> Dict[str, Any]:
        return {
            "system_state": self.system_state,
            "uptime_minutes": round(self.uptime_minutes, 1),
            "total_checks": len(self.checks),
            "failed_checks": len(self.failed_checks),
            "warned_checks": len(self.warned_checks),
            "total_alerts": len(self.alerts),
            "unresolved_alerts": len(self.unresolved_alerts),
            "active_issues": self.active_issues,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
        }

    def checks_to_records(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.checks]

    def alerts_to_records(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.alerts]


def render_health(data: Optional[HealthPageData] = None) -> Dict[str, Any]:
    data = data or HealthPageData()
    return {
        "page": "Health",
        "summary": data.get_summary(),
        "checks": data.checks_to_records(),
        "alerts": data.alerts_to_records(),
    }
