"""Signals Page — Phase 10.1: View and filter trading signals."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class SignalRow:
    setup_id: str
    timestamp: datetime
    direction: str
    grade: str
    entry: float
    sl: float
    tp1: float
    tp2: Optional[float]
    status: str
    net_r: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "grade": self.grade,
            "entry": self.entry,
            "sl": self.sl,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "status": self.status,
            "net_r": self.net_r,
        }


@dataclass
class SignalsPageData:
    signals: List[SignalRow] = field(default_factory=list)

    def add_signal(self, signal: SignalRow) -> None:
        self.signals.append(signal)

    @property
    def total(self) -> int:
        return len(self.signals)

    def filter_by_grade(self, grades: List[str]) -> List[SignalRow]:
        return [s for s in self.signals if s.grade in grades]

    def filter_by_direction(self, direction: str) -> List[SignalRow]:
        return [s for s in self.signals if s.direction == direction]

    def filter_by_status(self, status: str) -> List[SignalRow]:
        return [s for s in self.signals if s.status == status]

    def filter_by_date_range(self, start: datetime, end: datetime) -> List[SignalRow]:
        return [s for s in self.signals if start <= s.timestamp <= end]

    def get_summary(self) -> Dict[str, Any]:
        if not self.signals:
            return {"total": 0, "by_grade": {}, "by_direction": {}, "by_status": {}}
        by_grade: Dict[str, int] = {}
        by_direction: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        for s in self.signals:
            by_grade[s.grade] = by_grade.get(s.grade, 0) + 1
            by_direction[s.direction] = by_direction.get(s.direction, 0) + 1
            by_status[s.status] = by_status.get(s.status, 0) + 1
        return {
            "total": len(self.signals),
            "by_grade": by_grade,
            "by_direction": by_direction,
            "by_status": by_status,
        }

    def to_records(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.signals]


def render_signals(data: Optional[SignalsPageData] = None) -> Dict[str, Any]:
    data = data or SignalsPageData()
    return {
        "page": "Signals",
        "summary": data.get_summary(),
        "records": data.to_records(),
    }
