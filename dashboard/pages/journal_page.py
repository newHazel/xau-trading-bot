"""Journal Page — Phase 10.1: View paper/live trade journal entries."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class JournalEntry:
    setup_id: str
    timestamp: datetime
    direction: str
    grade: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: Optional[float]
    result: str  # "win" | "loss" | "breakeven" | "open"
    net_r: Optional[float] = None
    notes: str = ""
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "grade": self.grade,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp1_price": self.tp1_price,
            "tp2_price": self.tp2_price,
            "result": self.result,
            "net_r": self.net_r,
            "notes": self.notes,
            "violations": self.violations,
        }


@dataclass
class JournalPageData:
    entries: List[JournalEntry] = field(default_factory=list)

    def add_entry(self, entry: JournalEntry) -> None:
        self.entries.append(entry)

    @property
    def total(self) -> int:
        return len(self.entries)

    def filter_by_result(self, result: str) -> List[JournalEntry]:
        return [e for e in self.entries if e.result == result]

    def filter_by_grade(self, grades: List[str]) -> List[JournalEntry]:
        return [e for e in self.entries if e.grade in grades]

    def filter_by_direction(self, direction: str) -> List[JournalEntry]:
        return [e for e in self.entries if e.direction == direction]

    def filter_by_date_range(self, start: datetime, end: datetime) -> List[JournalEntry]:
        return [e for e in self.entries if start <= e.timestamp <= end]

    def get_with_violations(self) -> List[JournalEntry]:
        return [e for e in self.entries if e.violations]

    def get_summary(self) -> Dict[str, Any]:
        if not self.entries:
            return {"total": 0, "by_result": {}, "violation_count": 0}
        by_result: Dict[str, int] = {}
        for e in self.entries:
            by_result[e.result] = by_result.get(e.result, 0) + 1
        closed = [e for e in self.entries if e.net_r is not None]
        avg_r = round(sum(e.net_r for e in closed) / len(closed), 4) if closed else 0.0
        return {
            "total": len(self.entries),
            "by_result": by_result,
            "avg_net_r": avg_r,
            "violation_count": len(self.get_with_violations()),
        }

    def to_records(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.entries]


def render_journal(data: Optional[JournalPageData] = None) -> Dict[str, Any]:
    data = data or JournalPageData()
    return {
        "page": "Journal",
        "summary": data.get_summary(),
        "records": data.to_records(),
    }
