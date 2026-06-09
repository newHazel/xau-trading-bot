"""
Paper Journal — Phase 7.3.

Per-signal log with:
  - All conditions checked (mandatory + optional)
  - Entry/SL/TP levels
  - Result (win/loss/breakeven)
  - Net R multiple
  - Notes (manual or auto-generated)
  - Violations tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class JournalEntry:
    entry_id: int
    setup_id: str
    direction: str
    grade: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_type: Optional[str] = None
    net_r: float = 0.0
    gross_r: float = 0.0
    net_pnl: float = 0.0
    mandatory_conditions: Dict[str, bool] = field(default_factory=dict)
    optional_scores: Dict[str, int] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    notes: str = ""
    result: str = "pending"  # pending / win / loss / breakeven

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "setup_id": self.setup_id,
            "direction": self.direction,
            "grade": self.grade,
            "entry_price": round(self.entry_price, 2),
            "sl_price": round(self.sl_price, 2),
            "tp1_price": round(self.tp1_price, 2),
            "tp2_price": round(self.tp2_price, 2),
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": round(self.exit_price, 2) if self.exit_price else None,
            "exit_type": self.exit_type,
            "net_r": round(self.net_r, 3),
            "gross_r": round(self.gross_r, 3),
            "net_pnl": round(self.net_pnl, 2),
            "result": self.result,
            "violations": self.violations,
            "notes": self.notes,
        }


class PaperJournal:
    """Records and queries paper trading journal entries."""

    def __init__(self) -> None:
        self._entries: List[JournalEntry] = []
        self._counter = 0

    @property
    def entries(self) -> List[JournalEntry]:
        return list(self._entries)

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    def add_entry(
        self,
        setup_id: str,
        direction: str,
        grade: str,
        entry_price: float,
        sl_price: float,
        tp1_price: float,
        tp2_price: float,
        entry_time: datetime,
        mandatory_conditions: Optional[Dict[str, bool]] = None,
        optional_scores: Optional[Dict[str, int]] = None,
        notes: str = "",
    ) -> JournalEntry:
        self._counter += 1
        entry = JournalEntry(
            entry_id=self._counter,
            setup_id=setup_id,
            direction=direction,
            grade=grade,
            entry_price=entry_price,
            sl_price=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            entry_time=entry_time,
            mandatory_conditions=mandatory_conditions or {},
            optional_scores=optional_scores or {},
            notes=notes,
        )
        self._entries.append(entry)
        return entry

    def close_entry(
        self,
        entry_id: int,
        exit_price: float,
        exit_time: datetime,
        exit_type: str,
        net_r: float,
        gross_r: float,
        net_pnl: float = 0.0,
        notes: str = "",
    ) -> Optional[JournalEntry]:
        entry = self._find(entry_id)
        if entry is None:
            return None

        entry.exit_price = exit_price
        entry.exit_time = exit_time
        entry.exit_type = exit_type
        entry.net_r = net_r
        entry.gross_r = gross_r
        entry.net_pnl = net_pnl

        if net_r > 0.1:
            entry.result = "win"
        elif net_r < -0.1:
            entry.result = "loss"
        else:
            entry.result = "breakeven"

        if notes:
            entry.notes = f"{entry.notes}; {notes}" if entry.notes else notes

        return entry

    def add_violation(self, entry_id: int, violation: str) -> None:
        entry = self._find(entry_id)
        if entry:
            entry.violations.append(violation)

    def get_open(self) -> List[JournalEntry]:
        return [e for e in self._entries if e.result == "pending"]

    def get_closed(self) -> List[JournalEntry]:
        return [e for e in self._entries if e.result != "pending"]

    def get_violations(self) -> List[JournalEntry]:
        return [e for e in self._entries if e.violations]

    def get_by_grade(self, grade: str) -> List[JournalEntry]:
        return [e for e in self._entries if e.grade == grade]

    def get_by_direction(self, direction: str) -> List[JournalEntry]:
        return [e for e in self._entries if e.direction == direction.strip().lower()]

    def _find(self, entry_id: int) -> Optional[JournalEntry]:
        for e in self._entries:
            if e.entry_id == entry_id:
                return e
        return None

    def clear(self) -> None:
        self._entries.clear()
        self._counter = 0
