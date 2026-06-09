"""Filter components — Phase 10.2: Shared filtering logic for dashboard pages."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class FilterConfig:
    grades: Optional[List[str]] = None
    directions: Optional[List[str]] = None
    statuses: Optional[List[str]] = None
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None
    min_r: Optional[float] = None
    max_r: Optional[float] = None

    def is_empty(self) -> bool:
        return all(v is None for v in [
            self.grades, self.directions, self.statuses,
            self.date_start, self.date_end, self.min_r, self.max_r,
        ])

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.grades is not None:
            d["grades"] = self.grades
        if self.directions is not None:
            d["directions"] = self.directions
        if self.statuses is not None:
            d["statuses"] = self.statuses
        if self.date_start is not None:
            d["date_start"] = self.date_start.isoformat()
        if self.date_end is not None:
            d["date_end"] = self.date_end.isoformat()
        if self.min_r is not None:
            d["min_r"] = self.min_r
        if self.max_r is not None:
            d["max_r"] = self.max_r
        return d


def apply_filters(records: List[Dict[str, Any]], config: FilterConfig) -> List[Dict[str, Any]]:
    if config.is_empty():
        return records

    result = records

    if config.grades is not None:
        result = [r for r in result if r.get("grade") in config.grades]

    if config.directions is not None:
        result = [r for r in result if r.get("direction") in config.directions]

    if config.statuses is not None:
        result = [r for r in result if r.get("status") in config.statuses]

    if config.date_start is not None:
        start_iso = config.date_start.isoformat()
        result = [r for r in result if r.get("timestamp", "") >= start_iso]

    if config.date_end is not None:
        end_iso = config.date_end.isoformat()
        result = [r for r in result if r.get("timestamp", "") <= end_iso]

    if config.min_r is not None:
        result = [r for r in result if (r.get("net_r") or 0) >= config.min_r]

    if config.max_r is not None:
        result = [r for r in result if (r.get("net_r") or 0) <= config.max_r]

    return result
