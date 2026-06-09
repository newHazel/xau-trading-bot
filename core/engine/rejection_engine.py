"""
Rejection Engine — Phase 4.3.

Records why a signal was rejected with full context. Every rejection
is stored for analysis — understanding why we *didn't* trade is as
important as understanding why we did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Rejection:
    setup_id: Optional[str]
    symbol: str
    timestamp: datetime
    attempted_direction: str
    main_reason: str
    failed_conditions: List[str]
    passed_conditions: List[str]
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "attempted_direction": self.attempted_direction,
            "main_reason": self.main_reason,
            "failed_conditions": self.failed_conditions,
            "passed_conditions": self.passed_conditions,
            "context": self.context,
        }


class RejectionEngine:
    """Creates and stores rejection records."""

    def __init__(self) -> None:
        self._rejections: List[Rejection] = []

    def reject(
        self,
        symbol: str,
        timestamp: datetime,
        attempted_direction: str,
        failed_conditions: List[str],
        passed_conditions: List[str],
        context: Optional[Dict[str, Any]] = None,
        setup_id: Optional[str] = None,
    ) -> Rejection:
        main_reason = failed_conditions[0] if failed_conditions else "unknown"
        rejection = Rejection(
            setup_id=setup_id,
            symbol=symbol,
            timestamp=timestamp,
            attempted_direction=attempted_direction,
            main_reason=main_reason,
            failed_conditions=failed_conditions,
            passed_conditions=passed_conditions,
            context=context or {},
        )
        self._rejections.append(rejection)
        return rejection

    @property
    def rejections(self) -> List[Rejection]:
        return list(self._rejections)

    @property
    def count(self) -> int:
        return len(self._rejections)

    def get_by_direction(self, direction: str) -> List[Rejection]:
        d = direction.strip().lower()
        return [r for r in self._rejections if r.attempted_direction.lower() == d]

    def get_recent(self, n: int = 10) -> List[Rejection]:
        return self._rejections[-n:]

    def reset(self) -> None:
        self._rejections.clear()
