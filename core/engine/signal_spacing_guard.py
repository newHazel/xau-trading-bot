"""
Signal Spacing Guard — Phase 4.7.

Prevents over-trading by enforcing minimum time and price distance
between signals in the same direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class _RecentSignal:
    direction: str
    entry_price: float
    timestamp: datetime


class SignalSpacingGuard:
    """Enforces minimum spacing between signals."""

    def __init__(self, config: Dict[str, Any]) -> None:
        spacing = config.get("signal_spacing", {})
        self._min_minutes = spacing.get("min_minutes_between_signals_same_direction", 30)
        self._min_atr_distance = spacing.get("min_atr_distance_between_entries", 1.0)
        self._recent: List[_RecentSignal] = []

    def can_send(
        self,
        direction: str,
        entry_price: float,
        now: datetime,
        atr: float,
    ) -> tuple[bool, Optional[str]]:
        direction = direction.strip().lower()

        for sig in self._recent:
            if sig.direction != direction:
                continue
            minutes_elapsed = (now - sig.timestamp).total_seconds() / 60
            if minutes_elapsed < self._min_minutes:
                return False, f"too soon: {int(minutes_elapsed)}min < {self._min_minutes}min"

            if atr > 0:
                price_distance = abs(entry_price - sig.entry_price)
                atr_distance = price_distance / atr
                if atr_distance < self._min_atr_distance:
                    return False, f"too close: {atr_distance:.2f} ATR < {self._min_atr_distance} ATR"

        return True, None

    def register_signal(
        self,
        direction: str,
        entry_price: float,
        timestamp: datetime,
    ) -> None:
        self._recent.append(_RecentSignal(
            direction=direction.strip().lower(),
            entry_price=entry_price,
            timestamp=timestamp,
        ))

    def cleanup(self, now: datetime, max_age_minutes: int = 240) -> None:
        cutoff = now - timedelta(minutes=max_age_minutes)
        self._recent = [s for s in self._recent if s.timestamp >= cutoff]

    def reset(self) -> None:
        self._recent.clear()
