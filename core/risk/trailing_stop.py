"""
Trailing Stop — Phase 5.8.

Trailing logic:
  - Before TP1: SL stays at original level
  - After TP1 hit: move SL to breakeven + costs
  - After 3R profit: trailing starts using structure or ATR method
  - Structure method: trail below/above the last 1m/5m swing
  - ATR method: trail at price ± ATR * 1.5
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class TrailingPhase(str, Enum):
    INITIAL = "initial"
    BREAKEVEN = "breakeven"
    TRAILING = "trailing"


@dataclass(frozen=True)
class TrailingResult:
    phase: TrailingPhase
    new_sl: float
    moved: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "new_sl": round(self.new_sl, 2),
            "moved": self.moved,
            "detail": self.detail,
        }


class TrailingStopManager:
    """Manages trailing stop progression through phases."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._tp1_r = config.get("tp1_r", 2.0)
        self._trailing_start_r = config.get("trailing_start_r", 3.0)
        self._buffer_ratio = config.get("sl_buffer_atr_ratio", 0.20)
        self._atr_trail_mult = config.get("atr_trailing_multiplier", 1.5)

    def update(
        self,
        direction: str,
        entry: float,
        current_sl: float,
        current_price: float,
        sl_distance: float,
        atr: float,
        costs: float = 0.0,
        swing_level: Optional[float] = None,
    ) -> TrailingResult:
        direction = direction.strip().lower()

        if direction == "long":
            current_r = (current_price - entry) / sl_distance if sl_distance > 0 else 0
        else:
            current_r = (entry - current_price) / sl_distance if sl_distance > 0 else 0

        if current_r < self._tp1_r:
            return TrailingResult(TrailingPhase.INITIAL, current_sl, False, "below TP1, SL unchanged")

        be_price = self._breakeven_price(direction, entry, costs)

        if current_r < self._trailing_start_r:
            if direction == "long" and be_price > current_sl:
                return TrailingResult(TrailingPhase.BREAKEVEN, be_price, True, f"moved to BE {be_price:.2f}")
            elif direction == "short" and be_price < current_sl:
                return TrailingResult(TrailingPhase.BREAKEVEN, be_price, True, f"moved to BE {be_price:.2f}")
            return TrailingResult(TrailingPhase.BREAKEVEN, current_sl, False, "BE already set")

        buffer = atr * self._buffer_ratio
        if swing_level is not None:
            trail_sl = self._trail_by_structure(direction, swing_level, buffer)
        else:
            trail_sl = self._trail_by_atr(direction, current_price, atr)

        if direction == "long":
            new_sl = max(current_sl, trail_sl, be_price)
            moved = new_sl > current_sl
        else:
            new_sl = min(current_sl, trail_sl, be_price)
            moved = new_sl < current_sl

        return TrailingResult(
            TrailingPhase.TRAILING, new_sl, moved,
            f"trailing at {new_sl:.2f} ({'structure' if swing_level else 'ATR'})",
        )

    def _breakeven_price(self, direction: str, entry: float, costs: float) -> float:
        if direction == "long":
            return entry + costs
        return entry - costs

    def _trail_by_structure(self, direction: str, swing_level: float, buffer: float) -> float:
        if direction == "long":
            return swing_level - buffer
        return swing_level + buffer

    def _trail_by_atr(self, direction: str, price: float, atr: float) -> float:
        if direction == "long":
            return price - atr * self._atr_trail_mult
        return price + atr * self._atr_trail_mult
