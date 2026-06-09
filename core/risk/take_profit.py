"""
Take Profit Calculator — Phase 5.3 (continued).

TP1 = entry ± (sl_distance * tp1_r)  → always 2R
TP2 = liquidity target if available, else entry ± (sl_distance * tp2_r)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TakeProfitResult:
    tp1: float
    tp2: float
    tp1_r: float
    tp2_r: float
    tp2_from_liquidity: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp1": round(self.tp1, 2),
            "tp2": round(self.tp2, 2),
            "tp1_r": self.tp1_r,
            "tp2_r": round(self.tp2_r, 2),
            "tp2_from_liquidity": self.tp2_from_liquidity,
            "detail": self.detail,
        }


class TakeProfitCalculator:
    """Computes TP1 and TP2 levels."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._tp1_r = config.get("tp1_r", 2.0)
        self._tp2_r = config.get("tp2_r", 3.5)

    def calculate(
        self,
        direction: str,
        entry: float,
        sl_distance: float,
        liquidity_target_price: Optional[float] = None,
    ) -> TakeProfitResult:
        direction = direction.strip().lower()

        if direction == "long":
            tp1 = entry + sl_distance * self._tp1_r
            if liquidity_target_price and liquidity_target_price > tp1:
                tp2 = liquidity_target_price
                tp2_r = (tp2 - entry) / sl_distance
                from_liq = True
            else:
                tp2 = entry + sl_distance * self._tp2_r
                tp2_r = self._tp2_r
                from_liq = False
        else:
            tp1 = entry - sl_distance * self._tp1_r
            if liquidity_target_price and liquidity_target_price < tp1:
                tp2 = liquidity_target_price
                tp2_r = (entry - tp2) / sl_distance
                from_liq = True
            else:
                tp2 = entry - sl_distance * self._tp2_r
                tp2_r = self._tp2_r
                from_liq = False

        detail = f"TP1 at {self._tp1_r}R, TP2 at {tp2_r:.1f}R"
        if from_liq:
            detail += " (liquidity target)"

        return TakeProfitResult(
            tp1=tp1, tp2=tp2,
            tp1_r=self._tp1_r, tp2_r=tp2_r,
            tp2_from_liquidity=from_liq,
            detail=detail,
        )
