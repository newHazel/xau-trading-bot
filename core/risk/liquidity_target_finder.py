"""
Liquidity Target Finder — Phase 5.3.

Searches for structural targets for take-profit placement:
  - EQH/EQL nearby
  - Previous swing highs/lows
  - Opposite FVG/OB zones
  - Asia High/Low
  - Fix levels

Rules:
  - TP1 = 2R (fixed)
  - TP2 = nearest liquidity target above 2R
  - If no target between 2R and 5R → grade degrades
  - If no target before 2R → no trade
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LiquidityTarget:
    price: float
    target_type: str
    r_multiple: float
    distance: float


@dataclass(frozen=True)
class LiquidityTargetResult:
    targets: List[LiquidityTarget]
    tp2_target: Optional[LiquidityTarget]
    has_target_before_2r: bool
    has_target_between_2r_5r: bool
    valid: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "targets_count": len(self.targets),
            "tp2_price": round(self.tp2_target.price, 2) if self.tp2_target else None,
            "tp2_type": self.tp2_target.target_type if self.tp2_target else None,
            "tp2_r_multiple": round(self.tp2_target.r_multiple, 2) if self.tp2_target else None,
            "has_target_before_2r": self.has_target_before_2r,
            "has_target_between_2r_5r": self.has_target_between_2r_5r,
            "valid": self.valid,
            "detail": self.detail,
        }


class LiquidityTargetFinder:
    """Finds structural liquidity targets for TP placement."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._require_for_tp2 = config.get("require_liquidity_target_for_tp2", True)
        self._tp1_r = config.get("tp1_r", 2.0)
        self._tp2_r = config.get("tp2_r", 3.5)

    def find(
        self,
        direction: str,
        entry: float,
        sl_distance: float,
        levels: List[Dict[str, Any]],
    ) -> LiquidityTargetResult:
        direction = direction.strip().lower()
        if sl_distance <= 0:
            return LiquidityTargetResult([], None, False, False, False, "invalid SL distance")

        targets: List[LiquidityTarget] = []

        for level in levels:
            price = level["price"]
            ltype = level.get("type", "unknown")

            if direction == "long" and price <= entry:
                continue
            if direction == "short" and price >= entry:
                continue

            distance = abs(price - entry)
            r_mult = distance / sl_distance

            targets.append(LiquidityTarget(
                price=price,
                target_type=ltype,
                r_multiple=r_mult,
                distance=distance,
            ))

        targets.sort(key=lambda t: t.distance)

        has_before_2r = any(t.r_multiple <= self._tp1_r for t in targets)
        has_2r_5r = any(self._tp1_r < t.r_multiple <= 5.0 for t in targets)

        # Strictly BEYOND tp1 (not >=): TakeProfitCalculator only adopts a liquidity TP2
        # when it is past tp1 (LONG price > tp1 / SHORT < tp1). A target sitting exactly at
        # tp1_r used to pass this >= filter and set liquidity_target_clear=True while
        # TakeProfit silently fell back to the 3.5R generic level — an inconsistent booster
        # flag at the 2R boundary. Match the > threshold so the two agree.
        tp2_candidates = [t for t in targets if t.r_multiple > self._tp1_r]
        tp2_target = tp2_candidates[0] if tp2_candidates else None

        valid = True
        detail = "targets found"
        if not has_before_2r and not targets:
            valid = True
            detail = "no nearby targets but tradeable"
        if self._require_for_tp2 and tp2_target is None:
            detail = "no liquidity target for TP2"

        return LiquidityTargetResult(
            targets=targets,
            tp2_target=tp2_target,
            has_target_before_2r=has_before_2r,
            has_target_between_2r_5r=has_2r_5r,
            valid=valid,
            detail=detail,
        )
