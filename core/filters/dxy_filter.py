"""
DXY Confluence Filter — Phase 3.4.

Checks whether DXY (US Dollar Index) movement is aligned with the
intended XAU trade direction:
  - Long XAU  → DXY should be weak/falling
  - Short XAU → DXY should be strong/rising

DXY alignment is optional in v1 — it affects signal grade, not a hard block.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class DXYState(str, Enum):
    WEAK = "weak"
    STRONG = "strong"
    NEUTRAL = "neutral"
    NO_DATA = "no_data"


class DXYAlignment(str, Enum):
    ALIGNED = "aligned"
    NOT_ALIGNED = "not_aligned"
    NEUTRAL = "neutral"
    NO_DATA = "no_data"


@dataclass(frozen=True)
class DXYResult:
    state: DXYState
    alignment: DXYAlignment
    change_percent: Optional[float]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "alignment": self.alignment.value,
            "change_percent": round(self.change_percent, 4) if self.change_percent is not None else None,
            "detail": self.detail,
        }


class DXYFilter:
    """
    Evaluates DXY strength over a lookback window and checks alignment
    with a proposed XAU trade direction.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._required = config.get("dxy_required", False)
        self._lookback = config.get("dxy_lookback_candles", 20)
        self._weak_threshold = config.get("dxy_weak_threshold_pct", -0.05)
        self._strong_threshold = config.get("dxy_strong_threshold_pct", 0.05)

    def check(
        self,
        dxy_closes: Optional[List[float]],
        direction: str,
    ) -> DXYResult:
        if dxy_closes is None or len(dxy_closes) < 2:
            return DXYResult(
                state=DXYState.NO_DATA,
                alignment=DXYAlignment.NO_DATA,
                change_percent=None,
                detail="no DXY data available",
            )

        window = dxy_closes[-self._lookback:] if len(dxy_closes) >= self._lookback else dxy_closes
        start = window[0]
        end = window[-1]

        if start == 0:
            return DXYResult(
                state=DXYState.NO_DATA,
                alignment=DXYAlignment.NO_DATA,
                change_percent=None,
                detail="DXY start price is zero",
            )

        change_pct = ((end - start) / start) * 100

        if change_pct <= self._weak_threshold:
            state = DXYState.WEAK
        elif change_pct >= self._strong_threshold:
            state = DXYState.STRONG
        else:
            state = DXYState.NEUTRAL

        direction_lower = direction.strip().lower()

        if state == DXYState.NEUTRAL:
            alignment = DXYAlignment.NEUTRAL
        elif direction_lower == "long" and state == DXYState.WEAK:
            alignment = DXYAlignment.ALIGNED
        elif direction_lower == "short" and state == DXYState.STRONG:
            alignment = DXYAlignment.ALIGNED
        else:
            alignment = DXYAlignment.NOT_ALIGNED

        return DXYResult(
            state=state,
            alignment=alignment,
            change_percent=change_pct,
            detail=f"DXY {state.value} ({change_pct:+.3f}%), {alignment.value} for {direction_lower}",
        )

    def is_aligned(
        self,
        dxy_closes: Optional[List[float]],
        direction: str,
    ) -> bool:
        result = self.check(dxy_closes, direction)
        return result.alignment == DXYAlignment.ALIGNED

    @property
    def required(self) -> bool:
        return self._required
