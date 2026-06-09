"""
Volatility / ATR Regime Filter — Phase 3.6.

Classifies the current volatility regime based on ATR relative to
a rolling median. Blocks trading in low or extreme volatility.

Regimes:
  - LOW:     ATR < 0.5x median → no trade (market dead)
  - NORMAL:  0.5x–1.5x median  → trade normally
  - HIGH:    1.5x–2.5x median  → trade with lower risk
  - EXTREME: > 2.5x median     → no trade (market erratic)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class VolatilityRegime(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"
    NO_DATA = "no_data"


@dataclass(frozen=True)
class VolatilityResult:
    regime: VolatilityRegime
    trade_allowed: bool
    current_atr: Optional[float]
    median_atr: Optional[float]
    atr_ratio: Optional[float]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "trade_allowed": self.trade_allowed,
            "current_atr": round(self.current_atr, 4) if self.current_atr is not None else None,
            "median_atr": round(self.median_atr, 4) if self.median_atr is not None else None,
            "atr_ratio": round(self.atr_ratio, 3) if self.atr_ratio is not None else None,
            "detail": self.detail,
        }


class VolatilityFilter:
    """Classifies ATR regime and blocks trading at extremes."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._low_threshold = config.get("low_atr_ratio", 0.5)
        self._high_threshold = config.get("high_atr_ratio", 1.5)
        self._extreme_threshold = config.get("extreme_atr_ratio", 2.5)
        self._median_lookback = config.get("atr_median_lookback", 100)

    def check(self, atr_values: Optional[List[float]]) -> VolatilityResult:
        if atr_values is None or len(atr_values) < 10:
            return VolatilityResult(
                regime=VolatilityRegime.NO_DATA,
                trade_allowed=False,
                current_atr=None,
                median_atr=None,
                atr_ratio=None,
                detail="insufficient ATR data",
            )

        current = atr_values[-1]
        window = atr_values[-self._median_lookback:]
        median = float(np.median(window))

        if median == 0:
            return VolatilityResult(
                regime=VolatilityRegime.NO_DATA,
                trade_allowed=False,
                current_atr=current,
                median_atr=0.0,
                atr_ratio=None,
                detail="median ATR is zero",
            )

        ratio = current / median

        if ratio < self._low_threshold:
            regime = VolatilityRegime.LOW
            allowed = False
        elif ratio > self._extreme_threshold:
            regime = VolatilityRegime.EXTREME
            allowed = False
        elif ratio > self._high_threshold:
            regime = VolatilityRegime.HIGH
            allowed = True
        else:
            regime = VolatilityRegime.NORMAL
            allowed = True

        return VolatilityResult(
            regime=regime,
            trade_allowed=allowed,
            current_atr=current,
            median_atr=median,
            atr_ratio=ratio,
            detail=f"{regime.value} volatility (ratio {ratio:.2f}x median)",
        )

    def is_trade_allowed(self, atr_values: Optional[List[float]]) -> bool:
        return self.check(atr_values).trade_allowed
