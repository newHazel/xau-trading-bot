"""
SL Invalidation Logic — Phase 5.2.

Determines whether a stop-loss level has been invalidated by price action.
By default requires a candle CLOSE beyond the SL level (not just a wick).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SLInvalidationResult:
    invalidated: bool
    method: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invalidated": self.invalidated,
            "method": self.method,
            "detail": self.detail,
        }


class SLInvalidationChecker:
    """Checks whether a SL level has been breached."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._require_close = config.get("sl_require_close_for_invalidation", True)

    def check(
        self,
        direction: str,
        sl_price: float,
        candle_close: float,
        candle_low: Optional[float] = None,
        candle_high: Optional[float] = None,
    ) -> SLInvalidationResult:
        direction = direction.strip().lower()

        if self._require_close:
            if direction == "long" and candle_close < sl_price:
                return SLInvalidationResult(True, "close", f"close {candle_close:.2f} < SL {sl_price:.2f}")
            if direction == "short" and candle_close > sl_price:
                return SLInvalidationResult(True, "close", f"close {candle_close:.2f} > SL {sl_price:.2f}")
        else:
            if direction == "long" and candle_low is not None and candle_low < sl_price:
                return SLInvalidationResult(True, "wick", f"low {candle_low:.2f} < SL {sl_price:.2f}")
            if direction == "short" and candle_high is not None and candle_high > sl_price:
                return SLInvalidationResult(True, "wick", f"high {candle_high:.2f} > SL {sl_price:.2f}")

        return SLInvalidationResult(False, "none", "SL intact")
