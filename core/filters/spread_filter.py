"""
Spread Filter — Phase 3.8.

Blocks trading when the bid-ask spread exceeds a configurable
threshold. Spread can be provided dynamically or defaults to
the config value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SpreadResult:
    trade_allowed: bool
    current_spread: float
    max_spread: float
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_allowed": self.trade_allowed,
            "current_spread": round(self.current_spread, 4),
            "max_spread": round(self.max_spread, 4),
            "detail": self.detail,
        }


class SpreadFilter:
    """Blocks trading when spread is too wide."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._default_spread = config.get("default_spread", 0.25)
        self._max_spread_atr_ratio = config.get("max_spread_atr_ratio", 0.15)
        self._use_bid_ask = config.get("use_bid_ask_if_available", True)
        # "percent" mode derives the spread from price (spread_pct*price) so a cheap
        # coin isn't rejected by a gold-sized absolute default. See RRCalculator.
        self._cost_model = str(config.get("cost_model", "absolute")).lower()
        self._spread_pct = float(config.get("spread_pct", 0.0))
        if self._cost_model == "percent" and self._spread_pct <= 0:
            print("[SpreadFilter] WARNING: cost_model='percent' but spread_pct<=0 — "
                  "spread check effectively disabled (did you forget spread_pct?)")

    def check(
        self,
        spread: Optional[float] = None,
        atr: Optional[float] = None,
        price: Optional[float] = None,
    ) -> SpreadResult:
        if spread is not None:
            current = spread
        elif self._cost_model == "percent":
            # price-proportional. If price is missing/NaN, do NOT fall back to the
            # gold-sized absolute default (that would re-block the cheap coins this
            # mode exists to support) — use 0 (fail-open) so bad data can't false-reject.
            usable = price is not None and math.isfinite(price) and price > 0
            current = self._spread_pct * abs(price) if usable else 0.0
        else:
            current = self._default_spread

        if atr is not None and atr > 0:
            max_spread = atr * self._max_spread_atr_ratio
        else:
            max_spread = self._default_spread * 3

        allowed = current <= max_spread

        return SpreadResult(
            trade_allowed=allowed,
            current_spread=current,
            max_spread=max_spread,
            detail=f"spread {current:.3f} {'<=' if allowed else '>'} max {max_spread:.3f}",
        )

    def is_trade_allowed(
        self,
        spread: Optional[float] = None,
        atr: Optional[float] = None,
        price: Optional[float] = None,
    ) -> bool:
        return self.check(spread, atr, price).trade_allowed
