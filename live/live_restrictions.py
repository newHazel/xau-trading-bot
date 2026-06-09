"""
Live Restrictions — Phase 8.3.

Enforces:
  - A/A+ grade only
  - No trading during news events
  - Stop after 2nd loss in a day
  - No trading in extreme volatility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class RestrictionCheckResult:
    allowed: bool
    blocked_reasons: List[str]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "blocked_reasons": self.blocked_reasons,
            "detail": self.detail,
        }


class LiveRestrictions:
    """Enforces live trading restrictions."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._allowed_grades = set(config.get("allowed_grades", ["A+", "A"]))
        self._block_during_news = config.get("block_during_news", True)
        self._max_daily_losses = config.get("max_daily_losses_live", 2)
        self._block_extreme_volatility = config.get("block_extreme_volatility", True)

    def check(
        self,
        grade: str,
        is_news_time: bool = False,
        daily_losses: int = 0,
        volatility_regime: str = "normal",
    ) -> RestrictionCheckResult:
        blocked: List[str] = []

        if grade not in self._allowed_grades:
            blocked.append(f"grade {grade} not allowed (need {self._allowed_grades})")

        if self._block_during_news and is_news_time:
            blocked.append("news event active")

        if daily_losses >= self._max_daily_losses:
            blocked.append(f"daily losses={daily_losses} >= max={self._max_daily_losses}")

        if self._block_extreme_volatility and volatility_regime == "extreme":
            blocked.append("extreme volatility")

        allowed = len(blocked) == 0
        detail = "trade allowed" if allowed else f"blocked: {'; '.join(blocked)}"

        return RestrictionCheckResult(
            allowed=allowed,
            blocked_reasons=blocked,
            detail=detail,
        )
