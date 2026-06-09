"""
Scaling Manager — Phase 8.4.

Gradual scaling rules:
  - After 30 live trades → review
  - If performance meets criteria → +25% risk max increase
  - Never exceed absolute caps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ScalingReviewResult:
    eligible_for_scaling: bool
    current_risk_pct: float
    proposed_risk_pct: float
    live_trades: int
    live_win_rate: float
    criteria_met: List[str]
    criteria_failed: List[str]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eligible_for_scaling": self.eligible_for_scaling,
            "current_risk_pct": round(self.current_risk_pct, 4),
            "proposed_risk_pct": round(self.proposed_risk_pct, 4),
            "live_trades": self.live_trades,
            "live_win_rate": round(self.live_win_rate, 4),
            "criteria_met": self.criteria_met,
            "criteria_failed": self.criteria_failed,
            "detail": self.detail,
        }


class ScalingManager:
    """Manages gradual risk scaling based on live performance."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._review_interval_trades = config.get("review_interval_trades", 30)
        self._scale_increase_pct = config.get("scale_increase_pct", 25.0)
        self._absolute_max_risk_pct = config.get("absolute_max_risk_pct", 1.0)
        self._min_win_rate_for_scale = config.get("min_win_rate_for_scale", 0.45)
        self._min_avg_r_for_scale = config.get("min_avg_r_for_scale", 0.5)
        self._require_positive_total_r = config.get("require_positive_total_r", True)

    def review(
        self,
        current_risk_pct: float,
        live_trades: int,
        win_rate: float,
        avg_r: float,
        total_r: float,
    ) -> ScalingReviewResult:
        met: List[str] = []
        failed: List[str] = []

        if live_trades >= self._review_interval_trades:
            met.append(f"trades={live_trades} >= {self._review_interval_trades}")
        else:
            failed.append(f"trades={live_trades} < {self._review_interval_trades}")

        if win_rate >= self._min_win_rate_for_scale:
            met.append(f"win_rate={win_rate:.2%} >= {self._min_win_rate_for_scale:.2%}")
        else:
            failed.append(f"win_rate={win_rate:.2%} < {self._min_win_rate_for_scale:.2%}")

        if avg_r >= self._min_avg_r_for_scale:
            met.append(f"avg_r={avg_r:.3f} >= {self._min_avg_r_for_scale}")
        else:
            failed.append(f"avg_r={avg_r:.3f} < {self._min_avg_r_for_scale}")

        if not self._require_positive_total_r or total_r > 0:
            met.append(f"total_r={total_r:.3f} > 0")
        else:
            failed.append(f"total_r={total_r:.3f} <= 0")

        eligible = len(failed) == 0
        proposed = current_risk_pct
        if eligible:
            proposed = current_risk_pct * (1 + self._scale_increase_pct / 100)
            proposed = min(proposed, self._absolute_max_risk_pct)

        detail = f"scale to {proposed:.4f}%" if eligible else f"not eligible: {'; '.join(failed)}"

        return ScalingReviewResult(
            eligible_for_scaling=eligible,
            current_risk_pct=current_risk_pct,
            proposed_risk_pct=proposed,
            live_trades=live_trades,
            live_win_rate=win_rate,
            criteria_met=met,
            criteria_failed=failed,
            detail=detail,
        )
