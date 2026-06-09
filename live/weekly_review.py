"""
Weekly Review — Phase 8.5.

Compares live performance vs backtest expected:
  - Flags degradation
  - Checks for consistency
  - Recommends action (continue / pause / reduce size)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


class ReviewAction:
    CONTINUE = "continue"
    REDUCE_SIZE = "reduce_size"
    PAUSE = "pause"
    REVIEW_NEEDED = "review_needed"


@dataclass(frozen=True)
class WeeklyReviewResult:
    week_number: int
    live_trades: int
    live_win_rate: float
    live_avg_r: float
    live_total_r: float
    backtest_win_rate: float
    backtest_avg_r: float
    win_rate_ratio: float
    avg_r_ratio: float
    action: str
    warnings: List[str]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "week_number": self.week_number,
            "live_trades": self.live_trades,
            "live_win_rate": round(self.live_win_rate, 4),
            "live_avg_r": round(self.live_avg_r, 3),
            "live_total_r": round(self.live_total_r, 3),
            "backtest_win_rate": round(self.backtest_win_rate, 4),
            "backtest_avg_r": round(self.backtest_avg_r, 3),
            "win_rate_ratio": round(self.win_rate_ratio, 4),
            "avg_r_ratio": round(self.avg_r_ratio, 4),
            "action": self.action,
            "warnings": self.warnings,
            "detail": self.detail,
        }


class WeeklyReviewEngine:
    """Runs weekly performance review against backtest baseline."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._degradation_threshold = config.get("degradation_threshold", 0.70)
        self._severe_degradation = config.get("severe_degradation_threshold", 0.50)
        self._min_trades_for_review = config.get("min_trades_for_review", 3)

    def review(
        self,
        week_number: int,
        live_trades: int,
        live_win_rate: float,
        live_avg_r: float,
        live_total_r: float,
        backtest_win_rate: float,
        backtest_avg_r: float,
    ) -> WeeklyReviewResult:
        warnings: List[str] = []

        wr_ratio = live_win_rate / backtest_win_rate if backtest_win_rate > 0 else 1.0
        ar_ratio = live_avg_r / backtest_avg_r if backtest_avg_r > 0 else 1.0

        if live_trades < self._min_trades_for_review:
            action = ReviewAction.CONTINUE
            warnings.append(f"only {live_trades} trades — insufficient for review")
        elif wr_ratio < self._severe_degradation or ar_ratio < self._severe_degradation:
            action = ReviewAction.PAUSE
            warnings.append("severe performance degradation")
        elif wr_ratio < self._degradation_threshold or ar_ratio < self._degradation_threshold:
            action = ReviewAction.REDUCE_SIZE
            warnings.append("performance below 70% of backtest")
        elif live_total_r < 0:
            action = ReviewAction.REVIEW_NEEDED
            warnings.append("negative total R despite acceptable ratios")
        else:
            action = ReviewAction.CONTINUE

        if live_win_rate < 0.40 and live_trades >= self._min_trades_for_review:
            warnings.append("win rate below 40%")
        if live_avg_r < 0 and live_trades >= self._min_trades_for_review:
            warnings.append("negative average R")

        detail = f"week {week_number}: {live_trades} trades, WR ratio={wr_ratio:.0%}, avgR ratio={ar_ratio:.0%} → {action}"

        return WeeklyReviewResult(
            week_number=week_number,
            live_trades=live_trades,
            live_win_rate=live_win_rate,
            live_avg_r=live_avg_r,
            live_total_r=live_total_r,
            backtest_win_rate=backtest_win_rate,
            backtest_avg_r=backtest_avg_r,
            win_rate_ratio=wr_ratio,
            avg_r_ratio=ar_ratio,
            action=action,
            warnings=warnings,
            detail=detail,
        )
