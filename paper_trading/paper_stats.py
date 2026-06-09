"""
Paper Stats — Phase 7.4.

Compares paper trading results against backtest expected performance.
Flags degradation if paper performance < 70% of backtest metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ComparisonResult:
    metric_name: str
    backtest_value: float
    paper_value: float
    ratio: float
    passed: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "backtest_value": round(self.backtest_value, 4),
            "paper_value": round(self.paper_value, 4),
            "ratio": round(self.ratio, 4),
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PaperStatsResult:
    total_paper_trades: int
    comparisons: List[ComparisonResult]
    overall_passed: bool
    degraded_metrics: List[str]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_paper_trades": self.total_paper_trades,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "overall_passed": self.overall_passed,
            "degraded_metrics": self.degraded_metrics,
            "detail": self.detail,
        }


class PaperStats:
    """Compares paper performance against backtest baseline."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._min_ratio = config.get("min_performance_ratio", 0.70)
        self._min_paper_trades = config.get("min_paper_trades", 20)
        self._metrics_to_compare = config.get("metrics_to_compare", [
            "win_rate", "avg_r", "profit_factor", "expectancy",
        ])

    def compare(
        self,
        backtest_metrics: Dict[str, float],
        paper_metrics: Dict[str, float],
        total_paper_trades: int,
    ) -> PaperStatsResult:
        comparisons: List[ComparisonResult] = []
        degraded: List[str] = []

        for metric in self._metrics_to_compare:
            bt_val = backtest_metrics.get(metric, 0)
            pp_val = paper_metrics.get(metric, 0)

            if bt_val == 0:
                ratio = 1.0 if pp_val >= 0 else 0.0
            else:
                ratio = pp_val / bt_val

            passed = ratio >= self._min_ratio
            if not passed:
                degraded.append(metric)

            comparisons.append(ComparisonResult(
                metric_name=metric,
                backtest_value=bt_val,
                paper_value=pp_val,
                ratio=ratio,
                passed=passed,
                detail=f"{metric}: paper={pp_val:.4f} vs bt={bt_val:.4f} ({ratio:.0%})",
            ))

        insufficient_trades = total_paper_trades < self._min_paper_trades
        overall = len(degraded) == 0 and not insufficient_trades

        if insufficient_trades:
            detail = f"insufficient trades: {total_paper_trades} < {self._min_paper_trades}"
        elif degraded:
            detail = f"degraded metrics: {', '.join(degraded)}"
        else:
            detail = "paper performance within expected range"

        return PaperStatsResult(
            total_paper_trades=total_paper_trades,
            comparisons=comparisons,
            overall_passed=overall,
            degraded_metrics=degraded,
            detail=detail,
        )
