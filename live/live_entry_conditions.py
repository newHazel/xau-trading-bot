"""
Live Entry Conditions — Phase 8.1.

Before entering live mode, the system must verify:
  - Paper trading positive on 20-30 trades
  - Zero Rulebook violations in paper
  - Paper stats comparison passed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class LiveEntryResult:
    ready: bool
    paper_trades: int
    paper_win_rate: float
    paper_violations: int
    paper_stats_passed: bool
    failed_conditions: List[str]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "paper_trades": self.paper_trades,
            "paper_win_rate": round(self.paper_win_rate, 4),
            "paper_violations": self.paper_violations,
            "paper_stats_passed": self.paper_stats_passed,
            "failed_conditions": self.failed_conditions,
            "detail": self.detail,
        }


class LiveEntryConditions:
    """Checks whether the system is ready to go live."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._min_paper_trades = config.get("min_paper_trades", 20)
        self._max_paper_trades = config.get("max_paper_trades", 30)
        self._require_zero_violations = config.get("require_zero_violations", True)
        self._require_paper_stats_passed = config.get("require_paper_stats_passed", True)
        self._min_paper_win_rate = config.get("min_paper_win_rate", 0.45)

    def check(
        self,
        paper_trades: int,
        paper_win_rate: float,
        paper_violations: int,
        paper_stats_passed: bool,
    ) -> LiveEntryResult:
        failed: List[str] = []

        if paper_trades < self._min_paper_trades:
            failed.append(f"paper_trades={paper_trades} < min={self._min_paper_trades}")

        if paper_win_rate < self._min_paper_win_rate:
            failed.append(f"paper_win_rate={paper_win_rate:.2%} < min={self._min_paper_win_rate:.2%}")

        if self._require_zero_violations and paper_violations > 0:
            failed.append(f"paper_violations={paper_violations}, must be 0")

        if self._require_paper_stats_passed and not paper_stats_passed:
            failed.append("paper stats comparison failed")

        ready = len(failed) == 0
        detail = "ready for live trading" if ready else f"not ready: {'; '.join(failed)}"

        return LiveEntryResult(
            ready=ready,
            paper_trades=paper_trades,
            paper_win_rate=paper_win_rate,
            paper_violations=paper_violations,
            paper_stats_passed=paper_stats_passed,
            failed_conditions=failed,
            detail=detail,
        )
