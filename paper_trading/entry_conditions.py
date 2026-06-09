"""
Paper Trading Entry Conditions — Phase 7.1.

Before entering paper trading mode, the system must verify:
  - 6+ months of backtest data
  - 100+ setups evaluated
  - Walk-Forward validation passed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EntryConditionResult:
    ready: bool
    backtest_months: float
    total_setups: int
    walk_forward_passed: bool
    failed_conditions: list
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "backtest_months": round(self.backtest_months, 1),
            "total_setups": self.total_setups,
            "walk_forward_passed": self.walk_forward_passed,
            "failed_conditions": self.failed_conditions,
            "detail": self.detail,
        }


class PaperEntryConditions:
    """Checks whether the system is ready to enter paper trading."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._min_backtest_months = config.get("min_backtest_months", 6)
        self._min_setups = config.get("min_setups", 100)
        self._require_walk_forward = config.get("require_walk_forward_passed", True)

    def check(
        self,
        backtest_months: float,
        total_setups: int,
        walk_forward_passed: bool,
    ) -> EntryConditionResult:
        failed = []

        if backtest_months < self._min_backtest_months:
            failed.append(f"backtest_months={backtest_months:.1f} < {self._min_backtest_months}")

        if total_setups < self._min_setups:
            failed.append(f"total_setups={total_setups} < {self._min_setups}")

        if self._require_walk_forward and not walk_forward_passed:
            failed.append("walk_forward not passed")

        ready = len(failed) == 0
        detail = "ready for paper trading" if ready else f"not ready: {'; '.join(failed)}"

        return EntryConditionResult(
            ready=ready,
            backtest_months=backtest_months,
            total_setups=total_setups,
            walk_forward_passed=walk_forward_passed,
            failed_conditions=failed,
            detail=detail,
        )
