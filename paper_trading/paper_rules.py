"""
Paper Trading Rules — Phase 7.5.

Enforces:
  - Minimum 20-30 trades before graduating
  - A/A+ grade only (configurable)
  - Zero Rulebook violations
  - No forced overrides
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GraduationResult:
    ready: bool
    total_trades: int
    violations_count: int
    grade_distribution: Dict[str, int]
    failed_rules: List[str]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "total_trades": self.total_trades,
            "violations_count": self.violations_count,
            "grade_distribution": self.grade_distribution,
            "failed_rules": self.failed_rules,
            "detail": self.detail,
        }


class PaperRules:
    """Enforces paper trading rules and graduation criteria."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._min_trades = config.get("min_paper_trades", 20)
        self._max_trades = config.get("max_paper_trades", 30)
        self._allowed_grades = set(config.get("allowed_grades", ["A+", "A"]))
        self._zero_violations = config.get("require_zero_violations", True)
        self._min_win_rate = config.get("min_win_rate_for_graduation", 0.45)

    def check_signal_allowed(self, grade: str) -> Dict[str, Any]:
        if grade in self._allowed_grades:
            return {"allowed": True}
        return {"allowed": False, "reason": f"grade {grade} not in {self._allowed_grades}"}

    def check_graduation(
        self,
        total_trades: int,
        violations_count: int,
        grade_distribution: Dict[str, int],
        win_rate: float,
        paper_stats_passed: bool,
    ) -> GraduationResult:
        failed = []

        if total_trades < self._min_trades:
            failed.append(f"trades={total_trades} < min={self._min_trades}")

        if self._zero_violations and violations_count > 0:
            failed.append(f"violations={violations_count}, must be 0")

        disallowed = {g: c for g, c in grade_distribution.items() if g not in self._allowed_grades and c > 0}
        if disallowed:
            failed.append(f"disallowed grades traded: {disallowed}")

        if win_rate < self._min_win_rate:
            failed.append(f"win_rate={win_rate:.2%} < min={self._min_win_rate:.2%}")

        if not paper_stats_passed:
            failed.append("paper stats comparison failed")

        ready = len(failed) == 0
        detail = "ready to graduate to live" if ready else f"not ready: {'; '.join(failed)}"

        return GraduationResult(
            ready=ready,
            total_trades=total_trades,
            violations_count=violations_count,
            grade_distribution=grade_distribution,
            failed_rules=failed,
            detail=detail,
        )
