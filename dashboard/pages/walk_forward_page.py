"""Walk-Forward Page — Phase 10.1: View walk-forward validation results."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class FoldResult:
    fold_index: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_trades: int
    oos_trades: int
    is_win_rate: float
    oos_win_rate: float
    oos_total_r: float
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "is_period": f"{self.is_start} → {self.is_end}",
            "oos_period": f"{self.oos_start} → {self.oos_end}",
            "is_trades": self.is_trades,
            "oos_trades": self.oos_trades,
            "is_win_rate": round(self.is_win_rate, 4),
            "oos_win_rate": round(self.oos_win_rate, 4),
            "oos_total_r": round(self.oos_total_r, 4),
            "passed": self.passed,
        }


@dataclass
class WalkForwardPageData:
    folds: List[FoldResult] = field(default_factory=list)
    overall_passed: bool = False

    def add_fold(self, fold: FoldResult) -> None:
        self.folds.append(fold)

    @property
    def total_folds(self) -> int:
        return len(self.folds)

    @property
    def passed_folds(self) -> int:
        return sum(1 for f in self.folds if f.passed)

    @property
    def failed_folds(self) -> int:
        return sum(1 for f in self.folds if not f.passed)

    def get_oos_metrics(self) -> Dict[str, Any]:
        if not self.folds:
            return {}
        oos_wrs = [f.oos_win_rate for f in self.folds]
        oos_rs = [f.oos_total_r for f in self.folds]
        return {
            "avg_oos_win_rate": round(sum(oos_wrs) / len(oos_wrs), 4),
            "avg_oos_total_r": round(sum(oos_rs) / len(oos_rs), 4),
            "min_oos_win_rate": round(min(oos_wrs), 4),
            "max_oos_total_r": round(max(oos_rs), 4),
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_folds": self.total_folds,
            "passed_folds": self.passed_folds,
            "failed_folds": self.failed_folds,
            "overall_passed": self.overall_passed,
            "oos_metrics": self.get_oos_metrics(),
        }

    def to_records(self) -> List[Dict[str, Any]]:
        return [f.to_dict() for f in self.folds]


def render_walk_forward(data: Optional[WalkForwardPageData] = None) -> Dict[str, Any]:
    data = data or WalkForwardPageData()
    return {
        "page": "Walk-Forward",
        "summary": data.get_summary(),
        "records": data.to_records(),
    }
