"""Backtest Page — Phase 10.1: View backtest results and metrics."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class BacktestResult:
    experiment_id: str
    config_hash: str
    total_trades: int
    win_rate: float
    avg_r: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe: float
    expectancy: float
    total_r: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_r": round(self.avg_r, 4),
            "profit_factor": round(self.profit_factor, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe": round(self.sharpe, 4),
            "expectancy": round(self.expectancy, 4),
            "total_r": round(self.total_r, 4),
        }


@dataclass
class BacktestPageData:
    results: List[BacktestResult] = field(default_factory=list)

    def add_result(self, result: BacktestResult) -> None:
        self.results.append(result)

    @property
    def total_experiments(self) -> int:
        return len(self.results)

    def get_best_by(self, metric: str) -> Optional[BacktestResult]:
        if not self.results:
            return None
        return max(self.results, key=lambda r: getattr(r, metric, 0))

    def filter_by_min_trades(self, min_trades: int) -> List[BacktestResult]:
        return [r for r in self.results if r.total_trades >= min_trades]

    def filter_by_min_win_rate(self, min_wr: float) -> List[BacktestResult]:
        return [r for r in self.results if r.win_rate >= min_wr]

    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {"total_experiments": 0}
        win_rates = [r.win_rate for r in self.results]
        total_rs = [r.total_r for r in self.results]
        return {
            "total_experiments": len(self.results),
            "avg_win_rate": round(sum(win_rates) / len(win_rates), 4),
            "best_total_r": round(max(total_rs), 4),
            "worst_total_r": round(min(total_rs), 4),
        }

    def to_records(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.results]


def render_backtest(data: Optional[BacktestPageData] = None) -> Dict[str, Any]:
    data = data or BacktestPageData()
    return {
        "page": "Backtest",
        "summary": data.get_summary(),
        "records": data.to_records(),
    }
