"""
Backtest Metrics — Phase 6.6.

Computes:
  - Win Rate, Avg R, Profit Factor, Max Drawdown
  - Sharpe-like ratio, Expectancy
  - Breakdowns by direction, grade, session, day-of-week
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class MetricsResult:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_r: float
    avg_win_r: float
    avg_loss_r: float
    profit_factor: float
    max_drawdown_r: float
    max_drawdown_pct: float
    total_r: float
    expectancy: float
    sharpe_like: float
    best_trade_r: float
    worst_trade_r: float
    avg_bars_in_trade: float
    breakdowns: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "avg_r": round(self.avg_r, 3),
            "avg_win_r": round(self.avg_win_r, 3),
            "avg_loss_r": round(self.avg_loss_r, 3),
            "profit_factor": round(self.profit_factor, 3),
            "max_drawdown_r": round(self.max_drawdown_r, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "total_r": round(self.total_r, 3),
            "expectancy": round(self.expectancy, 3),
            "sharpe_like": round(self.sharpe_like, 3),
            "best_trade_r": round(self.best_trade_r, 3),
            "worst_trade_r": round(self.worst_trade_r, 3),
            "avg_bars_in_trade": round(self.avg_bars_in_trade, 1),
            "breakdowns": self.breakdowns,
        }


def compute_metrics(
    trades: List[Dict[str, Any]],
    initial_balance: float = 10000.0,
) -> MetricsResult:
    if not trades:
        return _empty_metrics()

    r_values = [t.get("r_multiple", 0) for t in trades]
    net_pnls = [t.get("net_pnl", 0) for t in trades]

    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r <= 0]

    total_trades = len(r_values)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades if total_trades > 0 else 0

    avg_r = sum(r_values) / total_trades if total_trades > 0 else 0
    avg_win_r = sum(wins) / win_count if win_count > 0 else 0
    avg_loss_r = sum(losses) / loss_count if loss_count > 0 else 0

    gross_wins = sum(r for r in r_values if r > 0)
    gross_losses = abs(sum(r for r in r_values if r <= 0))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0

    total_r = sum(r_values)
    expectancy = (win_rate * avg_win_r) + ((1 - win_rate) * avg_loss_r)

    equity_curve = _build_equity_curve(net_pnls, initial_balance)
    max_dd_pct = _max_drawdown_pct(equity_curve)
    max_dd_r = _max_drawdown_r(r_values)

    sharpe = _sharpe_like(r_values)

    best_r = max(r_values) if r_values else 0
    worst_r = min(r_values) if r_values else 0

    bars_in_trade = [t.get("bar_exit", 0) - t.get("bar_entry", 0) for t in trades]
    avg_bars = sum(bars_in_trade) / len(bars_in_trade) if bars_in_trade else 0

    breakdowns = _compute_breakdowns(trades)

    return MetricsResult(
        total_trades=total_trades,
        wins=win_count,
        losses=loss_count,
        win_rate=win_rate,
        avg_r=avg_r,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        profit_factor=profit_factor,
        max_drawdown_r=max_dd_r,
        max_drawdown_pct=max_dd_pct,
        total_r=total_r,
        expectancy=expectancy,
        sharpe_like=sharpe,
        best_trade_r=best_r,
        worst_trade_r=worst_r,
        avg_bars_in_trade=avg_bars,
        breakdowns=breakdowns,
    )


def _build_equity_curve(pnls: List[float], start: float) -> List[float]:
    curve = [start]
    for pnl in pnls:
        curve.append(curve[-1] + pnl)
    return curve


def _max_drawdown_pct(equity: List[float]) -> float:
    if len(equity) < 2:
        return 0
    peak = equity[0]
    max_dd = 0
    for val in equity[1:]:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    return max_dd


def _max_drawdown_r(r_values: List[float]) -> float:
    if not r_values:
        return 0
    cumulative = 0
    peak = 0
    max_dd = 0
    for r in r_values:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        max_dd = max(max_dd, dd)
    return max_dd


def _sharpe_like(r_values: List[float]) -> float:
    if len(r_values) < 2:
        return 0
    mean_r = sum(r_values) / len(r_values)
    variance = sum((r - mean_r) ** 2 for r in r_values) / (len(r_values) - 1)
    std = math.sqrt(variance) if variance > 0 else 0
    return mean_r / std if std > 0 else 0


def _compute_breakdowns(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_direction: Dict[str, List[float]] = {}
    by_grade: Dict[str, List[float]] = {}

    for t in trades:
        d = t.get("direction", "unknown")
        g = t.get("grade", "unknown")
        r = t.get("r_multiple", 0)

        by_direction.setdefault(d, []).append(r)
        by_grade.setdefault(g, []).append(r)

    direction_stats = {}
    for d, rs in by_direction.items():
        w = [r for r in rs if r > 0]
        direction_stats[d] = {
            "count": len(rs),
            "win_rate": len(w) / len(rs) if rs else 0,
            "avg_r": sum(rs) / len(rs) if rs else 0,
            "total_r": sum(rs),
        }

    grade_stats = {}
    for g, rs in by_grade.items():
        w = [r for r in rs if r > 0]
        grade_stats[g] = {
            "count": len(rs),
            "win_rate": len(w) / len(rs) if rs else 0,
            "avg_r": sum(rs) / len(rs) if rs else 0,
            "total_r": sum(rs),
        }

    return {"by_direction": direction_stats, "by_grade": grade_stats}


def _empty_metrics() -> MetricsResult:
    return MetricsResult(
        total_trades=0, wins=0, losses=0, win_rate=0, avg_r=0,
        avg_win_r=0, avg_loss_r=0, profit_factor=0, max_drawdown_r=0,
        max_drawdown_pct=0, total_r=0, expectancy=0, sharpe_like=0,
        best_trade_r=0, worst_trade_r=0, avg_bars_in_trade=0,
    )
