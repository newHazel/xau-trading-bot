"""Chart components — Phase 10.2: Equity curve, drawdown, trade scatter."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class EquityPoint:
    trade_index: int
    cumulative_r: float
    timestamp: Optional[str] = None


@dataclass
class DrawdownPoint:
    trade_index: int
    drawdown_pct: float
    drawdown_r: float


@dataclass
class TradePoint:
    trade_index: int
    net_r: float
    direction: str
    grade: str


def plot_equity_curve(trades_r: List[float]) -> Dict[str, Any]:
    if not trades_r:
        return {"chart": "equity_curve", "points": [], "final_r": 0.0}
    points = []
    cumulative = 0.0
    for i, r in enumerate(trades_r):
        cumulative += r
        points.append(EquityPoint(trade_index=i + 1, cumulative_r=round(cumulative, 4)))
    return {
        "chart": "equity_curve",
        "points": [{"trade_index": p.trade_index, "cumulative_r": p.cumulative_r} for p in points],
        "final_r": round(cumulative, 4),
    }


def plot_drawdown_curve(trades_r: List[float]) -> Dict[str, Any]:
    if not trades_r:
        return {"chart": "drawdown", "points": [], "max_drawdown_r": 0.0}
    points = []
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for i, r in enumerate(trades_r):
        cumulative += r
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)
        pct = (dd / peak * 100) if peak > 0 else 0.0
        points.append(DrawdownPoint(trade_index=i + 1, drawdown_pct=round(pct, 2), drawdown_r=round(dd, 4)))
    return {
        "chart": "drawdown",
        "points": [{"trade_index": p.trade_index, "drawdown_pct": p.drawdown_pct, "drawdown_r": p.drawdown_r} for p in points],
        "max_drawdown_r": round(max_dd, 4),
    }


def plot_trade_scatter(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"chart": "trade_scatter", "points": [], "total": 0}
    points = []
    for i, t in enumerate(trades):
        points.append(TradePoint(
            trade_index=i + 1,
            net_r=t.get("net_r", 0.0),
            direction=t.get("direction", "unknown"),
            grade=t.get("grade", "unknown"),
        ))
    winners = [p for p in points if p.net_r > 0]
    losers = [p for p in points if p.net_r < 0]
    return {
        "chart": "trade_scatter",
        "points": [{"trade_index": p.trade_index, "net_r": p.net_r, "direction": p.direction, "grade": p.grade} for p in points],
        "total": len(points),
        "winners": len(winners),
        "losers": len(losers),
    }
