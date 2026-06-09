"""
Live Size Manager — Phase 8.2.

Controls position sizing for live small mode:
  - 5-10% of capital allocation
  - 0.25-0.5% risk per trade
  - 1 trade per day max
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class LiveSizeResult:
    allowed_capital: float
    risk_per_trade_pct: float
    max_risk_money: float
    max_daily_trades: int
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed_capital": round(self.allowed_capital, 2),
            "risk_per_trade_pct": round(self.risk_per_trade_pct, 4),
            "max_risk_money": round(self.max_risk_money, 2),
            "max_daily_trades": self.max_daily_trades,
            "detail": self.detail,
        }


class LiveSizeManager:
    """Manages position sizing constraints for live small mode."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._capital_pct_min = config.get("capital_allocation_min_pct", 5.0)
        self._capital_pct_max = config.get("capital_allocation_max_pct", 10.0)
        self._risk_pct_min = config.get("risk_per_trade_min_pct", 0.25)
        self._risk_pct_max = config.get("risk_per_trade_max_pct", 0.50)
        self._max_daily_trades = config.get("max_daily_trades_live", 1)
        self._current_capital_pct = config.get("current_capital_allocation_pct", 5.0)
        self._current_risk_pct = config.get("current_risk_per_trade_pct", 0.25)

    def get_limits(self, total_balance: float) -> LiveSizeResult:
        cap_pct = max(self._capital_pct_min, min(self._current_capital_pct, self._capital_pct_max))
        risk_pct = max(self._risk_pct_min, min(self._current_risk_pct, self._risk_pct_max))

        allowed_capital = total_balance * (cap_pct / 100)
        max_risk_money = allowed_capital * (risk_pct / 100)

        return LiveSizeResult(
            allowed_capital=allowed_capital,
            risk_per_trade_pct=risk_pct,
            max_risk_money=max_risk_money,
            max_daily_trades=self._max_daily_trades,
            detail=f"{cap_pct:.1f}% capital (${allowed_capital:.2f}), {risk_pct:.2f}% risk/trade (${max_risk_money:.2f})",
        )

    def validate_trade_size(
        self, total_balance: float, proposed_risk_money: float, daily_trades_so_far: int,
    ) -> Dict[str, Any]:
        limits = self.get_limits(total_balance)

        if daily_trades_so_far >= limits.max_daily_trades:
            return {"allowed": False, "reason": f"daily trade limit: {daily_trades_so_far} >= {limits.max_daily_trades}"}

        if proposed_risk_money > limits.max_risk_money:
            return {"allowed": False, "reason": f"risk ${proposed_risk_money:.2f} > max ${limits.max_risk_money:.2f}"}

        return {"allowed": True, "limits": limits.to_dict()}
