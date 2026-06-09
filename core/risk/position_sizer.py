"""
Position Sizer — Phase 5.5.

Computes lot size based on:
  - Account balance
  - Risk percent per trade
  - Entry and SL distance
  - Spread + slippage costs

Never rounds up. If net risk exceeds tolerance, reduces size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PositionSizeResult:
    lot_size: float
    money_at_risk_gross: float
    money_at_risk_net: float
    sl_distance_points: float
    cost_per_lot: float
    valid: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lot_size": self.lot_size,
            "money_at_risk_gross": round(self.money_at_risk_gross, 2),
            "money_at_risk_net": round(self.money_at_risk_net, 2),
            "sl_distance_points": round(self.sl_distance_points, 2),
            "cost_per_lot": round(self.cost_per_lot, 4),
            "valid": self.valid,
            "detail": self.detail,
        }


class PositionSizer:
    """Computes position size to stay within risk limits."""

    def __init__(
        self,
        risk_config: Dict[str, Any],
        cost_config: Dict[str, Any],
    ) -> None:
        self._risk_pct = risk_config.get("risk_per_trade_percent", 0.5)
        self._max_risk_pct = risk_config.get("max_risk_per_trade_percent", 1.0)
        self._spread = cost_config.get("default_spread", 0.25)
        self._slippage = cost_config.get("default_slippage", 0.10)
        self._commission = cost_config.get("commission_per_lot", 0.0)
        self._point_value = cost_config.get("point_value_per_lot", 100.0)

    def calculate(
        self,
        account_balance: float,
        entry: float,
        sl: float,
        spread: Optional[float] = None,
        slippage: Optional[float] = None,
    ) -> PositionSizeResult:
        sl_distance = abs(entry - sl)
        if sl_distance <= 0:
            return PositionSizeResult(0, 0, 0, 0, 0, False, "SL distance is zero")

        spread_cost = spread if spread is not None else self._spread
        slip_cost = slippage if slippage is not None else self._slippage
        cost_per_lot = (spread_cost + slip_cost) * self._point_value + self._commission

        max_risk_money = account_balance * (self._risk_pct / 100)
        risk_per_lot = sl_distance * self._point_value + cost_per_lot

        if risk_per_lot <= 0:
            return PositionSizeResult(0, 0, 0, sl_distance, cost_per_lot, False, "risk per lot <= 0")

        lot_size_raw = max_risk_money / risk_per_lot
        lot_size = math.floor(lot_size_raw * 100) / 100  # round DOWN to 0.01

        if lot_size <= 0:
            return PositionSizeResult(0, 0, 0, sl_distance, cost_per_lot, False, "lot size too small")

        money_gross = lot_size * sl_distance * self._point_value
        money_net = lot_size * risk_per_lot

        max_allowed = account_balance * (self._max_risk_pct / 100)
        if money_net > max_allowed:
            lot_size = math.floor((max_allowed / risk_per_lot) * 100) / 100
            money_gross = lot_size * sl_distance * self._point_value
            money_net = lot_size * risk_per_lot

        return PositionSizeResult(
            lot_size=lot_size,
            money_at_risk_gross=money_gross,
            money_at_risk_net=money_net,
            sl_distance_points=sl_distance,
            cost_per_lot=cost_per_lot,
            valid=True,
            detail=f"{lot_size} lots, ${money_net:.2f} net risk ({money_net/account_balance*100:.2f}%)",
        )
