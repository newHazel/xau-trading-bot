"""
R:R Calculator — Phase 5.4.

Computes gross and net risk-reward ratios including execution costs.
Net R:R must be >= 2.0 for entry, >= 2.5 for A+.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RRResult:
    gross_rr: float
    net_rr: float
    spread_cost: float
    slippage_cost: float
    commission_cost: float
    total_cost: float
    valid: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gross_rr": round(self.gross_rr, 3),
            "net_rr": round(self.net_rr, 3),
            "spread_cost": round(self.spread_cost, 4),
            "slippage_cost": round(self.slippage_cost, 4),
            "commission_cost": round(self.commission_cost, 4),
            "total_cost": round(self.total_cost, 4),
            "valid": self.valid,
            "detail": self.detail,
        }


class RRCalculator:
    """Computes risk-reward with execution costs."""

    def __init__(self, risk_config: Dict[str, Any], cost_config: Dict[str, Any]) -> None:
        self._min_rr = risk_config.get("rr_tiers", {}).get("min_to_enter", 2.0)
        self._spread = cost_config.get("default_spread", 0.25)
        self._slippage = cost_config.get("default_slippage", 0.10)
        self._commission = cost_config.get("commission_per_lot", 0.0)
        self._news_slip_mult = cost_config.get("news_slippage_multiplier", 3.0)
        self._high_vol_slip_mult = cost_config.get("high_volatility_slippage_multiplier", 2.0)

    def calculate(
        self,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        is_news_time: bool = False,
        is_high_volatility: bool = False,
        spread: Optional[float] = None,
    ) -> RRResult:
        direction = direction.strip().lower()
        spread_cost = spread if spread is not None else self._spread
        slippage = self._slippage

        if is_news_time:
            slippage *= self._news_slip_mult
        elif is_high_volatility:
            slippage *= self._high_vol_slip_mult

        total_cost = spread_cost + slippage

        if direction == "long":
            risk_gross = entry - sl
            reward_gross = tp - entry
            risk_net = risk_gross + total_cost
            reward_net = reward_gross - total_cost
        else:
            risk_gross = sl - entry
            reward_gross = entry - tp
            risk_net = risk_gross + total_cost
            reward_net = reward_gross - total_cost

        if risk_gross <= 0:
            return RRResult(0, 0, spread_cost, slippage, self._commission, total_cost,
                            False, "invalid: risk <= 0")
        if risk_net <= 0:
            return RRResult(0, 0, spread_cost, slippage, self._commission, total_cost,
                            False, "invalid: net risk <= 0")

        gross_rr = reward_gross / risk_gross
        net_rr = reward_net / risk_net

        valid = net_rr >= self._min_rr

        return RRResult(
            gross_rr=gross_rr,
            net_rr=net_rr,
            spread_cost=spread_cost,
            slippage_cost=slippage,
            commission_cost=self._commission,
            total_cost=total_cost,
            valid=valid,
            detail=f"gross {gross_rr:.2f}R, net {net_rr:.2f}R {'✓' if valid else '✗'}",
        )
