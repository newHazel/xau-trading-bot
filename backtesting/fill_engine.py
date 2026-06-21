"""
Conservative Fill Engine — Phase 6.2.

Rules:
  - If candle range covers both SL and TP → SL fills first (conservative)
  - Uses 1m candles within the bar if available for intra-bar resolution
  - Partial close at TP1 (50%), remainder trails to TP2
  - All fills include spread + slippage costs
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class FillType(str, Enum):
    ENTRY = "entry"
    SL_HIT = "sl_hit"
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    TRAILING_SL = "trailing_sl"
    EXPIRED = "expired"
    GAP_INVALIDATED = "gap_invalidated"


@dataclass(frozen=True)
class FillResult:
    fill_type: FillType
    fill_price: float
    bar_index: int
    gross_pnl: float
    net_pnl: float
    costs: float
    r_multiple: float
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fill_type": self.fill_type.value,
            "fill_price": round(self.fill_price, 2),
            "bar_index": self.bar_index,
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "costs": round(self.costs, 4),
            "r_multiple": round(self.r_multiple, 3),
            "detail": self.detail,
        }


@dataclass
class OpenPosition:
    direction: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    lot_size: float
    remaining_lots: float
    entry_bar: int
    setup_id: str
    sl_distance: float
    tp1_hit: bool = False
    costs_per_lot: float = 0.0
    # accumulated realized PnL from the partial (TP1) leg, so a scaled-out trade's final
    # TradeRecord can BLEND the banked partial with the surviving leg (set by BacktestRunner).
    realized_gross_pnl: float = 0.0
    realized_net_pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.remaining_lots > 0


class FillEngine:
    """Conservative fill logic for backtesting."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._tp1_close_pct = config.get("tp1_partial_close_percent", 0.50)
        self._spread = config.get("default_spread", 0.25)
        self._slippage = config.get("default_slippage", 0.10)
        self._conservative = config.get("conservative_backtest", True)

    def check_fills(
        self,
        position: OpenPosition,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_index: int,
        intrabar_candles: Optional[List[Dict[str, float]]] = None,
    ) -> List[FillResult]:
        results: List[FillResult] = []
        if not position.is_open:
            return results

        if intrabar_candles:
            return self._check_intrabar(position, intrabar_candles, bar_index)

        sl_hit = self._sl_touched(position, bar_high, bar_low)
        tp_hit = self._tp_touched(position, bar_high, bar_low)

        if sl_hit and tp_hit and self._conservative:
            results.append(self._create_sl_fill(position, bar_index))
            return results

        if sl_hit:
            results.append(self._create_sl_fill(position, bar_index))
            return results

        if tp_hit:
            if not position.tp1_hit:
                results.append(self._create_tp1_fill(position, bar_index))
            else:
                tp2_hit = self._tp2_touched(position, bar_high, bar_low)
                if tp2_hit:
                    results.append(self._create_tp2_fill(position, bar_index))

        return results

    def _check_intrabar(
        self, position: OpenPosition, candles: List[Dict[str, float]], bar_index: int,
    ) -> List[FillResult]:
        results: List[FillResult] = []
        for candle in candles:
            if not position.is_open:
                break
            h, l = candle["high"], candle["low"]
            sl_hit = self._sl_touched(position, h, l)
            tp_hit = self._tp_touched(position, h, l)

            if sl_hit:
                results.append(self._create_sl_fill(position, bar_index))
                break
            if tp_hit:
                if not position.tp1_hit:
                    results.append(self._create_tp1_fill(position, bar_index))
                else:
                    if self._tp2_touched(position, h, l):
                        results.append(self._create_tp2_fill(position, bar_index))
                        break
        return results

    def _sl_touched(self, pos: OpenPosition, high: float, low: float) -> bool:
        if pos.direction == "long":
            return low <= pos.sl_price
        return high >= pos.sl_price

    def _tp_touched(self, pos: OpenPosition, high: float, low: float) -> bool:
        target = pos.tp1_price if not pos.tp1_hit else pos.tp2_price
        if pos.direction == "long":
            return high >= target
        return low <= target

    def _tp2_touched(self, pos: OpenPosition, high: float, low: float) -> bool:
        if pos.direction == "long":
            return high >= pos.tp2_price
        return low <= pos.tp2_price

    def _create_sl_fill(self, pos: OpenPosition, bar_index: int) -> FillResult:
        slip = self._slippage
        if pos.direction == "long":
            fill_price = pos.sl_price - slip
            gross_pnl = (fill_price - pos.entry_price) * pos.remaining_lots
        else:
            fill_price = pos.sl_price + slip
            gross_pnl = (pos.entry_price - fill_price) * pos.remaining_lots

        costs = pos.costs_per_lot * pos.remaining_lots
        net_pnl = gross_pnl - costs
        r_mult = (fill_price - pos.entry_price) / pos.sl_distance if pos.direction == "long" else \
                 (pos.entry_price - fill_price) / pos.sl_distance
        pos.remaining_lots = 0
        return FillResult(FillType.SL_HIT, fill_price, bar_index, gross_pnl, net_pnl, costs, r_mult,
                          f"SL hit at {fill_price:.2f}")

    def _create_tp1_fill(self, pos: OpenPosition, bar_index: int) -> FillResult:
        close_lots = pos.remaining_lots * self._tp1_close_pct
        slip = self._slippage  # F2: TP fills include slippage too — symmetric with SL,
        if pos.direction == "long":   # so winners' R is net-of-slippage, not gross.
            fill_price = pos.tp1_price - slip
            gross_pnl = (fill_price - pos.entry_price) * close_lots
        else:
            fill_price = pos.tp1_price + slip
            gross_pnl = (pos.entry_price - fill_price) * close_lots

        costs = pos.costs_per_lot * close_lots
        net_pnl = gross_pnl - costs
        r_mult = (fill_price - pos.entry_price) / pos.sl_distance if pos.direction == "long" else \
                 (pos.entry_price - fill_price) / pos.sl_distance

        pos.remaining_lots -= close_lots
        pos.tp1_hit = True
        return FillResult(FillType.TP1_HIT, fill_price, bar_index, gross_pnl, net_pnl, costs, r_mult,
                          f"TP1 hit at {fill_price:.2f}, closed {close_lots:.2f} lots")

    def _create_tp2_fill(self, pos: OpenPosition, bar_index: int) -> FillResult:
        slip = self._slippage  # F2: TP fills include slippage too (symmetric with SL)
        if pos.direction == "long":
            fill_price = pos.tp2_price - slip
            gross_pnl = (fill_price - pos.entry_price) * pos.remaining_lots
        else:
            fill_price = pos.tp2_price + slip
            gross_pnl = (pos.entry_price - fill_price) * pos.remaining_lots

        costs = pos.costs_per_lot * pos.remaining_lots
        net_pnl = gross_pnl - costs
        r_mult = (fill_price - pos.entry_price) / pos.sl_distance if pos.direction == "long" else \
                 (pos.entry_price - fill_price) / pos.sl_distance

        pos.remaining_lots = 0
        return FillResult(FillType.TP2_HIT, fill_price, bar_index, gross_pnl, net_pnl, costs, r_mult,
                          f"TP2 hit at {fill_price:.2f}")
