"""
Paper Engine — Phase 7.2.

Same state machine as live, alerts only, conservative fills.
Processes signals in real-time simulation mode:
  - Validates against rulebook
  - Grades signal (A/A+ only for paper)
  - Tracks position through fill engine
  - Logs to paper journal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


@dataclass
class PaperSignal:
    setup_id: str
    direction: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    grade: str
    timestamp: datetime
    conditions_met: Dict[str, bool] = field(default_factory=dict)
    optional_scores: Dict[str, int] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 2),
            "sl_price": round(self.sl_price, 2),
            "tp1_price": round(self.tp1_price, 2),
            "tp2_price": round(self.tp2_price, 2),
            "grade": self.grade,
            "timestamp": self.timestamp.isoformat(),
            "conditions_met": self.conditions_met,
        }


@dataclass
class PaperPosition:
    signal: PaperSignal
    entry_time: datetime
    lot_size: float
    remaining_lots: float
    tp1_hit: bool = False
    current_sl: float = 0.0
    status: str = "open"  # open / closed_sl / closed_tp1 / closed_tp2 / closed_trailing

    @property
    def is_open(self) -> bool:
        return self.status == "open"


@dataclass
class PaperTradeResult:
    setup_id: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: Optional[datetime]
    exit_type: str
    grade: str
    net_r: float
    gross_r: float
    lot_size: float
    net_pnl: float
    costs: float
    conditions_met: Dict[str, bool] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_type": self.exit_type,
            "grade": self.grade,
            "net_r": round(self.net_r, 3),
            "gross_r": round(self.gross_r, 3),
            "lot_size": self.lot_size,
            "net_pnl": round(self.net_pnl, 2),
            "costs": round(self.costs, 4),
            "notes": self.notes,
        }


class PaperEngine:
    """Paper trading engine — alerts only, no real execution."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._allowed_grades = set(config.get("allowed_grades", ["A+", "A"]))
        self._max_daily_trades = config.get("max_daily_trades", 3)
        self._max_daily_losses = config.get("max_daily_losses", 2)
        self._stop_after_tp = config.get("stop_after_tp", True)
        self._spread = config.get("default_spread", 0.25)
        self._slippage = config.get("default_slippage", 0.10)

        self._position: Optional[PaperPosition] = None
        self._results: List[PaperTradeResult] = []
        self._daily_trades: Dict[str, int] = {}
        self._daily_losses: Dict[str, int] = {}
        self._daily_wins: Dict[str, int] = {}

    @property
    def results(self) -> List[PaperTradeResult]:
        return list(self._results)

    @property
    def has_open_position(self) -> bool:
        return self._position is not None and self._position.is_open

    def accept_signal(self, signal: PaperSignal) -> Dict[str, Any]:
        day_key = signal.timestamp.strftime("%Y-%m-%d")

        if signal.grade not in self._allowed_grades:
            return {"accepted": False, "reason": f"grade {signal.grade} not in {self._allowed_grades}"}

        if self._daily_trades.get(day_key, 0) >= self._max_daily_trades:
            return {"accepted": False, "reason": "daily trade limit reached"}

        if self._daily_losses.get(day_key, 0) >= self._max_daily_losses:
            return {"accepted": False, "reason": "daily loss limit reached"}

        if self._stop_after_tp and self._daily_wins.get(day_key, 0) > 0:
            return {"accepted": False, "reason": "stop after TP — already won today"}

        if self.has_open_position:
            return {"accepted": False, "reason": "position already open"}

        self._daily_trades[day_key] = self._daily_trades.get(day_key, 0) + 1

        sl_distance = abs(signal.entry_price - signal.sl_price)
        self._position = PaperPosition(
            signal=signal,
            entry_time=signal.timestamp,
            lot_size=0.01,
            remaining_lots=0.01,
            current_sl=signal.sl_price,
        )

        return {"accepted": True, "setup_id": signal.setup_id}

    def update_price(
        self,
        high: float,
        low: float,
        close: float,
        timestamp: datetime,
    ) -> Optional[PaperTradeResult]:
        if not self.has_open_position:
            return None

        pos = self._position
        sig = pos.signal
        sl_distance = abs(sig.entry_price - sig.sl_price)

        sl_hit = (sig.direction == "long" and low <= pos.current_sl) or \
                 (sig.direction == "short" and high >= pos.current_sl)
        tp1_hit = (sig.direction == "long" and high >= sig.tp1_price) or \
                  (sig.direction == "short" and low <= sig.tp1_price)
        tp2_hit = (sig.direction == "long" and high >= sig.tp2_price) or \
                  (sig.direction == "short" and low <= sig.tp2_price)

        if sl_hit and tp1_hit:
            return self._close_position("sl_hit", pos.current_sl, timestamp)

        if sl_hit:
            return self._close_position("sl_hit", pos.current_sl, timestamp)

        if not pos.tp1_hit and tp1_hit:
            pos.tp1_hit = True
            costs = self._spread + self._slippage
            if sig.direction == "long":
                pos.current_sl = sig.entry_price + costs
            else:
                pos.current_sl = sig.entry_price - costs

        if pos.tp1_hit and tp2_hit:
            return self._close_position("tp2_hit", sig.tp2_price, timestamp)

        return None

    def _close_position(
        self, exit_type: str, exit_price: float, timestamp: datetime,
    ) -> PaperTradeResult:
        pos = self._position
        sig = pos.signal
        sl_distance = abs(sig.entry_price - sig.sl_price)
        costs = self._spread + self._slippage

        if sig.direction == "long":
            gross_r = (exit_price - sig.entry_price) / sl_distance if sl_distance > 0 else 0
            net_pnl = (exit_price - sig.entry_price - costs) * pos.lot_size
        else:
            gross_r = (sig.entry_price - exit_price) / sl_distance if sl_distance > 0 else 0
            net_pnl = (sig.entry_price - exit_price - costs) * pos.lot_size

        net_r = gross_r - (costs / sl_distance if sl_distance > 0 else 0)

        day_key = timestamp.strftime("%Y-%m-%d")
        if gross_r <= 0:
            self._daily_losses[day_key] = self._daily_losses.get(day_key, 0) + 1
        else:
            self._daily_wins[day_key] = self._daily_wins.get(day_key, 0) + 1

        result = PaperTradeResult(
            setup_id=sig.setup_id,
            direction=sig.direction,
            entry_price=sig.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            exit_type=exit_type,
            grade=sig.grade,
            net_r=net_r,
            gross_r=gross_r,
            lot_size=pos.lot_size,
            net_pnl=net_pnl,
            costs=costs,
            conditions_met=sig.conditions_met,
        )
        self._results.append(result)
        pos.status = f"closed_{exit_type}"
        self._position = None
        return result

    def reset_daily(self, day_key: str) -> None:
        self._daily_trades.pop(day_key, None)
        self._daily_losses.pop(day_key, None)
        self._daily_wins.pop(day_key, None)

    def reset(self) -> None:
        self._position = None
        self._results.clear()
        self._daily_trades.clear()
        self._daily_losses.clear()
        self._daily_wins.clear()
