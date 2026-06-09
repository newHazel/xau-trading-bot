"""
Daily Limits — Phase 5.9.

Tracks daily trade count, losses, and enforces day-lock rules:
  - max_daily_losses: 2 → day locked
  - max_daily_trades: 3 → day locked
  - stop_after_tp: lock after first TP hit (if configured)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Dict, Optional


class DayLockReason(str, Enum):
    MAX_LOSSES = "max_losses"
    MAX_TRADES = "max_trades"
    AFTER_TP = "after_tp"


@dataclass(frozen=True)
class DailyLimitResult:
    trade_allowed: bool
    lock_reason: Optional[DayLockReason]
    trades_today: int
    losses_today: int
    wins_today: int
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_allowed": self.trade_allowed,
            "lock_reason": self.lock_reason.value if self.lock_reason else None,
            "trades_today": self.trades_today,
            "losses_today": self.losses_today,
            "wins_today": self.wins_today,
            "detail": self.detail,
        }


class DailyLimits:
    """Enforces daily trading limits."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._max_losses = config.get("max_daily_losses", 2)
        self._max_trades = config.get("max_daily_trades", 3)
        self._stop_after_tp = config.get("stop_after_tp", True)
        self._current_date: Optional[date] = None
        self._trades = 0
        self._losses = 0
        self._wins = 0
        self._tp_hit = False
        self._locked = False
        self._lock_reason: Optional[DayLockReason] = None

    def check(self, today: date) -> DailyLimitResult:
        self._maybe_reset(today)

        if self._locked:
            return DailyLimitResult(
                False, self._lock_reason, self._trades, self._losses, self._wins,
                f"day locked: {self._lock_reason.value}" if self._lock_reason else "day locked",
            )

        return DailyLimitResult(
            True, None, self._trades, self._losses, self._wins,
            f"trades: {self._trades}/{self._max_trades}, losses: {self._losses}/{self._max_losses}",
        )

    def register_trade(self, today: date) -> None:
        self._maybe_reset(today)
        self._trades += 1
        if self._trades >= self._max_trades:
            self._locked = True
            self._lock_reason = DayLockReason.MAX_TRADES

    def register_loss(self, today: date) -> None:
        self._maybe_reset(today)
        self._losses += 1
        if self._losses >= self._max_losses:
            self._locked = True
            self._lock_reason = DayLockReason.MAX_LOSSES

    def register_win(self, today: date) -> None:
        self._maybe_reset(today)
        self._wins += 1
        if self._stop_after_tp:
            self._tp_hit = True
            self._locked = True
            self._lock_reason = DayLockReason.AFTER_TP

    @property
    def is_locked(self) -> bool:
        return self._locked

    def _maybe_reset(self, today: date) -> None:
        if self._current_date != today:
            self._current_date = today
            self._trades = 0
            self._losses = 0
            self._wins = 0
            self._tp_hit = False
            self._locked = False
            self._lock_reason = None

    def reset(self) -> None:
        self._current_date = None
        self._trades = 0
        self._losses = 0
        self._wins = 0
        self._tp_hit = False
        self._locked = False
        self._lock_reason = None
