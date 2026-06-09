"""
Trade Logger — records opened and closed trades.
Tracks gross/net R, costs, partial closes, trailing stop, and exit reasons.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from core.logging.db import Database

logger = logging.getLogger(__name__)

_INSERT = """
INSERT INTO trades (
    signal_id, setup_id, symbol, direction, entry, stop_loss, tp1, tp2,
    status, opened_at
) VALUES (
    :signal_id, :setup_id, :symbol, :direction, :entry, :stop_loss, :tp1, :tp2,
    'open', :opened_at
)
"""

_CLOSE = """
UPDATE trades SET
    exit_price          = :exit_price,
    exit_reason         = :exit_reason,
    result              = :result,
    gross_r             = :gross_r,
    net_r               = :net_r,
    spread_cost         = :spread_cost,
    slippage_cost       = :slippage_cost,
    commission_cost     = :commission_cost,
    partial_close_executed = :partial_close_executed,
    breakeven_moved     = :breakeven_moved,
    trailing_active     = :trailing_active,
    status              = 'closed',
    closed_at           = :closed_at
WHERE setup_id = ?
"""


class TradeLogger:
    def __init__(self, db: Database) -> None:
        self._db = db

    def open_trade(self, trade: Dict[str, Any]) -> int:
        """
        Insert an open trade row. Returns the new row id.
        Required keys: setup_id, symbol, direction, entry, stop_loss, tp1, opened_at.
        """
        row = {
            "signal_id":  trade.get("signal_id"),
            "setup_id":   trade["setup_id"],
            "symbol":     trade["symbol"],
            "direction":  trade["direction"],
            "entry":      trade["entry"],
            "stop_loss":  trade["stop_loss"],
            "tp1":        trade["tp1"],
            "tp2":        trade.get("tp2"),
            "opened_at":  trade["opened_at"],
        }
        try:
            cur = self._db.execute(_INSERT, row)
            logger.info(
                "[TradeLogger] Opened trade %s | %s %s @ %.2f",
                row["setup_id"], row["direction"], row["symbol"], row["entry"],
            )
            return cur.lastrowid
        except Exception as exc:
            logger.error("[TradeLogger] Failed to open trade %s: %s", row.get("setup_id"), exc)
            return -1

    def close_trade(self, setup_id: str, close_data: Dict[str, Any]) -> None:
        """
        Update trade row with exit details and compute result label.

        close_data keys: exit_price, exit_reason, gross_r, net_r,
        spread_cost, slippage_cost, commission_cost, closed_at,
        partial_close_executed, breakeven_moved, trailing_active.
        """
        gross_r = close_data.get("gross_r", 0.0)
        result = "win" if gross_r > 0 else ("loss" if gross_r < 0 else "breakeven")

        try:
            with self._db.connection() as conn:
                conn.execute(
                    "UPDATE trades SET "
                    "exit_price=?, exit_reason=?, result=?, gross_r=?, net_r=?, "
                    "spread_cost=?, slippage_cost=?, commission_cost=?, "
                    "partial_close_executed=?, breakeven_moved=?, trailing_active=?, "
                    "status='closed', closed_at=? "
                    "WHERE setup_id=?",
                    (
                        close_data.get("exit_price"),
                        close_data.get("exit_reason"),
                        result,
                        gross_r,
                        close_data.get("net_r"),
                        close_data.get("spread_cost", 0.0),
                        close_data.get("slippage_cost", 0.0),
                        close_data.get("commission_cost", 0.0),
                        int(close_data.get("partial_close_executed", False)),
                        int(close_data.get("breakeven_moved", False)),
                        int(close_data.get("trailing_active", False)),
                        close_data.get("closed_at"),
                        setup_id,
                    ),
                )
            logger.info(
                "[TradeLogger] Closed trade %s | result=%s gross_r=%.2f net_r=%.2f",
                setup_id, result, gross_r, close_data.get("net_r") or 0.0,
            )
        except Exception as exc:
            logger.error("[TradeLogger] Failed to close trade %s: %s", setup_id, exc)

    def get_open_trades(self, symbol: Optional[str] = None) -> list:
        if symbol:
            return self._db.fetchall(
                "SELECT * FROM trades WHERE status='open' AND symbol=?", (symbol,)
            )
        return self._db.fetchall("SELECT * FROM trades WHERE status='open'")

    def count_losses_today(self, symbol: str, date_str: str) -> int:
        row = self._db.fetchone(
            "SELECT COUNT(*) FROM trades WHERE symbol=? AND result='loss' AND closed_at LIKE ?",
            (symbol, f"{date_str}%"),
        )
        return row[0] if row else 0
