"""
Rejection Logger — every failed signal is logged here with full context.
No-trade decisions are as important as trades for strategy analysis.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from core.logging.db import Database

logger = logging.getLogger(__name__)

_INSERT = """
INSERT INTO rejected_signals (
    setup_id, symbol, timestamp, attempted_direction, htf_bias,
    reason_main, reasons_json, failed_conditions_json,
    passed_conditions_json, context_snapshot_json
) VALUES (
    :setup_id, :symbol, :timestamp, :attempted_direction, :htf_bias,
    :reason_main, :reasons_json, :failed_conditions_json,
    :passed_conditions_json, :context_snapshot_json
)
"""


class RejectionLogger:
    def __init__(self, db: Database) -> None:
        self._db = db

    def log_rejection(
        self,
        *,
        symbol: str,
        timestamp: str,
        reason_main: str,
        attempted_direction: Optional[str] = None,
        htf_bias: Optional[str] = None,
        setup_id: Optional[str] = None,
        failed_conditions: Optional[List[str]] = None,
        passed_conditions: Optional[List[str]] = None,
        reasons: Optional[List[str]] = None,
        context_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a rejected signal with full diagnostic context.

        Parameters
        ----------
        reason_main : str
            Primary rejection reason (e.g. 'fvg_mitigation_exceeded').
        failed_conditions : list
            All conditions that failed (e.g. ['fvg_mitigation_within_limit']).
        passed_conditions : list
            Conditions that DID pass — helps identify near-misses.
        context_snapshot : dict
            Full market context at rejection time (price, ATR, bias, etc.)
        """
        row = {
            "setup_id":               setup_id,
            "symbol":                 symbol,
            "timestamp":              timestamp,
            "attempted_direction":    attempted_direction,
            "htf_bias":               htf_bias,
            "reason_main":            reason_main,
            "reasons_json":           json.dumps(reasons or []),
            "failed_conditions_json": json.dumps(failed_conditions or []),
            "passed_conditions_json": json.dumps(passed_conditions or []),
            "context_snapshot_json":  json.dumps(context_snapshot or {}),
        }
        try:
            self._db.execute(_INSERT, row)
            logger.info(
                "[RejectionLogger] %s %s | reason=%s failed=%s",
                symbol, timestamp, reason_main,
                (failed_conditions or []),
            )
        except Exception as exc:
            logger.error("[RejectionLogger] Failed to log rejection: %s", exc)

    def count_today(self, symbol: str, date_str: str) -> int:
        """Count rejections for a symbol on a given date (YYYY-MM-DD)."""
        row = self._db.fetchone(
            "SELECT COUNT(*) FROM rejected_signals "
            "WHERE symbol = ? AND timestamp LIKE ?",
            (symbol, f"{date_str}%"),
        )
        return row[0] if row else 0
