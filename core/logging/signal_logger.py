"""
Signal Logger — writes validated signals to the signals table.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from core.logging.db import Database

logger = logging.getLogger(__name__)

_INSERT = """
INSERT OR IGNORE INTO signals (
    setup_id, symbol, timestamp, direction, entry, stop_loss, tp1, tp2,
    rr, grade, confidence_score, htf_bias, structure_15m, price_zone,
    sweep_found, sweep_quality, fvg_valid, fvg_freshness,
    displacement_strength, ob_valid, ob_strength, news_clear,
    news_tier_nearest, dxy_aligned, correlation_state, trigger_confirmed,
    session, liquidity_target_distance, status, config_hash, strategy_version
) VALUES (
    :setup_id, :symbol, :timestamp, :direction, :entry, :stop_loss, :tp1, :tp2,
    :rr, :grade, :confidence_score, :htf_bias, :structure_15m, :price_zone,
    :sweep_found, :sweep_quality, :fvg_valid, :fvg_freshness,
    :displacement_strength, :ob_valid, :ob_strength, :news_clear,
    :news_tier_nearest, :dxy_aligned, :correlation_state, :trigger_confirmed,
    :session, :liquidity_target_distance, :status, :config_hash, :strategy_version
)
"""

_UPDATE_STATUS = "UPDATE signals SET status = ? WHERE setup_id = ?"


class SignalLogger:
    def __init__(self, db: Database) -> None:
        self._db = db

    def log_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Insert a signal row.  Returns True on success, False if setup_id
        already exists (duplicate guard via INSERT OR IGNORE).
        """
        row = _defaults(signal)
        try:
            cur = self._db.execute(_INSERT, row)
            inserted = cur.rowcount > 0
            if inserted:
                logger.info("[SignalLogger] Logged signal %s | grade=%s rr=%.2f",
                            row["setup_id"], row["grade"], row["rr"])
            else:
                logger.warning("[SignalLogger] Duplicate setup_id ignored: %s", row["setup_id"])
            return inserted
        except Exception as exc:
            logger.error("[SignalLogger] Failed to log signal %s: %s", row.get("setup_id"), exc)
            return False

    def update_status(self, setup_id: str, status: str) -> None:
        """Update signal lifecycle status (pending → sent → active → closed)."""
        try:
            self._db.execute(_UPDATE_STATUS, (status, setup_id))
            logger.debug("[SignalLogger] %s status → %s", setup_id, status)
        except Exception as exc:
            logger.error("[SignalLogger] Status update failed for %s: %s", setup_id, exc)

    def exists(self, setup_id: str) -> bool:
        """Check if a setup_id already exists — duplicate prevention."""
        row = self._db.fetchone(
            "SELECT 1 FROM signals WHERE setup_id = ?", (setup_id,)
        )
        return row is not None


def _defaults(d: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in None for every optional column so named bindings work."""
    cols = [
        "setup_id", "symbol", "timestamp", "direction", "entry", "stop_loss",
        "tp1", "tp2", "rr", "grade", "confidence_score", "htf_bias",
        "structure_15m", "price_zone", "sweep_found", "sweep_quality",
        "fvg_valid", "fvg_freshness", "displacement_strength", "ob_valid",
        "ob_strength", "news_clear", "news_tier_nearest", "dxy_aligned",
        "correlation_state", "trigger_confirmed", "session",
        "liquidity_target_distance", "status", "config_hash", "strategy_version",
    ]
    return {c: d.get(c) for c in cols}
