"""
State Logger — records every state machine transition.
Critical for debugging why the system is stuck in a state.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.logging.db import Database

logger = logging.getLogger(__name__)

_INSERT = """
INSERT INTO state_logs (setup_id, timestamp, from_state, to_state, reason, context_json)
VALUES (:setup_id, :timestamp, :from_state, :to_state, :reason, :context_json)
"""


class StateLogger:
    def __init__(self, db: Database) -> None:
        self._db = db

    def log_transition(
        self,
        setup_id: str,
        from_state: str,
        to_state: str,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Record one state machine transition.

        Parameters
        ----------
        setup_id   : Unique setup identifier.
        from_state : Previous state name.
        to_state   : New state name.
        reason     : Why the transition happened (e.g. 'sweep_confirmed').
        context    : Optional snapshot of relevant data at transition time.
        timestamp  : ISO UTC string — defaults to now.
        """
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        row = {
            "setup_id":    setup_id,
            "timestamp":   ts,
            "from_state":  from_state,
            "to_state":    to_state,
            "reason":      reason,
            "context_json": json.dumps(context or {}),
        }
        try:
            self._db.execute(_INSERT, row)
            logger.debug(
                "[StateLogger] %s: %s → %s (%s)",
                setup_id, from_state, to_state, reason or ""
            )
        except Exception as exc:
            logger.error("[StateLogger] Failed to log transition for %s: %s", setup_id, exc)

    def get_history(self, setup_id: str) -> list:
        """Return all transitions for a setup_id, ordered by rowid."""
        return self._db.fetchall(
            "SELECT * FROM state_logs WHERE setup_id = ? ORDER BY id ASC",
            (setup_id,),
        )

    def get_current_state(self, setup_id: str) -> Optional[str]:
        """Return the most recent to_state for a setup_id."""
        row = self._db.fetchone(
            "SELECT to_state FROM state_logs WHERE setup_id = ? ORDER BY id DESC LIMIT 1",
            (setup_id,),
        )
        return row[0] if row else None
