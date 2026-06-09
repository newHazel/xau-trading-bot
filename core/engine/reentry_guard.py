"""
Re-Entry Guard — Phase 4.8.

Prevents dangerous re-entries after a stop-loss hit. Requires:
  - A completely new setup (new sweep + new FVG)
  - A new unique setup_id
  - A cooldown period after a loss
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


@dataclass
class _ExitRecord:
    direction: str
    exit_reason: str
    exit_time: datetime
    setup_id: str


class ReentryGuard:
    """Blocks re-entry after SL until conditions for a fresh setup are met."""

    def __init__(self, config: Dict[str, Any]) -> None:
        re_cfg = config.get("re_entry", {})
        self._allow_after_sl = re_cfg.get("allow_after_sl", False)
        self._allow_after_tp = re_cfg.get("allow_after_tp", False)
        self._require_new_setup_id = re_cfg.get("require_new_setup_id", True)
        self._require_new_sweep = re_cfg.get("require_new_sweep", True)
        self._cooldown_after_loss = re_cfg.get("cooldown_minutes_after_loss", 60)
        self._last_exits: Dict[str, _ExitRecord] = {}

    def register_exit(
        self,
        direction: str,
        exit_reason: str,
        exit_time: datetime,
        setup_id: str,
    ) -> None:
        direction = direction.strip().lower()
        self._last_exits[direction] = _ExitRecord(
            direction=direction,
            exit_reason=exit_reason,
            exit_time=exit_time,
            setup_id=setup_id,
        )

    def can_enter(
        self,
        direction: str,
        now: datetime,
        setup_id: str,
        has_new_sweep: bool = False,
    ) -> tuple[bool, Optional[str]]:
        direction = direction.strip().lower()
        last = self._last_exits.get(direction)

        if last is None:
            return True, None

        if last.exit_reason == "sl":
            if not self._allow_after_sl:
                cooldown_end = last.exit_time + timedelta(minutes=self._cooldown_after_loss)
                if now < cooldown_end:
                    remaining = int((cooldown_end - now).total_seconds() / 60)
                    return False, f"SL cooldown: {remaining}min remaining"

            if self._require_new_setup_id and setup_id == last.setup_id:
                return False, "requires new setup_id after SL"

            if self._require_new_sweep and not has_new_sweep:
                return False, "requires new sweep after SL"

        elif last.exit_reason == "tp1" or last.exit_reason == "tp2":
            if not self._allow_after_tp:
                if self._require_new_setup_id and setup_id == last.setup_id:
                    return False, "requires new setup_id after TP"

        return True, None

    def clear_direction(self, direction: str) -> None:
        self._last_exits.pop(direction.strip().lower(), None)

    def reset(self) -> None:
        self._last_exits.clear()
