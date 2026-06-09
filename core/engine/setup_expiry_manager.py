"""
Setup Expiry Manager — Phase 4.6.

Tracks when a setup was created and determines if it has expired
based on the configured expiry window. The timer starts from
fvg_creation, sweep, or zone_arrival (configurable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


@dataclass
class ActiveSetup:
    setup_id: str
    direction: str
    created_at: datetime
    expiry_at: datetime
    origin: str


class SetupExpiryManager:
    """Manages setup lifetimes and expiry."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._expiry_minutes = config.get("setup_expiry_minutes", 90)
        self._starts_from = config.get("setup_expiry_starts_from", "fvg_creation")
        self._max_active = config.get("max_active_setups_per_symbol", 1)
        self._active: Dict[str, ActiveSetup] = {}

    def register_setup(
        self,
        setup_id: str,
        direction: str,
        created_at: datetime,
    ) -> bool:
        if len(self._active) >= self._max_active:
            return False
        if setup_id in self._active:
            return False
        expiry_at = created_at + timedelta(minutes=self._expiry_minutes)
        self._active[setup_id] = ActiveSetup(
            setup_id=setup_id,
            direction=direction,
            created_at=created_at,
            expiry_at=expiry_at,
            origin=self._starts_from,
        )
        return True

    def is_expired(self, setup_id: str, now: datetime) -> bool:
        setup = self._active.get(setup_id)
        if setup is None:
            return True
        return now >= setup.expiry_at

    def check_and_expire(self, now: datetime) -> list[str]:
        expired = [
            sid for sid, s in self._active.items()
            if now >= s.expiry_at
        ]
        for sid in expired:
            del self._active[sid]
        return expired

    def remove_setup(self, setup_id: str) -> bool:
        return self._active.pop(setup_id, None) is not None

    def get_setup(self, setup_id: str) -> Optional[ActiveSetup]:
        return self._active.get(setup_id)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def active_setups(self) -> Dict[str, ActiveSetup]:
        return dict(self._active)

    def reset(self) -> None:
        self._active.clear()
