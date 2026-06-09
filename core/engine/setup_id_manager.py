"""
Setup ID Manager — Phase 4.5.

Generates unique setup IDs in the format:
  XAU-{YYYYMMDD}-{HHMM}-{LONG/SHORT}-{ZONE_ID}

Tracks issued IDs to prevent duplicate signals for the same setup.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Set


class SetupIDManager:
    """Generates and tracks unique setup IDs."""

    def __init__(self) -> None:
        self._issued: Set[str] = set()

    def generate(
        self,
        dt: datetime,
        direction: str,
        zone_id: str,
        symbol: str = "XAU",
    ) -> str:
        date_str = dt.strftime("%Y%m%d")
        time_str = dt.strftime("%H%M")
        dir_str = direction.strip().upper()
        setup_id = f"{symbol}-{date_str}-{time_str}-{dir_str}-{zone_id}"
        return setup_id

    def register(self, setup_id: str) -> bool:
        if setup_id in self._issued:
            return False
        self._issued.add(setup_id)
        return True

    def is_duplicate(self, setup_id: str) -> bool:
        return setup_id in self._issued

    def generate_and_register(
        self,
        dt: datetime,
        direction: str,
        zone_id: str,
        symbol: str = "XAU",
    ) -> Optional[str]:
        setup_id = self.generate(dt, direction, zone_id, symbol)
        if self.register(setup_id):
            return setup_id
        return None

    @property
    def issued_count(self) -> int:
        return len(self._issued)

    def reset(self) -> None:
        self._issued.clear()
