"""
Telegram Dedup — Phase 9.4.

Prevents duplicate Telegram messages using hash:
  hash = content + setup_id + timestamp_minute
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, Optional


class TelegramDedup:
    """Deduplicates Telegram messages using content hashing."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._max_history = config.get("max_dedup_history", 1000)
        self._seen: OrderedDict[str, datetime] = OrderedDict()

    def should_send(
        self,
        content: str,
        setup_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        msg_hash = self._compute_hash(content, setup_id, timestamp)
        if msg_hash in self._seen:
            return False
        self._seen[msg_hash] = timestamp or datetime.utcnow()
        self._trim()
        return True

    def mark_sent(
        self,
        content: str,
        setup_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        msg_hash = self._compute_hash(content, setup_id, timestamp)
        self._seen[msg_hash] = timestamp or datetime.utcnow()
        self._trim()
        return msg_hash

    def is_duplicate(
        self,
        content: str,
        setup_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        msg_hash = self._compute_hash(content, setup_id, timestamp)
        return msg_hash in self._seen

    @property
    def history_size(self) -> int:
        return len(self._seen)

    def clear(self) -> None:
        self._seen.clear()

    def _compute_hash(
        self,
        content: str,
        setup_id: Optional[str],
        timestamp: Optional[datetime],
    ) -> str:
        ts_minute = ""
        if timestamp:
            ts_minute = timestamp.strftime("%Y%m%d%H%M")
        raw = f"{content}|{setup_id or ''}|{ts_minute}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _trim(self) -> None:
        while len(self._seen) > self._max_history:
            self._seen.popitem(last=False)
