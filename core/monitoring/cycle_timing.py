"""
Align the live loop to ROUND clock marks.

5m candles close on round 5-minute boundaries (…:00, :05, :10 UTC). If the bot
just slept a fixed `interval` from whenever the process happened to start, every
scan would drift to an arbitrary phase (e.g. always :02:37) — fetching the fresh
bar late. seconds_until_next_mark() instead returns how long to sleep so the NEXT
scan lands right after the next boundary, every cycle.

Epoch-based alignment: UTC midnight is an exact multiple of 300s (86400 % 300 == 0),
so 300-second marks coincide exactly with …:00, :05, :10 of every hour. A small
buffer is added so the just-closed candle is actually available from the data API
(and to absorb minor clock skew); the fetchers drop the in-progress bar anyway, so
the buffer only needs to be a few seconds.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def seconds_until_next_mark(
    interval_seconds: int,
    buffer_seconds: float = 5.0,
    now: Optional[datetime] = None,
) -> float:
    """Seconds to sleep so the next cycle fires just after the next round mark.

    interval_seconds=300 -> next of …:00, :05, :10 (+buffer) UTC.
    If we're sitting exactly on a mark, we wait a full interval (we just ran it).
    """
    now = now or datetime.now(timezone.utc)
    epoch = now.timestamp()
    step = max(1, int(interval_seconds))
    next_mark = (int(epoch // step) + 1) * step
    wait = next_mark - epoch + buffer_seconds
    if wait <= 0:  # buffer overshot the mark (only if buffer >= step) — clamp forward
        wait += step
    return wait
