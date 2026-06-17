"""
Near-miss telemetry — answers "is the bot too selective?" with real data.

A "near-miss" is a setup that completed the FULL SMC sequence (HTF bias → 15m align →
zone → sweep → FVG → retrace → micro-CHoCH → confirmation) but was then REJECTED at an
at-entry gate (off kill-zone, news window, R:R < 2 net, daily limit, blocking filter,
or grade below B). These are the setups the eye sees that the bot deliberately skips.

This tracker tallies near-misses by reason and (optionally) sends a brief Telegram note
per near-miss, so the user can SEE exactly what is filtered and why — and judge whether
the selectivity is right or whether #2/#3 (the signal-adders) are worth deploying.
Measurement only — it changes no signal logic.
"""

from __future__ import annotations

import html
from typing import Any, Dict, Optional

from core.alerts.telegram_sender import fmt_price


class NearMissTracker:
    def __init__(self, notify: bool = True, label: str = "") -> None:
        self.by_reason: Dict[str, int] = {}
        self.total = 0
        self._notify_enabled = notify
        self._label = label  # short ticker (e.g. "ETH") shown per note; "" for gold

    def record(self, nm: Dict[str, Any], sender: Any = None) -> None:
        reason = str(nm.get("reason", "?"))
        self.by_reason[reason] = self.by_reason.get(reason, 0) + 1
        self.total += 1
        if self._notify_enabled:
            self._notify(sender, nm)

    def summary_line(self) -> str:
        if not self.total:
            return "near-misses: 0"
        parts = ", ".join(f"{n}×{r}" for r, n in
                          sorted(self.by_reason.items(), key=lambda kv: -kv[1]))
        return f"near-misses: {self.total} ({parts})"

    def _notify(self, sender: Any, nm: Dict[str, Any]) -> None:
        if sender is None:
            return
        entry = nm.get("entry")
        rr = nm.get("rr")
        # adaptive precision so a cheap coin (DOGE @ 0.087) isn't shown as "@ 0.09"
        ep = f" @ {fmt_price(entry)}" if isinstance(entry, (int, float)) else ""
        rrp = f", R:R {rr:.1f}" if isinstance(rr, (int, float)) else ""
        # This message uses HTML, but reason/grade/direction are free-form (e.g. the
        # reason "R:R < 2 net" contains a '<'). Escape them so Telegram's HTML parser
        # doesn't reject the whole message and silently drop the note.
        tag = html.escape(self._label) + " " if self._label else ""
        grade = html.escape(str(nm.get("grade", "?")))
        direction = html.escape(str(nm.get("direction", "?")))
        reason = html.escape(str(nm.get("reason", "?")))
        text = (f"⚪ <b>{tag}Near-miss</b> — {grade} {direction}{ep} "
                f"completed but skipped: <b>{reason}</b>{rrp}")
        try:
            sender.send(text)
        except Exception:
            pass
