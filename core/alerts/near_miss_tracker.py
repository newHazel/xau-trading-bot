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

from typing import Any, Dict, Optional


class NearMissTracker:
    def __init__(self, notify: bool = True) -> None:
        self.by_reason: Dict[str, int] = {}
        self.total = 0
        self._notify_enabled = notify

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
        ep = f" @ {entry:.2f}" if isinstance(entry, (int, float)) else ""
        rrp = f", R:R {rr:.1f}" if isinstance(rr, (int, float)) else ""
        text = (f"⚪ <b>Near-miss</b> — {nm.get('grade', '?')} {nm.get('direction', '?')}{ep} "
                f"completed but skipped: <b>{nm.get('reason', '?')}</b>{rrp}")
        try:
            sender.send(text)
        except Exception:
            pass
