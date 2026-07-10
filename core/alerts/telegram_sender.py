"""
Telegram sender — the real transport for alerts.

Reads TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID from the environment (.env) and
posts messages to the Telegram Bot API. Plugs into the Phase-9 FailureAlerter /
HeartbeatManager via `send` (signature: str -> bool).

Setup (one time):
    1. In Telegram, message @BotFather → /newbot → get the BOT TOKEN.
    2. Message your new bot once (say "hi"), then get your CHAT ID:
       open  https://api.telegram.org/bot<TOKEN>/getUpdates  → find "chat":{"id":...}
    3. Put both in .env:
         TELEGRAM_BOT_TOKEN=...
         TELEGRAM_CHAT_ID=...
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramSender:
    """Sends alert messages to a Telegram chat via the Bot API."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None) -> None:
        self._token = token if token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")

    @property
    def is_configured(self) -> bool:
        return bool(self._token) and bool(self._chat_id)

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message. Returns True on success. Matches FailureAlerter's hook."""
        if not self.is_configured:
            return False
        try:
            import requests
            resp = requests.post(
                _API.format(token=self._token),
                json={"chat_id": self._chat_id, "text": text,
                      "parse_mode": parse_mode, "disable_web_page_preview": True},
                timeout=10,
            )
            return bool(resp.json().get("ok", False))
        except Exception:
            return False

    def send_signal(self, signal: Dict[str, Any]) -> bool:
        return self.send(self.format_signal(signal))

    # ------------------------------------------------------------------ #

    @staticmethod
    def format_signal(s: Dict[str, Any]) -> str:
        """Format a graded signal into a readable Telegram alert."""
        grade = s.get("grade", "?")
        emoji = {"A+": "🟢🟢", "A": "🟢", "B": "🟡"}.get(grade, "⚪")
        direction = str(s.get("direction", "")).upper()
        arrow = "⬆️" if direction == "LONG" else "⬇️"
        rr = s.get("rr") or s.get("net_rr") or 0.0
        ticker = ticker_label(s.get("symbol")) or "XAU"
        lines = [
            f"{emoji} <b>{ticker} {direction}</b> {arrow}  —  Grade <b>{grade}</b>",
            f"Setup: <code>{s.get('setup_id', '')}</code>",
            "",
            f"Entry: <b>{_fmt(s.get('entry'))}</b>",
            f"SL:    {_fmt(s.get('sl') or s.get('stop_loss'))}",
            f"TP1:   {_fmt(s.get('tp1'))}",
            f"TP2:   {_fmt(s.get('tp2'))}",
            f"R:R:   {rr:.2f}",
        ]
        src = s.get("sweep_src")
        if src:
            lines.append(f"Sweep: {str(src).replace('_', ' ').upper()}")
        ts = s.get("timestamp")
        if ts:
            lines.append(f"\n🕐 {ts}")
        lines.append("\n<i>Alert only — no auto-trading. Verify before entering.</i>")
        return "\n".join(lines)


def fmt_price(v: Any) -> str:
    """Adaptive price precision so cheap coins don't collapse to identical levels.
    Gold/SOL/ETH (>=100) keep 2 decimals (byte-identical to before); LINK/NEAR
    (1-100) get 4; sub-dollar coins like DOGE/SAND (<1) get 5."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    a = abs(x)
    if a >= 100:
        return f"{x:.2f}"
    if a >= 1:
        return f"{x:.4f}"
    return f"{x:.5f}"


def ticker_label(symbol: Any) -> str:
    """ETHUSDT -> ETH, XAUUSD -> XAU (clean ticker for alert headers/labels)."""
    s = str(symbol or "").upper()
    for suf in ("USDT", "USD"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


# backward-compatible alias (was the only formatter before crypto support)
_fmt = fmt_price
