"""Tests for TelegramSender — mocked HTTP, no network."""

import pytest
from core.alerts.telegram_sender import TelegramSender


class _Resp:
    def __init__(self, ok): self._ok = ok
    def json(self): return {"ok": self._ok}


class TestConfig:
    def test_not_configured(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert TelegramSender().is_configured is False

    def test_configured(self):
        assert TelegramSender(token="t", chat_id="c").is_configured is True


class TestSend:
    def test_send_skips_when_unconfigured(self):
        assert TelegramSender(token="", chat_id="").send("hi") is False

    def test_send_ok(self, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(True))
        assert TelegramSender(token="t", chat_id="c").send("hi") is True

    def test_send_api_false(self, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(False))
        assert TelegramSender(token="t", chat_id="c").send("hi") is False

    def test_send_exception_safe(self, monkeypatch):
        import requests
        def boom(*a, **k): raise ConnectionError("no net")
        monkeypatch.setattr(requests, "post", boom)
        assert TelegramSender(token="t", chat_id="c").send("hi") is False


class TestFormat:
    def test_format_signal(self):
        msg = TelegramSender.format_signal({
            "grade": "A+", "direction": "long", "setup_id": "XAU-1",
            "entry": 2650.5, "sl": 2640, "tp1": 2670, "tp2": 2685, "rr": 3.0,
        })
        assert "A+" in msg and "LONG" in msg
        assert "2650.50" in msg
        assert "no auto-trading" in msg.lower()

    def test_format_short(self):
        msg = TelegramSender.format_signal({"grade": "A", "direction": "short", "entry": 2700})
        assert "SHORT" in msg

    def test_format_missing_fields(self):
        msg = TelegramSender.format_signal({"grade": "B", "direction": "long"})
        assert "—" in msg  # missing prices render as dash
