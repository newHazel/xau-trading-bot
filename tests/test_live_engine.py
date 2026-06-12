"""Tests for LiveAlertEngine — fully mocked (no network, no DB writes)."""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from core.alerts.live_engine import LiveAlertEngine, LiveConfig

NOW = datetime(2026, 1, 21, 16, 0, tzinfo=timezone.utc)

CONFIG = {
    "rr_tiers": {"min_to_enter": 2.0, "required_for_grade_b": 1.5,
                 "required_for_grade_a": 2.0, "required_for_grade_a_plus": 2.5},
    "costs": {"default_spread": 0.25, "default_slippage": 0.10},
}


class _Status:
    value = "ok"


class _Res:
    def __init__(self, df): self.status = _Status(); self.data = df


def _df(n=60):
    idx = pd.date_range("2026-01-20", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": 2650.0, "high": 2655.0, "low": 2645.0,
                         "close": 2650.0, "volume": 100.0}, index=idx)


class _FakeFetcher:
    def __init__(self, df): self._df = df
    def fetch_latest_candles(self, symbol, tf, count): return _Res(self._df)


class _CapturingSender:
    def __init__(self): self.sent = []
    @property
    def is_configured(self): return True
    def send(self, text, parse_mode="HTML"): self.sent.append(text); return True
    def send_signal(self, signal): self.sent.append(signal); return True


def _engine(sender=None, fetcher=None):
    return LiveAlertEngine(
        config=CONFIG,
        live=LiveConfig(symbol="XAUUSD", execution_tf="15m"),
        fetcher=fetcher or _FakeFetcher(_df()),
        sender=sender or _CapturingSender(),
        now_fn=lambda: NOW,
    )


class TestFetchHistory:
    def test_builds_history_dict(self):
        eng = _engine()
        hist = eng.fetch_history(NOW)
        assert "15m" in hist
        assert not hist["15m"].empty

    def test_htf_cached_within_window(self):
        # cycle 1 fetches all 4 TFs; cycle 2 (same minute) reuses cached HTF,
        # fetching only the 2 fast TFs → 4 + 2 = 6 total (not 8).
        eng = _engine()
        eng.fetch_history(NOW)
        assert eng.fetch_count == 4
        eng.fetch_history(NOW)
        assert eng.fetch_count == 6
        assert "4h" in eng.fetch_history(NOW)  # HTF still present from cache

    def test_htf_refetched_after_window(self):
        eng = _engine()
        eng.fetch_history(NOW)                       # 4
        eng.fetch_history(NOW + timedelta(minutes=61))  # HTF due again → +4
        assert eng.fetch_count == 8


class TestCheckOnce:
    def test_no_signal_on_flat_data(self):
        # flat candles → no complete setup → no alert
        sender = _CapturingSender()
        eng = _engine(sender=sender)
        result = eng.check_once(eng.fetch_history(), NOW)
        assert result is None
        assert eng.alerts_sent == 0

    def test_check_once_handles_empty_history(self):
        eng = _engine()
        assert eng.check_once({}, NOW) is None


class TestHeartbeat:
    def test_first_call_sends_startup(self):
        sender = _CapturingSender()
        eng = _engine(sender=sender)
        sent = eng.maybe_heartbeat(now=NOW)
        assert sent is True            # startup heartbeat fires immediately
        assert len(sender.sent) == 1

    def test_due_after_interval(self):
        sender = _CapturingSender()
        eng = _engine(sender=sender)
        eng.maybe_heartbeat(now=NOW)                        # startup (1)
        sent = eng.maybe_heartbeat(now=NOW + timedelta(minutes=61))
        assert sent is True                                 # interval (2)
        assert len(sender.sent) == 2

    def test_not_due_before_interval(self):
        eng = _engine()
        eng.maybe_heartbeat(now=NOW)                        # startup
        assert eng.maybe_heartbeat(now=NOW + timedelta(minutes=10)) is False

    def test_heartbeat_survives_html_special_chars_in_near_miss(self):
        # Regression: a near-miss reason like "R:R < 2 net" contains '<'. When the
        # heartbeat embedded it and sent with parse_mode="HTML", Telegram rejected the
        # whole message and the heartbeat silently stopped. The heartbeat must go out
        # as plain text so a stray '<' can never kill it.
        class _HtmlStrictSender:
            def __init__(self): self.sent = []
            @property
            def is_configured(self): return True
            def send(self, text, parse_mode="HTML"):
                # mimic Telegram: in HTML mode an unescaped '<' is a parse error → drop
                if parse_mode == "HTML" and "<" in text:
                    return False
                self.sent.append((text, parse_mode))
                return True

        sender = _HtmlStrictSender()
        eng = _engine(sender=sender)
        eng._near_miss.record({"reason": "R:R < 2 net", "grade": "B",
                               "direction": "long"})           # most common rejection
        sent = eng.maybe_heartbeat(now=NOW)
        assert sent is True
        assert len(sender.sent) == 1                            # delivered, not dropped
        text, parse_mode = sender.sent[0]
        assert parse_mode == ""                                 # plain text, no HTML
        assert "R:R < 2 net" in text


class TestRunCycle:
    def test_run_cycle_no_crash(self):
        eng = _engine()
        # flat data → returns None, but the full cycle (fetch+check+heartbeat) runs
        assert eng.run_cycle() is None


class TestAlertPath:
    """Force a tradeable signal via a stub pipeline to verify the alert path."""

    def test_alert_sent_and_deduped(self, monkeypatch):
        sender = _CapturingSender()
        eng = _engine(sender=sender)

        class _Grade:
            net_rr = 3.0
        class _Decision:
            grade = _Grade()
        class _Sig:
            setup_id = "XAU-TEST-1"; direction = "long"; grade = "A+"
            entry = 2650.0; sl = 2640.0; tp1 = 2670.0; tp2 = 2685.0
            score = 65; approved = True; timestamp = NOW; decision = _Decision()

        monkeypatch.setattr(eng._runner, "on_bar", lambda bar, hist: _Sig())

        first = eng.check_once(eng.fetch_history(), NOW)
        assert first is not None
        assert eng.alerts_sent == 1
        assert len(sender.sent) == 1

        # same setup, same minute → deduped (no second alert)
        second = eng.check_once(eng.fetch_history(), NOW)
        assert second is None
        assert eng.alerts_sent == 1

    def test_c_grade_not_alerted(self, monkeypatch):
        # B is now tradeable; C/D are not — verify C is filtered out.
        sender = _CapturingSender()
        eng = _engine(sender=sender)

        class _Sig:
            setup_id = "x"; direction = "long"; grade = "C"
            entry = 2650.0; sl = 2640.0; tp1 = 2670.0; tp2 = 2685.0
            score = 10; approved = True; timestamp = NOW; decision = None

        monkeypatch.setattr(eng._runner, "on_bar", lambda bar, hist: _Sig())
        assert eng.check_once(eng.fetch_history(), NOW) is None
        assert eng.alerts_sent == 0
