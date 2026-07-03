"""News calendar staleness guard — the live news gate must FAIL CLOSED when the
manual calendar has no forward visibility (newest event weeks in the past), while
historical backtests (no stale_fail_closed injection) stay unaffected.
"""

from datetime import datetime, timedelta, timezone

import pytest

from core.filters.news_filter import NewsFilter, NewsStatus


def _cfg(stale_fail_closed=False, stale_after_days=5):
    return {
        "tiers": {
            "tier_1": {"names": ["FOMC", "NFP"], "block_before_minutes": 60,
                       "block_after_minutes": 30, "degrade_grade": False},
        },
        "fallback": {
            "use_manual_csv_if_api_fails": False,  # tests load via API path
            "if_no_news_data": "degraded_mode",
            "degraded_mode_max_grade": "B",
            "stale_after_days": stale_after_days,
            "stale_fail_closed": stale_fail_closed,
        },
    }


def _event(dt, title="NFP"):
    return {"event_time": dt.isoformat(), "currency": "USD", "impact": "HIGH",
            "tier": 1, "title": title}


NOW = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)


class TestStaleFailClosed:
    def test_stale_calendar_blocks_when_fail_closed(self):
        nf = NewsFilter(_cfg(stale_fail_closed=True))
        nf.load_from_api([_event(NOW - timedelta(days=30))])
        res = nf.check(NOW)
        assert res.status == NewsStatus.BLOCKED
        assert "stale" in res.block_reason
        assert nf.is_blocked(NOW) is True

    def test_stale_calendar_clear_by_default(self):
        """Backtests (no injection) keep evaluating historical bars normally."""
        nf = NewsFilter(_cfg(stale_fail_closed=False))
        nf.load_from_api([_event(NOW - timedelta(days=30))])
        assert nf.check(NOW).status == NewsStatus.CLEAR

    def test_fresh_calendar_not_blocked_outside_window(self):
        nf = NewsFilter(_cfg(stale_fail_closed=True))
        nf.load_from_api([_event(NOW + timedelta(days=2))])
        assert nf.check(NOW).status == NewsStatus.CLEAR

    def test_fresh_calendar_still_blocks_inside_window(self):
        """The guard must not weaken the normal blackout logic."""
        nf = NewsFilter(_cfg(stale_fail_closed=True))
        nf.load_from_api([_event(NOW + timedelta(minutes=30))])
        assert nf.check(NOW).status == NewsStatus.BLOCKED

    def test_within_grace_days_not_stale(self):
        nf = NewsFilter(_cfg(stale_fail_closed=True, stale_after_days=5))
        nf.load_from_api([_event(NOW - timedelta(days=4))])
        assert nf.check(NOW).status == NewsStatus.CLEAR

    def test_historical_bar_before_events_unaffected(self):
        """A backtest bar BEFORE the newest event is never 'stale' (negative age)."""
        nf = NewsFilter(_cfg(stale_fail_closed=True))
        nf.load_from_api([_event(NOW + timedelta(days=200), title="CPI m/m")])
        assert nf.check(NOW).status == NewsStatus.CLEAR


class TestLiveEngineInjection:
    def test_injection_sets_fail_closed_and_preserves_config(self):
        from core.alerts.live_engine import _inject_live_news_policy

        cfg = {"news": {"tiers": {"tier_1": {}},
                        "fallback": {"manual_csv_path": "x.csv", "stale_after_days": 7}},
               "risk": {"a": 1}}
        out = _inject_live_news_policy(cfg)
        assert out["news"]["fallback"]["stale_fail_closed"] is True
        # existing fallback keys and sibling sections survive the injection
        assert out["news"]["fallback"]["manual_csv_path"] == "x.csv"
        assert out["news"]["fallback"]["stale_after_days"] == 7
        assert out["news"]["tiers"] == {"tier_1": {}}
        assert out["risk"] == {"a": 1}
        # the original dict is not mutated
        assert "stale_fail_closed" not in cfg["news"]["fallback"]

    def test_injection_handles_missing_news_section(self):
        from core.alerts.live_engine import _inject_live_news_policy

        out = _inject_live_news_policy({})
        assert out["news"]["fallback"]["stale_fail_closed"] is True

    def test_newsfilter_reads_injected_flag(self):
        from core.alerts.live_engine import _inject_live_news_policy

        cfg = _inject_live_news_policy({"news": _cfg(stale_fail_closed=False)})
        nf = NewsFilter(cfg["news"])
        nf.load_from_api([_event(NOW - timedelta(days=30))])
        assert nf.check(NOW).status == NewsStatus.BLOCKED
