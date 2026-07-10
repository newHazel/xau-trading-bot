"""Tests for News Filter + Tiers — Phase 3.3."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from core.filters.news_filter import NewsFilter, NewsStatus, NewsEvent
from core.filters.news_tiers import NewsTiers

UTC = ZoneInfo("Etc/UTC")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

NEWS_CONFIG = {
    "tiers": {
        "tier_1": {
            "events": ["FOMC", "Interest Rate Decision", "NFP", "Powell Speech"],
            "block_before_minutes": 60,
            "block_after_minutes": 60,
        },
        "tier_2": {
            "events": ["CPI", "PCE", "Core CPI", "Core PCE"],
            "block_before_minutes": 45,
            "block_after_minutes": 45,
        },
        "tier_3": {
            "events": ["Unemployment Rate", "Retail Sales", "GDP", "ISM Manufacturing"],
            "block_before_minutes": 30,
            "block_after_minutes": 30,
        },
        "tier_4": {
            "events": ["PPI", "Building Permits", "Consumer Confidence"],
            "block_before_minutes": 0,
            "block_after_minutes": 0,
            "degrade_grade": True,
        },
    },
    "fallback": {
        "use_manual_csv_if_api_fails": True,
        "manual_csv_path": "data/calendar/manual_news.csv",
        "if_no_news_data": "degraded_mode",
        "degraded_mode_max_grade": "B",
    },
}


# ── NewsTiers tests ──────────────────────────────────────────────


class TestNewsTiers:
    @pytest.fixture
    def tiers(self):
        return NewsTiers(NEWS_CONFIG)

    def test_classify_fomc(self, tiers):
        tc = tiers.classify("FOMC Meeting Minutes")
        assert tc is not None
        assert tc.tier == 1
        assert tc.block_before_minutes == 60

    def test_classify_cpi(self, tiers):
        tc = tiers.classify("CPI m/m")
        assert tc is not None
        assert tc.tier == 2

    def test_classify_gdp(self, tiers):
        tc = tiers.classify("GDP Preliminary")
        assert tc is not None
        assert tc.tier == 3
        assert tc.block_before_minutes == 30

    def test_classify_ppi(self, tiers):
        tc = tiers.classify("PPI Final")
        assert tc is not None
        assert tc.tier == 4
        assert tc.degrade_grade is True
        assert tc.block_before_minutes == 0

    def test_classify_unknown(self, tiers):
        assert tiers.classify("Random Event XYZ") is None

    def test_case_insensitive(self, tiers):
        tc = tiers.classify("fomc rate decision")
        assert tc is not None
        assert tc.tier == 1

    def test_get_tier_by_number(self, tiers):
        tc = tiers.get_tier(2)
        assert tc is not None
        assert tc.tier == 2

    def test_all_tiers(self, tiers):
        assert len(tiers.all_tiers) == 4


# ── NewsFilter tests ─────────────────────────────────────────────


@pytest.fixture
def nf():
    return NewsFilter(NEWS_CONFIG, project_root=PROJECT_ROOT)


def _make_events(events_data):
    """Helper to load events via API dict format."""
    nf = NewsFilter(NEWS_CONFIG, project_root=PROJECT_ROOT)
    nf.load_from_api(events_data)
    return nf


class TestTier1Blocking:
    def test_blocked_30min_before_fomc(self):
        fomc_time = datetime(2026, 5, 6, 18, 0, tzinfo=UTC)
        nf = _make_events([{
            "event_time": fomc_time,
            "currency": "USD",
            "impact": "HIGH",
            "tier": 1,
            "title": "FOMC Meeting Minutes",
        }])
        check_time = fomc_time - timedelta(minutes=30)
        result = nf.check(check_time)
        assert result.status == NewsStatus.BLOCKED
        assert result.nearest_tier == 1

    def test_blocked_30min_after_fomc(self):
        fomc_time = datetime(2026, 5, 6, 18, 0, tzinfo=UTC)
        nf = _make_events([{
            "event_time": fomc_time,
            "currency": "USD",
            "impact": "HIGH",
            "tier": 1,
            "title": "FOMC Meeting Minutes",
        }])
        check_time = fomc_time + timedelta(minutes=30)
        result = nf.check(check_time)
        assert result.status == NewsStatus.BLOCKED

    def test_clear_90min_before_fomc(self):
        fomc_time = datetime(2026, 5, 6, 18, 0, tzinfo=UTC)
        nf = _make_events([{
            "event_time": fomc_time,
            "currency": "USD",
            "impact": "HIGH",
            "tier": 1,
            "title": "FOMC Meeting Minutes",
        }])
        check_time = fomc_time - timedelta(minutes=90)
        result = nf.check(check_time)
        assert result.status == NewsStatus.CLEAR

    def test_blocked_exactly_at_event(self):
        fomc_time = datetime(2026, 5, 6, 18, 0, tzinfo=UTC)
        nf = _make_events([{
            "event_time": fomc_time,
            "currency": "USD",
            "impact": "HIGH",
            "tier": 1,
            "title": "FOMC Meeting Minutes",
        }])
        result = nf.check(fomc_time)
        assert result.status == NewsStatus.BLOCKED


class TestTier2Blocking:
    def test_cpi_blocked_40min_before(self):
        cpi_time = datetime(2026, 5, 9, 12, 30, tzinfo=UTC)
        nf = _make_events([{
            "event_time": cpi_time,
            "currency": "USD",
            "impact": "HIGH",
            "tier": 2,
            "title": "CPI m/m",
        }])
        check_time = cpi_time - timedelta(minutes=40)
        result = nf.check(check_time)
        assert result.status == NewsStatus.BLOCKED
        assert result.nearest_tier == 2

    def test_cpi_clear_50min_before(self):
        cpi_time = datetime(2026, 5, 9, 12, 30, tzinfo=UTC)
        nf = _make_events([{
            "event_time": cpi_time,
            "currency": "USD",
            "impact": "HIGH",
            "tier": 2,
            "title": "CPI m/m",
        }])
        check_time = cpi_time - timedelta(minutes=50)
        result = nf.check(check_time)
        assert result.status == NewsStatus.CLEAR


class TestTier3Blocking:
    def test_gdp_blocked_25min_before(self):
        gdp_time = datetime(2026, 6, 10, 12, 30, tzinfo=UTC)
        nf = _make_events([{
            "event_time": gdp_time,
            "currency": "USD",
            "impact": "MEDIUM",
            "tier": 3,
            "title": "GDP Preliminary",
        }])
        check_time = gdp_time - timedelta(minutes=25)
        result = nf.check(check_time)
        assert result.status == NewsStatus.BLOCKED

    def test_gdp_clear_35min_before(self):
        gdp_time = datetime(2026, 6, 10, 12, 30, tzinfo=UTC)
        nf = _make_events([{
            "event_time": gdp_time,
            "currency": "USD",
            "impact": "MEDIUM",
            "tier": 3,
            "title": "GDP Preliminary",
        }])
        check_time = gdp_time - timedelta(minutes=35)
        result = nf.check(check_time)
        assert result.status == NewsStatus.CLEAR


class TestTier4Degraded:
    def test_ppi_degrades_grade(self):
        ppi_time = datetime(2026, 6, 15, 12, 30, tzinfo=UTC)
        nf = _make_events([{
            "event_time": ppi_time,
            "currency": "USD",
            "impact": "LOW",
            "tier": 4,
            "title": "PPI Final",
        }])
        result = nf.check(ppi_time)
        assert result.status == NewsStatus.DEGRADED
        assert result.max_grade == "B"

    def test_ppi_never_blocks(self):
        ppi_time = datetime(2026, 6, 15, 12, 30, tzinfo=UTC)
        nf = _make_events([{
            "event_time": ppi_time,
            "currency": "USD",
            "impact": "LOW",
            "tier": 4,
            "title": "PPI Final",
        }])
        assert nf.is_blocked(ppi_time) is False


# data/calendar/manual_news.csv is a ROLLING operational file (rewritten weekly by
# scripts/update_news_calendar.py), so CSV tests pin their OWN fixture instead of
# asserting on live calendar content (which broke every refresh).
_FIXTURE_CSV = """event_time,currency,impact,tier,title,actual,forecast,previous
2026-05-06T18:00:00Z,USD,HIGH,1,FOMC Meeting Minutes,,,
2026-05-09T12:30:00Z,USD,HIGH,2,CPI m/m,,,
2026-06-03T12:30:00Z,USD,HIGH,1,NFP,,,
"""


@pytest.fixture
def fixture_csv(tmp_path):
    p = tmp_path / "manual_news.csv"
    p.write_text(_FIXTURE_CSV)
    return str(p)


class TestCSVFallback:
    def test_load_from_csv(self, fixture_csv):
        nf = NewsFilter(NEWS_CONFIG)
        loaded = nf.load_from_csv(fixture_csv)
        assert loaded is True
        assert nf.data_loaded is True
        assert len(nf.events) == 3

    def test_csv_events_have_correct_tiers(self, fixture_csv):
        nf = NewsFilter(NEWS_CONFIG)
        nf.load_from_csv(fixture_csv)
        fomc = [e for e in nf.events if "FOMC" in e.title]
        assert len(fomc) == 1
        assert fomc[0].tier == 1

    def test_csv_fomc_blocks(self, fixture_csv):
        nf = NewsFilter(NEWS_CONFIG)
        nf.load_from_csv(fixture_csv)
        fomc_time = datetime(2026, 5, 6, 18, 0, tzinfo=UTC)
        check_time = fomc_time - timedelta(minutes=30)
        result = nf.check(check_time)
        assert result.status == NewsStatus.BLOCKED

    def test_missing_csv_returns_false(self):
        nf = NewsFilter(NEWS_CONFIG, project_root=PROJECT_ROOT)
        loaded = nf.load_from_csv("data/calendar/nonexistent.csv")
        assert loaded is False

    def test_auto_fallback_on_check(self, fixture_csv):
        # check() auto-loads via ensure_loaded() from the configured fallback path
        cfg = {**NEWS_CONFIG,
               "fallback": {**NEWS_CONFIG.get("fallback", {}),
                            "use_manual_csv_if_api_fails": True,
                            "manual_csv_path": fixture_csv}}
        nf = NewsFilter(cfg)
        fomc_time = datetime(2026, 5, 6, 18, 0, tzinfo=UTC)
        check_time = fomc_time - timedelta(minutes=30)
        result = nf.check(check_time)
        assert nf.data_loaded is True
        assert result.status == NewsStatus.BLOCKED


class TestNoNewsData:
    def test_degraded_mode_when_no_data(self):
        config = {
            **NEWS_CONFIG,
            "fallback": {
                "use_manual_csv_if_api_fails": False,
                "if_no_news_data": "degraded_mode",
                "degraded_mode_max_grade": "B",
            },
        }
        nf = NewsFilter(config)
        dt = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
        result = nf.check(dt)
        assert result.status == NewsStatus.DEGRADED
        assert result.max_grade == "B"

    def test_clear_mode_when_configured(self):
        config = {
            **NEWS_CONFIG,
            "fallback": {
                "use_manual_csv_if_api_fails": False,
                "if_no_news_data": "clear",
                "degraded_mode_max_grade": "B",
            },
        }
        nf = NewsFilter(config)
        dt = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
        result = nf.check(dt)
        assert result.status == NewsStatus.CLEAR


class TestMultipleEvents:
    def test_nearest_event_wins(self):
        nf = _make_events([
            {
                "event_time": datetime(2026, 5, 6, 12, 0, tzinfo=UTC),
                "title": "GDP Preliminary",
                "tier": 3,
            },
            {
                "event_time": datetime(2026, 5, 6, 18, 0, tzinfo=UTC),
                "title": "FOMC Meeting Minutes",
                "tier": 1,
            },
        ])
        # 11:45 UTC → 15min before GDP (tier 3, 30min window) → blocked
        check_time = datetime(2026, 5, 6, 11, 45, tzinfo=UTC)
        result = nf.check(check_time)
        assert result.status == NewsStatus.BLOCKED
        assert result.nearest_tier == 3

    def test_between_events_clear(self):
        nf = _make_events([
            {
                "event_time": datetime(2026, 5, 6, 10, 0, tzinfo=UTC),
                "title": "GDP Preliminary",
                "tier": 3,
            },
            {
                "event_time": datetime(2026, 5, 6, 18, 0, tzinfo=UTC),
                "title": "FOMC Meeting Minutes",
                "tier": 1,
            },
        ])
        # 14:00 UTC → 4h after GDP, 4h before FOMC → clear
        check_time = datetime(2026, 5, 6, 14, 0, tzinfo=UTC)
        result = nf.check(check_time)
        assert result.status == NewsStatus.CLEAR


class TestIsBlocked:
    def test_is_blocked_true(self):
        nf = _make_events([{
            "event_time": datetime(2026, 5, 6, 18, 0, tzinfo=UTC),
            "title": "FOMC Meeting Minutes",
            "tier": 1,
        }])
        check_time = datetime(2026, 5, 6, 17, 30, tzinfo=UTC)
        assert nf.is_blocked(check_time) is True

    def test_is_blocked_false(self):
        nf = _make_events([{
            "event_time": datetime(2026, 5, 6, 18, 0, tzinfo=UTC),
            "title": "FOMC Meeting Minutes",
            "tier": 1,
        }])
        check_time = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
        assert nf.is_blocked(check_time) is False


class TestResultDict:
    def test_blocked_to_dict(self):
        nf = _make_events([{
            "event_time": datetime(2026, 5, 6, 18, 0, tzinfo=UTC),
            "title": "NFP",
            "tier": 1,
        }])
        result = nf.check(datetime(2026, 5, 6, 17, 30, tzinfo=UTC))
        d = result.to_dict()
        assert d["status"] == "blocked"
        assert d["nearest_tier"] == 1
        assert d["nearest_event_title"] == "NFP"

    def test_clear_to_dict(self):
        nf = _make_events([{
            "event_time": datetime(2026, 5, 6, 18, 0, tzinfo=UTC),
            "title": "NFP",
            "tier": 1,
        }])
        result = nf.check(datetime(2026, 5, 6, 12, 0, tzinfo=UTC))
        d = result.to_dict()
        assert d["status"] == "clear"
        assert d["block_reason"] is None


class TestBoundaryConditions:
    def test_exactly_at_boundary_before(self):
        """Exactly 60min before tier 1 → still blocked (<=)."""
        event_time = datetime(2026, 5, 6, 18, 0, tzinfo=UTC)
        nf = _make_events([{
            "event_time": event_time,
            "title": "FOMC",
            "tier": 1,
        }])
        check_time = event_time - timedelta(minutes=60)
        result = nf.check(check_time)
        assert result.status == NewsStatus.BLOCKED

    def test_one_minute_past_boundary(self):
        """61min before tier 1 → clear."""
        event_time = datetime(2026, 5, 6, 18, 0, tzinfo=UTC)
        nf = _make_events([{
            "event_time": event_time,
            "title": "FOMC",
            "tier": 1,
        }])
        check_time = event_time - timedelta(minutes=61)
        result = nf.check(check_time)
        assert result.status == NewsStatus.CLEAR
