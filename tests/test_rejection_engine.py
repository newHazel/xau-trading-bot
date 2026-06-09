"""Tests for Rejection Engine — Phase 4.3."""

import pytest
from datetime import datetime
from core.engine.rejection_engine import RejectionEngine


@pytest.fixture
def re():
    return RejectionEngine()


class TestReject:
    def test_creates_rejection(self, re):
        r = re.reject(
            symbol="XAUUSD",
            timestamp=datetime(2026, 3, 18, 12, 0),
            attempted_direction="long",
            failed_conditions=["news_clear", "kill_zone"],
            passed_conditions=["htf_bias", "sweep"],
        )
        assert r.main_reason == "news_clear"
        assert len(r.failed_conditions) == 2
        assert len(r.passed_conditions) == 2
        assert re.count == 1

    def test_main_reason_is_first_failure(self, re):
        r = re.reject(
            symbol="XAUUSD",
            timestamp=datetime(2026, 3, 18, 12, 0),
            attempted_direction="short",
            failed_conditions=["htf_bias"],
            passed_conditions=[],
        )
        assert r.main_reason == "htf_bias"

    def test_with_context(self, re):
        r = re.reject(
            symbol="XAUUSD",
            timestamp=datetime(2026, 3, 18, 12, 0),
            attempted_direction="long",
            failed_conditions=["fvg_valid"],
            passed_conditions=["htf_bias"],
            context={"fvg_mitigation": 0.65},
            setup_id="XAU-20260318-1200-LONG-FVG1",
        )
        assert r.context["fvg_mitigation"] == 0.65
        assert r.setup_id is not None


class TestQueries:
    def test_get_by_direction(self, re):
        re.reject("XAUUSD", datetime.now(), "long", ["a"], [])
        re.reject("XAUUSD", datetime.now(), "short", ["b"], [])
        re.reject("XAUUSD", datetime.now(), "long", ["c"], [])
        assert len(re.get_by_direction("long")) == 2
        assert len(re.get_by_direction("short")) == 1

    def test_get_recent(self, re):
        for i in range(20):
            re.reject("XAUUSD", datetime.now(), "long", [f"r{i}"], [])
        recent = re.get_recent(5)
        assert len(recent) == 5
        assert recent[-1].main_reason == "r19"


class TestToDict:
    def test_to_dict(self, re):
        r = re.reject("XAUUSD", datetime(2026, 3, 18, 12, 0), "long", ["a"], ["b"])
        d = r.to_dict()
        assert d["symbol"] == "XAUUSD"
        assert d["main_reason"] == "a"
        assert "timestamp" in d


class TestReset:
    def test_reset(self, re):
        re.reject("XAUUSD", datetime.now(), "long", ["a"], [])
        re.reset()
        assert re.count == 0
