"""Tests for LiveRestrictions — Phase 8.3."""

import pytest
from live.live_restrictions import LiveRestrictions, RestrictionCheckResult


@pytest.fixture
def restrictions():
    return LiveRestrictions({
        "allowed_grades": ["A+", "A"],
        "block_during_news": True,
        "max_daily_losses_live": 2,
        "block_extreme_volatility": True,
    })


class TestAllowed:
    def test_a_plus_normal(self, restrictions):
        r = restrictions.check("A+")
        assert r.allowed

    def test_a_normal(self, restrictions):
        r = restrictions.check("A")
        assert r.allowed


class TestBlocked:
    def test_bad_grade(self, restrictions):
        r = restrictions.check("B")
        assert not r.allowed
        assert any("grade" in b for b in r.blocked_reasons)

    def test_news_time(self, restrictions):
        r = restrictions.check("A+", is_news_time=True)
        assert not r.allowed
        assert any("news" in b for b in r.blocked_reasons)

    def test_max_losses(self, restrictions):
        r = restrictions.check("A+", daily_losses=2)
        assert not r.allowed
        assert any("losses" in b for b in r.blocked_reasons)

    def test_extreme_volatility(self, restrictions):
        r = restrictions.check("A+", volatility_regime="extreme")
        assert not r.allowed
        assert any("extreme" in b for b in r.blocked_reasons)

    def test_multiple_blocks(self, restrictions):
        r = restrictions.check("B", is_news_time=True, daily_losses=3, volatility_regime="extreme")
        assert not r.allowed
        assert len(r.blocked_reasons) == 4

    def test_normal_volatility_allowed(self, restrictions):
        r = restrictions.check("A+", volatility_regime="normal")
        assert r.allowed

    def test_high_volatility_allowed(self, restrictions):
        r = restrictions.check("A+", volatility_regime="high")
        assert r.allowed


class TestToDict:
    def test_to_dict(self, restrictions):
        r = restrictions.check("A+")
        d = r.to_dict()
        assert d["allowed"] is True
        assert "blocked_reasons" in d


class TestDefaults:
    def test_default_config(self):
        lr = LiveRestrictions()
        r = lr.check("A+")
        assert r.allowed
