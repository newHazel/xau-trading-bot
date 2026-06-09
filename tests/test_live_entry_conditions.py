"""Tests for LiveEntryConditions — Phase 8.1."""

import pytest
from live.live_entry_conditions import LiveEntryConditions, LiveEntryResult


@pytest.fixture
def checker():
    return LiveEntryConditions({
        "min_paper_trades": 20,
        "require_zero_violations": True,
        "require_paper_stats_passed": True,
        "min_paper_win_rate": 0.45,
    })


class TestReady:
    def test_all_met(self, checker):
        r = checker.check(paper_trades=25, paper_win_rate=0.55, paper_violations=0, paper_stats_passed=True)
        assert r.ready
        assert len(r.failed_conditions) == 0

    def test_exact_minimums(self, checker):
        r = checker.check(paper_trades=20, paper_win_rate=0.45, paper_violations=0, paper_stats_passed=True)
        assert r.ready


class TestNotReady:
    def test_insufficient_trades(self, checker):
        r = checker.check(paper_trades=15, paper_win_rate=0.55, paper_violations=0, paper_stats_passed=True)
        assert not r.ready
        assert any("paper_trades" in f for f in r.failed_conditions)

    def test_low_win_rate(self, checker):
        r = checker.check(paper_trades=25, paper_win_rate=0.30, paper_violations=0, paper_stats_passed=True)
        assert not r.ready

    def test_has_violations(self, checker):
        r = checker.check(paper_trades=25, paper_win_rate=0.55, paper_violations=3, paper_stats_passed=True)
        assert not r.ready

    def test_stats_not_passed(self, checker):
        r = checker.check(paper_trades=25, paper_win_rate=0.55, paper_violations=0, paper_stats_passed=False)
        assert not r.ready

    def test_all_fail(self, checker):
        r = checker.check(paper_trades=5, paper_win_rate=0.20, paper_violations=5, paper_stats_passed=False)
        assert not r.ready
        assert len(r.failed_conditions) == 4


class TestToDict:
    def test_to_dict(self, checker):
        r = checker.check(paper_trades=25, paper_win_rate=0.55, paper_violations=0, paper_stats_passed=True)
        d = r.to_dict()
        assert d["ready"] is True
        assert d["paper_trades"] == 25


class TestDefaults:
    def test_default_config(self):
        c = LiveEntryConditions()
        r = c.check(paper_trades=20, paper_win_rate=0.45, paper_violations=0, paper_stats_passed=True)
        assert r.ready
