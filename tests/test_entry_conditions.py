"""Tests for PaperEntryConditions — Phase 7.1."""

import pytest
from paper_trading.entry_conditions import PaperEntryConditions, EntryConditionResult


@pytest.fixture
def checker():
    return PaperEntryConditions({"min_backtest_months": 6, "min_setups": 100, "require_walk_forward_passed": True})


class TestReady:
    def test_all_conditions_met(self, checker):
        r = checker.check(backtest_months=8.0, total_setups=150, walk_forward_passed=True)
        assert r.ready
        assert len(r.failed_conditions) == 0

    def test_exact_minimums(self, checker):
        r = checker.check(backtest_months=6.0, total_setups=100, walk_forward_passed=True)
        assert r.ready


class TestNotReady:
    def test_insufficient_months(self, checker):
        r = checker.check(backtest_months=4.0, total_setups=150, walk_forward_passed=True)
        assert not r.ready
        assert any("backtest_months" in f for f in r.failed_conditions)

    def test_insufficient_setups(self, checker):
        r = checker.check(backtest_months=8.0, total_setups=50, walk_forward_passed=True)
        assert not r.ready
        assert any("total_setups" in f for f in r.failed_conditions)

    def test_walk_forward_not_passed(self, checker):
        r = checker.check(backtest_months=8.0, total_setups=150, walk_forward_passed=False)
        assert not r.ready
        assert any("walk_forward" in f for f in r.failed_conditions)

    def test_multiple_failures(self, checker):
        r = checker.check(backtest_months=3.0, total_setups=50, walk_forward_passed=False)
        assert not r.ready
        assert len(r.failed_conditions) == 3


class TestCustomConfig:
    def test_relaxed_config(self):
        c = PaperEntryConditions({"min_backtest_months": 3, "min_setups": 50, "require_walk_forward_passed": False})
        r = c.check(backtest_months=3.0, total_setups=50, walk_forward_passed=False)
        assert r.ready

    def test_default_config(self):
        c = PaperEntryConditions()
        r = c.check(backtest_months=6.0, total_setups=100, walk_forward_passed=True)
        assert r.ready


class TestToDict:
    def test_to_dict(self, checker):
        r = checker.check(backtest_months=8.0, total_setups=150, walk_forward_passed=True)
        d = r.to_dict()
        assert d["ready"] is True
        assert d["backtest_months"] == 8.0
        assert "failed_conditions" in d
