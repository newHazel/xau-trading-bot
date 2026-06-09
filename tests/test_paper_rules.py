"""Tests for PaperRules — Phase 7.5."""

import pytest
from paper_trading.paper_rules import PaperRules, GraduationResult


@pytest.fixture
def rules():
    return PaperRules({
        "min_paper_trades": 20,
        "max_paper_trades": 30,
        "allowed_grades": ["A+", "A"],
        "require_zero_violations": True,
        "min_win_rate_for_graduation": 0.45,
    })


class TestSignalAllowed:
    def test_a_plus_allowed(self, rules):
        r = rules.check_signal_allowed("A+")
        assert r["allowed"]

    def test_a_allowed(self, rules):
        r = rules.check_signal_allowed("A")
        assert r["allowed"]

    def test_b_not_allowed(self, rules):
        r = rules.check_signal_allowed("B")
        assert not r["allowed"]
        assert "grade" in r["reason"]

    def test_c_not_allowed(self, rules):
        r = rules.check_signal_allowed("C")
        assert not r["allowed"]


class TestGraduationReady:
    def test_all_criteria_met(self, rules):
        r = rules.check_graduation(
            total_trades=25,
            violations_count=0,
            grade_distribution={"A+": 15, "A": 10},
            win_rate=0.55,
            paper_stats_passed=True,
        )
        assert r.ready
        assert len(r.failed_rules) == 0

    def test_exact_minimums(self, rules):
        r = rules.check_graduation(
            total_trades=20,
            violations_count=0,
            grade_distribution={"A+": 10, "A": 10},
            win_rate=0.45,
            paper_stats_passed=True,
        )
        assert r.ready


class TestGraduationNotReady:
    def test_insufficient_trades(self, rules):
        r = rules.check_graduation(
            total_trades=15,
            violations_count=0,
            grade_distribution={"A+": 10, "A": 5},
            win_rate=0.55,
            paper_stats_passed=True,
        )
        assert not r.ready
        assert any("trades" in f for f in r.failed_rules)

    def test_has_violations(self, rules):
        r = rules.check_graduation(
            total_trades=25,
            violations_count=2,
            grade_distribution={"A+": 15, "A": 10},
            win_rate=0.55,
            paper_stats_passed=True,
        )
        assert not r.ready
        assert any("violations" in f for f in r.failed_rules)

    def test_disallowed_grades_traded(self, rules):
        r = rules.check_graduation(
            total_trades=25,
            violations_count=0,
            grade_distribution={"A+": 10, "A": 10, "B": 5},
            win_rate=0.55,
            paper_stats_passed=True,
        )
        assert not r.ready
        assert any("disallowed" in f for f in r.failed_rules)

    def test_low_win_rate(self, rules):
        r = rules.check_graduation(
            total_trades=25,
            violations_count=0,
            grade_distribution={"A+": 15, "A": 10},
            win_rate=0.30,
            paper_stats_passed=True,
        )
        assert not r.ready
        assert any("win_rate" in f for f in r.failed_rules)

    def test_paper_stats_failed(self, rules):
        r = rules.check_graduation(
            total_trades=25,
            violations_count=0,
            grade_distribution={"A+": 15, "A": 10},
            win_rate=0.55,
            paper_stats_passed=False,
        )
        assert not r.ready
        assert any("paper stats" in f for f in r.failed_rules)

    def test_multiple_failures(self, rules):
        r = rules.check_graduation(
            total_trades=10,
            violations_count=3,
            grade_distribution={"B": 10},
            win_rate=0.20,
            paper_stats_passed=False,
        )
        assert not r.ready
        assert len(r.failed_rules) >= 4


class TestDefaultConfig:
    def test_default_rules(self):
        r = PaperRules()
        assert r.check_signal_allowed("A+")["allowed"]
        assert not r.check_signal_allowed("B")["allowed"]


class TestToDict:
    def test_graduation_to_dict(self, rules):
        r = rules.check_graduation(
            total_trades=25,
            violations_count=0,
            grade_distribution={"A+": 15, "A": 10},
            win_rate=0.55,
            paper_stats_passed=True,
        )
        d = r.to_dict()
        assert d["ready"] is True
        assert "total_trades" in d
        assert "grade_distribution" in d
