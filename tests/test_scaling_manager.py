"""Tests for ScalingManager — Phase 8.4."""

import pytest
from live.scaling_manager import ScalingManager, ScalingReviewResult


@pytest.fixture
def manager():
    return ScalingManager({
        "review_interval_trades": 30,
        "scale_increase_pct": 25.0,
        "absolute_max_risk_pct": 1.0,
        "min_win_rate_for_scale": 0.45,
        "min_avg_r_for_scale": 0.5,
        "require_positive_total_r": True,
    })


class TestEligible:
    def test_all_criteria_met(self, manager):
        r = manager.review(current_risk_pct=0.25, live_trades=35, win_rate=0.55, avg_r=0.8, total_r=5.0)
        assert r.eligible_for_scaling
        assert r.proposed_risk_pct == 0.25 * 1.25  # +25%
        assert len(r.criteria_failed) == 0

    def test_exact_minimums(self, manager):
        r = manager.review(current_risk_pct=0.25, live_trades=30, win_rate=0.45, avg_r=0.5, total_r=0.1)
        assert r.eligible_for_scaling


class TestNotEligible:
    def test_insufficient_trades(self, manager):
        r = manager.review(current_risk_pct=0.25, live_trades=20, win_rate=0.55, avg_r=0.8, total_r=5.0)
        assert not r.eligible_for_scaling
        assert r.proposed_risk_pct == 0.25

    def test_low_win_rate(self, manager):
        r = manager.review(current_risk_pct=0.25, live_trades=35, win_rate=0.30, avg_r=0.8, total_r=5.0)
        assert not r.eligible_for_scaling

    def test_low_avg_r(self, manager):
        r = manager.review(current_risk_pct=0.25, live_trades=35, win_rate=0.55, avg_r=0.2, total_r=5.0)
        assert not r.eligible_for_scaling

    def test_negative_total_r(self, manager):
        r = manager.review(current_risk_pct=0.25, live_trades=35, win_rate=0.55, avg_r=0.8, total_r=-2.0)
        assert not r.eligible_for_scaling

    def test_multiple_failures(self, manager):
        r = manager.review(current_risk_pct=0.25, live_trades=10, win_rate=0.20, avg_r=0.1, total_r=-5.0)
        assert len(r.criteria_failed) == 4


class TestCapping:
    def test_capped_at_absolute_max(self, manager):
        r = manager.review(current_risk_pct=0.90, live_trades=35, win_rate=0.55, avg_r=0.8, total_r=5.0)
        assert r.proposed_risk_pct <= 1.0

    def test_already_at_max(self, manager):
        r = manager.review(current_risk_pct=1.0, live_trades=35, win_rate=0.55, avg_r=0.8, total_r=5.0)
        assert r.proposed_risk_pct == 1.0


class TestToDict:
    def test_to_dict(self, manager):
        r = manager.review(current_risk_pct=0.25, live_trades=35, win_rate=0.55, avg_r=0.8, total_r=5.0)
        d = r.to_dict()
        assert d["eligible_for_scaling"] is True
        assert "proposed_risk_pct" in d
        assert "criteria_met" in d


class TestDefaults:
    def test_default_config(self):
        m = ScalingManager()
        r = m.review(current_risk_pct=0.25, live_trades=30, win_rate=0.50, avg_r=0.6, total_r=3.0)
        assert r.eligible_for_scaling
