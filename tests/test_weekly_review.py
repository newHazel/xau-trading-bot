"""Tests for WeeklyReviewEngine — Phase 8.5."""

import pytest
from live.weekly_review import WeeklyReviewEngine, WeeklyReviewResult, ReviewAction


@pytest.fixture
def engine():
    return WeeklyReviewEngine({
        "degradation_threshold": 0.70,
        "severe_degradation_threshold": 0.50,
        "min_trades_for_review": 3,
    })


class TestContinue:
    def test_good_performance(self, engine):
        r = engine.review(week_number=1, live_trades=5, live_win_rate=0.55,
                          live_avg_r=1.2, live_total_r=3.0,
                          backtest_win_rate=0.60, backtest_avg_r=1.5)
        assert r.action == ReviewAction.CONTINUE
        assert len(r.warnings) == 0

    def test_equal_performance(self, engine):
        r = engine.review(week_number=1, live_trades=5, live_win_rate=0.60,
                          live_avg_r=1.5, live_total_r=5.0,
                          backtest_win_rate=0.60, backtest_avg_r=1.5)
        assert r.action == ReviewAction.CONTINUE


class TestReduceSize:
    def test_moderate_degradation(self, engine):
        r = engine.review(week_number=2, live_trades=5, live_win_rate=0.38,
                          live_avg_r=0.9, live_total_r=1.0,
                          backtest_win_rate=0.60, backtest_avg_r=1.5)
        assert r.action == ReviewAction.REDUCE_SIZE
        assert any("70%" in w for w in r.warnings)


class TestPause:
    def test_severe_degradation(self, engine):
        r = engine.review(week_number=3, live_trades=5, live_win_rate=0.25,
                          live_avg_r=0.3, live_total_r=-2.0,
                          backtest_win_rate=0.60, backtest_avg_r=1.5)
        assert r.action == ReviewAction.PAUSE
        assert any("severe" in w for w in r.warnings)


class TestReviewNeeded:
    def test_negative_total_r_despite_ok_ratios(self, engine):
        r = engine.review(week_number=4, live_trades=5, live_win_rate=0.50,
                          live_avg_r=1.2, live_total_r=-0.5,
                          backtest_win_rate=0.60, backtest_avg_r=1.5)
        assert r.action == ReviewAction.REVIEW_NEEDED


class TestInsufficientTrades:
    def test_too_few_trades(self, engine):
        r = engine.review(week_number=1, live_trades=1, live_win_rate=0.0,
                          live_avg_r=-1.0, live_total_r=-1.0,
                          backtest_win_rate=0.60, backtest_avg_r=1.5)
        assert r.action == ReviewAction.CONTINUE
        assert any("insufficient" in w for w in r.warnings)


class TestWarnings:
    def test_low_win_rate_warning(self, engine):
        r = engine.review(week_number=1, live_trades=5, live_win_rate=0.35,
                          live_avg_r=1.5, live_total_r=2.0,
                          backtest_win_rate=0.60, backtest_avg_r=1.5)
        assert any("win rate below 40%" in w for w in r.warnings)

    def test_negative_avg_r_warning(self, engine):
        r = engine.review(week_number=1, live_trades=5, live_win_rate=0.50,
                          live_avg_r=-0.3, live_total_r=-1.5,
                          backtest_win_rate=0.60, backtest_avg_r=1.5)
        assert any("negative average R" in w for w in r.warnings)


class TestEdgeCases:
    def test_zero_backtest_metrics(self, engine):
        r = engine.review(week_number=1, live_trades=5, live_win_rate=0.50,
                          live_avg_r=1.0, live_total_r=2.0,
                          backtest_win_rate=0, backtest_avg_r=0)
        assert r.win_rate_ratio == 1.0
        assert r.avg_r_ratio == 1.0


class TestToDict:
    def test_to_dict(self, engine):
        r = engine.review(week_number=1, live_trades=5, live_win_rate=0.55,
                          live_avg_r=1.2, live_total_r=3.0,
                          backtest_win_rate=0.60, backtest_avg_r=1.5)
        d = r.to_dict()
        assert d["week_number"] == 1
        assert d["action"] == "continue"
        assert "warnings" in d


class TestDefaults:
    def test_default_config(self):
        e = WeeklyReviewEngine()
        r = e.review(1, 5, 0.55, 1.2, 3.0, 0.60, 1.5)
        assert r.action == ReviewAction.CONTINUE
