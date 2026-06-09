"""Tests for Signal Grader — Phase 4.2 + Phase 11 indicator boosters."""

import pytest
from core.engine.signal_grader import SignalGrader, OPTIONAL_SCORES, INDICATOR_SCORES

CONFIG = {
    "rr_tiers": {
        "min_to_enter": 2.0,
        "required_for_grade_b": 1.5,
        "required_for_grade_a": 2.0,
        "required_for_grade_a_plus": 2.5,
    },
}

ALL_MANDATORY = {
    "htf_bias": True, "15m_aligned": True, "price_zone": True,
    "sweep": True, "sweep_confirmation": True, "fvg_valid": True,
    "fvg_freshness": True, "kill_zone": True, "news_clear": True,
    "retrace_to_zone": True, "micro_choch": True,
    "confirmation_candle": True, "rr_minimum": True,
    "daily_limits_ok": True, "no_blocking_filters": True,
}

ALL_OPTIONAL = {k: True for k in OPTIONAL_SCORES}
NO_OPTIONAL = {k: False for k in OPTIONAL_SCORES}


@pytest.fixture
def grader():
    return SignalGrader(CONFIG)


class TestGradeAPlus:
    def test_a_plus_all_conditions(self, grader):
        result = grader.grade(ALL_MANDATORY, ALL_OPTIONAL, net_rr=3.0)
        assert result.grade == "A+"
        assert result.mandatory_passed is True
        assert result.score == 50

    def test_a_plus_needs_fvg_fresh(self, grader):
        opt = dict(ALL_OPTIONAL)
        opt["fvg_fresh"] = False
        result = grader.grade(ALL_MANDATORY, opt, net_rr=3.0)
        assert result.grade == "A"  # not A+ without fvg_fresh


class TestGradeA:
    def test_a_with_enough_score(self, grader):
        opt = dict(NO_OPTIONAL)
        # Need 18+ points: dxy(5) + ob(5) + strong_disp(6) + overlap(5) = 21
        opt["dxy_aligned"] = True
        opt["ob_valid"] = True
        opt["strong_displacement"] = True
        opt["overlap_session"] = True
        result = grader.grade(ALL_MANDATORY, opt, net_rr=2.5)
        assert result.grade == "A"


class TestGradeB:
    def test_b_with_low_score(self, grader):
        opt = dict(NO_OPTIONAL)
        opt["dxy_aligned"] = True  # 5 points
        result = grader.grade(ALL_MANDATORY, opt, net_rr=2.0)
        assert result.grade == "B"
        assert "paper only" in result.detail.lower()


class TestGradeC:
    def test_c_single_mandatory_fail(self, grader):
        m = dict(ALL_MANDATORY)
        m["news_clear"] = False
        result = grader.grade(m, NO_OPTIONAL, net_rr=2.0)
        assert result.grade == "C"
        assert result.mandatory_passed is False


class TestGradeD:
    def test_d_multiple_mandatory_fail(self, grader):
        m = dict(ALL_MANDATORY)
        m["htf_bias"] = False
        m["sweep"] = False
        result = grader.grade(m, NO_OPTIONAL, net_rr=2.0)
        assert result.grade == "D"
        assert len(result.failed_mandatory) == 2


class TestScoring:
    def test_max_score(self, grader):
        result = grader.grade(ALL_MANDATORY, ALL_OPTIONAL, net_rr=3.0)
        assert result.score == 50

    def test_zero_score(self, grader):
        result = grader.grade(ALL_MANDATORY, NO_OPTIONAL, net_rr=2.0)
        assert result.score == 0

    def test_partial_score(self, grader):
        opt = dict(NO_OPTIONAL)
        opt["fvg_fresh"] = True  # 6
        opt["strong_displacement"] = True  # 6
        result = grader.grade(ALL_MANDATORY, opt, net_rr=2.0)
        assert result.score == 12


class TestResultDict:
    def test_to_dict(self, grader):
        result = grader.grade(ALL_MANDATORY, ALL_OPTIONAL, net_rr=3.0)
        d = result.to_dict()
        assert d["grade"] == "A+"
        assert d["score"] == 50
        assert d["mandatory_passed"] is True


ALL_INDICATORS = {k: True for k in INDICATOR_SCORES}
NO_INDICATORS = {k: False for k in INDICATOR_SCORES}


class TestIndicatorBackwardCompat:
    def test_none_indicators_identical(self, grader):
        result = grader.grade(ALL_MANDATORY, ALL_OPTIONAL, net_rr=3.0)
        assert result.score == 50
        assert result.core_score == 50
        assert result.indicator_score == 0

    def test_empty_indicators_no_boost(self, grader):
        result = grader.grade(ALL_MANDATORY, ALL_OPTIONAL, net_rr=3.0, indicator_results=NO_INDICATORS)
        assert result.score == 50
        assert result.indicator_score == 0


class TestIndicatorBoost:
    def test_indicators_add_to_score(self, grader):
        result = grader.grade(ALL_MANDATORY, ALL_OPTIONAL, net_rr=3.0, indicator_results=ALL_INDICATORS)
        assert result.core_score == 50
        assert result.indicator_score == 20
        assert result.score == 70

    def test_indicators_promote_to_a_plus(self, grader):
        # core 24 (below 30) would be grade A; +8 indicators pushes to 32 >= 30 → A+
        opt = dict(NO_OPTIONAL)
        opt["strong_displacement"] = True  # 6
        opt["fvg_fresh"] = True  # 6  (required for A+)
        opt["dxy_aligned"] = True  # 5
        opt["ob_valid"] = True  # 5
        opt["overlap_session"] = True  # 5
        # core = 27
        ind = dict(NO_INDICATORS)
        ind["vwap_aligned"] = True  # 6 → total 33
        result = grader.grade(ALL_MANDATORY, opt, net_rr=3.0, indicator_results=ind)
        assert result.core_score == 27
        assert result.score == 33
        assert result.grade == "A+"

    def test_indicators_alone_cannot_make_a_plus_without_fvg_fresh(self, grader):
        opt = dict(NO_OPTIONAL)
        opt["dxy_aligned"] = True  # 5
        opt["ob_valid"] = True  # 5
        opt["strong_displacement"] = True  # 6
        opt["overlap_session"] = True  # 5
        # core = 21, fvg_fresh NOT set
        result = grader.grade(ALL_MANDATORY, opt, net_rr=3.0, indicator_results=ALL_INDICATORS)
        assert result.score == 41
        assert result.grade != "A+"  # no fvg_fresh

    def test_passed_indicators_tracked(self, grader):
        ind = dict(NO_INDICATORS)
        ind["vwap_aligned"] = True
        ind["ema_trend_aligned"] = True
        result = grader.grade(ALL_MANDATORY, ALL_OPTIONAL, net_rr=3.0, indicator_results=ind)
        assert "vwap_aligned" in result.passed_indicators
        assert "ema_trend_aligned" in result.passed_indicators
        assert result.indicator_score == 10
