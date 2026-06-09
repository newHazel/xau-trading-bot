"""Tests for Rulebook Engine — Phase 4.1."""

import pytest
from datetime import datetime
from core.engine.rulebook_engine import RulebookEngine
from core.engine.signal_grader import OPTIONAL_SCORES

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
def engine():
    return RulebookEngine(CONFIG)


class TestApproval:
    def test_all_pass_approved(self, engine):
        decision = engine.evaluate(
            direction="long",
            mandatory=ALL_MANDATORY,
            optional=ALL_OPTIONAL,
            net_rr=3.0,
            timestamp=datetime(2026, 3, 18, 12, 0),
        )
        assert decision.approved is True
        assert decision.grade.grade == "A+"
        assert decision.rejection is None

    def test_a_grade(self, engine):
        opt = dict(NO_OPTIONAL)
        opt["dxy_aligned"] = True
        opt["ob_valid"] = True
        opt["strong_displacement"] = True
        opt["overlap_session"] = True
        decision = engine.evaluate("long", ALL_MANDATORY, opt, net_rr=2.5)
        assert decision.approved is True
        assert decision.grade.grade == "A"


class TestRejection:
    def test_single_mandatory_fail(self, engine):
        m = dict(ALL_MANDATORY)
        m["news_clear"] = False
        decision = engine.evaluate(
            direction="long",
            mandatory=m,
            optional=NO_OPTIONAL,
            net_rr=2.0,
            timestamp=datetime(2026, 3, 18, 12, 0),
            setup_id="S1",
        )
        assert decision.approved is False
        assert decision.rejection is not None
        assert decision.rejection.main_reason == "news_clear"
        assert decision.grade.grade == "C"

    def test_multiple_mandatory_fail(self, engine):
        m = dict(ALL_MANDATORY)
        m["htf_bias"] = False
        m["sweep"] = False
        m["kill_zone"] = False
        decision = engine.evaluate("long", m, NO_OPTIONAL, net_rr=2.0)
        assert decision.approved is False
        assert decision.grade.grade == "D"
        assert len(decision.rejection.failed_conditions) == 3


class TestRejectionTracking:
    def test_rejections_accumulated(self, engine):
        m = dict(ALL_MANDATORY)
        m["htf_bias"] = False
        for _ in range(5):
            engine.evaluate("long", m, NO_OPTIONAL, net_rr=2.0, timestamp=datetime.now())
        assert engine.rejection_engine.count == 5


class TestContext:
    def test_context_passed_to_rejection(self, engine):
        m = dict(ALL_MANDATORY)
        m["fvg_valid"] = False
        ctx = {"fvg_mitigation": 0.65, "price": 2340.5}
        decision = engine.evaluate("long", m, NO_OPTIONAL, context=ctx)
        assert decision.rejection.context["fvg_mitigation"] == 0.65


class TestDecisionDict:
    def test_approved_to_dict(self, engine):
        decision = engine.evaluate("long", ALL_MANDATORY, ALL_OPTIONAL, net_rr=3.0)
        d = decision.to_dict()
        assert d["approved"] is True
        assert d["grade"]["grade"] == "A+"
        assert d["rejection"] is None

    def test_rejected_to_dict(self, engine):
        m = dict(ALL_MANDATORY)
        m["sweep"] = False
        decision = engine.evaluate("long", m, NO_OPTIONAL, net_rr=2.0)
        d = decision.to_dict()
        assert d["approved"] is False
        assert d["rejection"]["main_reason"] == "sweep"


class TestShortDirection:
    def test_short_approved(self, engine):
        decision = engine.evaluate("short", ALL_MANDATORY, ALL_OPTIONAL, net_rr=3.0)
        assert decision.approved is True
        assert decision.direction == "short"


class TestIndicatorPassthrough:
    def test_indicators_flow_to_grade(self, engine):
        ind = {"vwap_aligned": True, "ema_trend_aligned": True,
               "rsi_divergence_confirms": True, "volume_profile_favorable": True}
        decision = engine.evaluate("long", ALL_MANDATORY, ALL_OPTIONAL, net_rr=3.0, indicators=ind)
        assert decision.grade.indicator_score == 20
        assert decision.grade.score == 70
        assert decision.indicator_results == ind

    def test_no_indicators_none(self, engine):
        decision = engine.evaluate("long", ALL_MANDATORY, ALL_OPTIONAL, net_rr=3.0)
        assert decision.indicator_results is None
        assert decision.grade.indicator_score == 0

    def test_indicators_in_dict(self, engine):
        ind = {"vwap_aligned": True, "ema_trend_aligned": False,
               "rsi_divergence_confirms": False, "volume_profile_favorable": False}
        decision = engine.evaluate("long", ALL_MANDATORY, ALL_OPTIONAL, net_rr=3.0, indicators=ind)
        d = decision.to_dict()
        assert d["indicator_results"]["vwap_aligned"] is True
