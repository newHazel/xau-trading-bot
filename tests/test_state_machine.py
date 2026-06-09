"""Tests for State Machine — Phase 4.4 + Phase 11 execution-TF integration."""

import pytest
from datetime import datetime, timezone
from core.engine.state_machine import StateMachine, State, TRIGGER_PHASE_STATES
from core.indicators.execution_switcher import ExecutionSwitcher, ExecutionTF


@pytest.fixture
def sm():
    return StateMachine()


class TestInitialState:
    def test_starts_at_market_open(self, sm):
        assert sm.state == State.WAITING_FOR_MARKET_OPEN

    def test_custom_initial(self):
        sm = StateMachine(State.WAITING_FOR_HTF_BIAS)
        assert sm.state == State.WAITING_FOR_HTF_BIAS


class TestValidTransitions:
    def test_market_open_to_htf_bias(self, sm):
        ok = sm.transition(State.WAITING_FOR_HTF_BIAS, "market opened", datetime.now())
        assert ok is True
        assert sm.state == State.WAITING_FOR_HTF_BIAS

    def test_full_happy_path(self, sm):
        now = datetime(2026, 3, 18, 12, 0)
        steps = [
            (State.WAITING_FOR_HTF_BIAS, "market opened"),
            (State.WAITING_FOR_15M_ALIGNMENT, "htf bias bullish"),
            (State.WAITING_FOR_PRICE_IN_ZONE, "15m aligned"),
            (State.WAITING_FOR_LIQUIDITY_SWEEP, "price in discount"),
            (State.WAITING_FOR_VALID_FVG_OR_OB, "sweep detected"),
            (State.WAITING_FOR_RETRACE_TO_ZONE, "fvg valid"),
            (State.WAITING_FOR_MICRO_CHOCH, "price retraced"),
            (State.WAITING_FOR_CONFIRMATION_CANDLE, "micro choch confirmed"),
            (State.SIGNAL_READY, "confirmation candle"),
            (State.SIGNAL_SENT, "signal created"),
            (State.TRADE_MONITORING, "alert sent"),
            (State.COOLDOWN, "trade closed"),
            (State.WAITING_FOR_HTF_BIAS, "cooldown ended"),
        ]
        for to_state, reason in steps:
            ok = sm.transition(to_state, reason, now)
            assert ok is True, f"Failed: {sm.state} → {to_state}"
        assert sm.state == State.WAITING_FOR_HTF_BIAS
        assert len(sm.history) == len(steps)


class TestInvalidTransitions:
    def test_cannot_skip_states(self, sm):
        ok = sm.transition(State.SIGNAL_READY, "shortcut", datetime.now())
        assert ok is False
        assert sm.state == State.WAITING_FOR_MARKET_OPEN

    def test_cooldown_cannot_go_to_signal(self):
        sm = StateMachine(State.COOLDOWN)
        ok = sm.transition(State.SIGNAL_READY, "cheat", datetime.now())
        assert ok is False

    def test_day_locked_only_to_market_open(self):
        sm = StateMachine(State.DAY_LOCKED)
        ok = sm.transition(State.WAITING_FOR_HTF_BIAS, "try", datetime.now())
        assert ok is False
        ok = sm.transition(State.WAITING_FOR_MARKET_OPEN, "next day", datetime.now())
        assert ok is True


class TestCanTransition:
    def test_can_transition_true(self, sm):
        assert sm.can_transition(State.WAITING_FOR_HTF_BIAS) is True

    def test_can_transition_false(self, sm):
        assert sm.can_transition(State.SIGNAL_READY) is False


class TestSetupID:
    def test_setup_id_tracking(self, sm):
        now = datetime.now()
        sm.transition(State.WAITING_FOR_HTF_BIAS, "open", now, setup_id="S1")
        assert sm.current_setup_id == "S1"

    def test_setup_id_cleared_on_day_lock(self, sm):
        now = datetime.now()
        sm.transition(State.WAITING_FOR_HTF_BIAS, "open", now, setup_id="S1")
        sm.transition(State.DAY_LOCKED, "max losses", now)
        assert sm.current_setup_id is None


class TestForceState:
    def test_force_bypasses_validation(self, sm):
        sm.force_state(State.SIGNAL_READY, "testing", datetime.now())
        assert sm.state == State.SIGNAL_READY
        assert sm.history[-1].reason.startswith("FORCED")


class TestHistory:
    def test_history_records(self, sm):
        now = datetime.now()
        sm.transition(State.WAITING_FOR_HTF_BIAS, "open", now)
        sm.transition(State.WAITING_FOR_15M_ALIGNMENT, "bias set", now)
        assert len(sm.history) == 2
        assert sm.history[0].from_state == State.WAITING_FOR_MARKET_OPEN
        assert sm.history[0].to_state == State.WAITING_FOR_HTF_BIAS

    def test_get_recent(self, sm):
        now = datetime.now()
        for i in range(5):
            sm.force_state(State.WAITING_FOR_HTF_BIAS, f"r{i}", now)
            sm.force_state(State.WAITING_FOR_MARKET_OPEN, f"r{i}b", now)
        recent = sm.get_recent_transitions(3)
        assert len(recent) == 3

    def test_transition_to_dict(self, sm):
        now = datetime(2026, 3, 18, 12, 0)
        sm.transition(State.WAITING_FOR_HTF_BIAS, "open", now)
        d = sm.history[0].to_dict()
        assert d["from_state"] == "waiting_for_market_open"
        assert d["to_state"] == "waiting_for_htf_bias"


class TestErrorAndDegraded:
    def test_any_state_can_go_to_error(self, sm):
        sm.transition(State.WAITING_FOR_HTF_BIAS, "open", datetime.now())
        ok = sm.transition(State.SYSTEM_ERROR, "crash", datetime.now())
        assert ok is True

    def test_error_to_degraded(self):
        sm = StateMachine(State.SYSTEM_ERROR)
        ok = sm.transition(State.DEGRADED_MODE, "partial recovery", datetime.now())
        assert ok is True

    def test_degraded_to_normal(self):
        sm = StateMachine(State.DEGRADED_MODE)
        ok = sm.transition(State.WAITING_FOR_HTF_BIAS, "recovered", datetime.now())
        assert ok is True


class TestReset:
    def test_reset(self, sm):
        sm.transition(State.WAITING_FOR_HTF_BIAS, "open", datetime.now())
        sm.reset()
        assert sm.state == State.WAITING_FOR_MARKET_OPEN
        assert len(sm.history) == 0
        assert sm.current_setup_id is None


# Israel time 16:30 = UTC 14:30 (winter); 12:00 IL = UTC 10:00
OVERLAP_UTC = datetime(2026, 1, 21, 14, 30, tzinfo=timezone.utc)
OUTSIDE_UTC = datetime(2026, 1, 21, 10, 0, tzinfo=timezone.utc)


class TestExecutionTFBackwardCompat:
    def test_no_switcher_returns_none(self, sm):
        assert sm.active_execution_tf is None
        assert sm.execution_decision is None

    def test_refresh_without_switcher_noop(self, sm):
        assert sm.refresh_execution_tf(OVERLAP_UTC, "high") is None
        assert sm.active_execution_tf is None


class TestExecutionTFIntegration:
    @pytest.fixture
    def sm_sw(self):
        return StateMachine(execution_switcher=ExecutionSwitcher())

    def test_defaults_to_5m_before_refresh(self, sm_sw):
        assert sm_sw.active_execution_tf == ExecutionTF.M5

    def test_picks_1m_in_overlap_high_vol(self, sm_sw):
        sm_sw.refresh_execution_tf(OVERLAP_UTC, "high")
        assert sm_sw.active_execution_tf == ExecutionTF.M1

    def test_picks_5m_outside_overlap(self, sm_sw):
        sm_sw.refresh_execution_tf(OUTSIDE_UTC, "high")
        assert sm_sw.active_execution_tf == ExecutionTF.M5

    def test_picks_5m_in_overlap_normal_vol(self, sm_sw):
        sm_sw.refresh_execution_tf(OVERLAP_UTC, "normal")
        assert sm_sw.active_execution_tf == ExecutionTF.M5

    def test_tf_change_logged_in_history(self, sm_sw):
        sm_sw.refresh_execution_tf(OUTSIDE_UTC, "high")  # 5m
        sm_sw.refresh_execution_tf(OVERLAP_UTC, "high")  # → 1m
        changes = [t for t in sm_sw.history if t.context.get("execution_tf_change")]
        assert len(changes) == 1
        assert changes[0].context["to_tf"] == "1m"

    def test_no_change_no_log(self, sm_sw):
        sm_sw.refresh_execution_tf(OVERLAP_UTC, "high")  # default 5m → 1m: 1 log
        sm_sw.refresh_execution_tf(OVERLAP_UTC, "high")  # 1m → 1m: no new log
        changes = [t for t in sm_sw.history if t.context.get("execution_tf_change")]
        assert len(changes) == 1  # only the initial 5m→1m switch

    def test_decision_exposed(self, sm_sw):
        sm_sw.refresh_execution_tf(OVERLAP_UTC, "extreme")
        d = sm_sw.execution_decision
        assert d is not None
        assert d.in_overlap
        assert d.chosen_tf == ExecutionTF.M1


class TestTriggerPhase:
    def test_trigger_phase_states(self):
        sm = StateMachine(State.WAITING_FOR_LIQUIDITY_SWEEP)
        assert sm.is_in_trigger_phase()

    def test_confirmation_candle_is_trigger(self):
        sm = StateMachine(State.WAITING_FOR_CONFIRMATION_CANDLE)
        assert sm.is_in_trigger_phase()

    def test_htf_bias_not_trigger(self):
        sm = StateMachine(State.WAITING_FOR_HTF_BIAS)
        assert not sm.is_in_trigger_phase()

    def test_market_open_not_trigger(self, sm):
        assert not sm.is_in_trigger_phase()

    def test_five_trigger_states(self):
        assert len(TRIGGER_PHASE_STATES) == 5
        assert State.WAITING_FOR_MICRO_CHOCH in TRIGGER_PHASE_STATES


class TestExecutionTFReset:
    def test_reset_clears_decision(self):
        sm = StateMachine(execution_switcher=ExecutionSwitcher())
        sm.refresh_execution_tf(OVERLAP_UTC, "high")
        sm.reset()
        assert sm.execution_decision is None
        assert sm.active_execution_tf == ExecutionTF.M5  # back to default
