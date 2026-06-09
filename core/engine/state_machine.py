"""
State Machine — Phase 4.4.

Tracks the system through 16 possible states. Every transition is
logged with reason and context. The machine enforces valid transitions
only — invalid jumps are rejected.

States:
  1. WAITING_FOR_MARKET_OPEN
  2. WAITING_FOR_HTF_BIAS
  3. WAITING_FOR_15M_ALIGNMENT
  4. WAITING_FOR_PRICE_IN_ZONE
  5. WAITING_FOR_LIQUIDITY_SWEEP
  6. WAITING_FOR_VALID_FVG_OR_OB
  7. WAITING_FOR_RETRACE_TO_ZONE
  8. WAITING_FOR_MICRO_CHOCH
  9. WAITING_FOR_CONFIRMATION_CANDLE
  10. SIGNAL_READY
  11. SIGNAL_SENT
  12. TRADE_MONITORING
  13. COOLDOWN
  14. DAY_LOCKED
  15. SYSTEM_ERROR
  16. DEGRADED_MODE
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.indicators.execution_switcher import ExecutionSwitcher, ExecutionTF, ExecutionDecision


class State(str, Enum):
    WAITING_FOR_MARKET_OPEN = "waiting_for_market_open"
    WAITING_FOR_HTF_BIAS = "waiting_for_htf_bias"
    WAITING_FOR_15M_ALIGNMENT = "waiting_for_15m_alignment"
    WAITING_FOR_PRICE_IN_ZONE = "waiting_for_price_in_zone"
    WAITING_FOR_LIQUIDITY_SWEEP = "waiting_for_liquidity_sweep"
    WAITING_FOR_VALID_FVG_OR_OB = "waiting_for_valid_fvg_or_ob"
    WAITING_FOR_RETRACE_TO_ZONE = "waiting_for_retrace_to_zone"
    WAITING_FOR_MICRO_CHOCH = "waiting_for_micro_choch"
    WAITING_FOR_CONFIRMATION_CANDLE = "waiting_for_confirmation_candle"
    SIGNAL_READY = "signal_ready"
    SIGNAL_SENT = "signal_sent"
    TRADE_MONITORING = "trade_monitoring"
    COOLDOWN = "cooldown"
    DAY_LOCKED = "day_locked"
    SYSTEM_ERROR = "system_error"
    DEGRADED_MODE = "degraded_mode"


VALID_TRANSITIONS: Dict[State, List[State]] = {
    State.WAITING_FOR_MARKET_OPEN: [
        State.WAITING_FOR_HTF_BIAS,
        State.DAY_LOCKED,
        State.SYSTEM_ERROR,
    ],
    State.WAITING_FOR_HTF_BIAS: [
        State.WAITING_FOR_15M_ALIGNMENT,
        State.WAITING_FOR_MARKET_OPEN,
        State.DAY_LOCKED,
        State.SYSTEM_ERROR,
        State.DEGRADED_MODE,
    ],
    State.WAITING_FOR_15M_ALIGNMENT: [
        State.WAITING_FOR_PRICE_IN_ZONE,
        State.WAITING_FOR_HTF_BIAS,
        State.DAY_LOCKED,
        State.SYSTEM_ERROR,
        State.DEGRADED_MODE,
    ],
    State.WAITING_FOR_PRICE_IN_ZONE: [
        State.WAITING_FOR_LIQUIDITY_SWEEP,
        State.WAITING_FOR_HTF_BIAS,
        State.DAY_LOCKED,
        State.SYSTEM_ERROR,
    ],
    State.WAITING_FOR_LIQUIDITY_SWEEP: [
        State.WAITING_FOR_VALID_FVG_OR_OB,
        State.WAITING_FOR_HTF_BIAS,
        State.DAY_LOCKED,
        State.SYSTEM_ERROR,
    ],
    State.WAITING_FOR_VALID_FVG_OR_OB: [
        State.WAITING_FOR_RETRACE_TO_ZONE,
        State.WAITING_FOR_LIQUIDITY_SWEEP,
        State.WAITING_FOR_HTF_BIAS,
        State.DAY_LOCKED,
        State.SYSTEM_ERROR,
    ],
    State.WAITING_FOR_RETRACE_TO_ZONE: [
        State.WAITING_FOR_MICRO_CHOCH,
        State.WAITING_FOR_LIQUIDITY_SWEEP,
        State.WAITING_FOR_HTF_BIAS,
        State.DAY_LOCKED,
        State.SYSTEM_ERROR,
    ],
    State.WAITING_FOR_MICRO_CHOCH: [
        State.WAITING_FOR_CONFIRMATION_CANDLE,
        State.WAITING_FOR_LIQUIDITY_SWEEP,
        State.WAITING_FOR_HTF_BIAS,
        State.DAY_LOCKED,
        State.SYSTEM_ERROR,
    ],
    State.WAITING_FOR_CONFIRMATION_CANDLE: [
        State.SIGNAL_READY,
        State.WAITING_FOR_LIQUIDITY_SWEEP,
        State.WAITING_FOR_HTF_BIAS,
        State.DAY_LOCKED,
        State.SYSTEM_ERROR,
    ],
    State.SIGNAL_READY: [
        State.SIGNAL_SENT,
        State.COOLDOWN,
        State.WAITING_FOR_HTF_BIAS,
        State.DAY_LOCKED,
        State.SYSTEM_ERROR,
    ],
    State.SIGNAL_SENT: [
        State.TRADE_MONITORING,
        State.COOLDOWN,
        State.DAY_LOCKED,
        State.SYSTEM_ERROR,
    ],
    State.TRADE_MONITORING: [
        State.COOLDOWN,
        State.DAY_LOCKED,
        State.SYSTEM_ERROR,
    ],
    State.COOLDOWN: [
        State.WAITING_FOR_HTF_BIAS,
        State.DAY_LOCKED,
        State.WAITING_FOR_MARKET_OPEN,
        State.SYSTEM_ERROR,
    ],
    State.DAY_LOCKED: [
        State.WAITING_FOR_MARKET_OPEN,
        State.SYSTEM_ERROR,
    ],
    State.SYSTEM_ERROR: [
        State.WAITING_FOR_MARKET_OPEN,
        State.WAITING_FOR_HTF_BIAS,
        State.DEGRADED_MODE,
    ],
    State.DEGRADED_MODE: [
        State.WAITING_FOR_HTF_BIAS,
        State.WAITING_FOR_MARKET_OPEN,
        State.SYSTEM_ERROR,
    ],
}


# Phase 11 — LTF "trigger" states where the 5m/1m execution choice applies.
# HTF/MTF context states (HTF_BIAS, 15M_ALIGNMENT, PRICE_IN_ZONE) are unaffected.
TRIGGER_PHASE_STATES = frozenset({
    State.WAITING_FOR_LIQUIDITY_SWEEP,
    State.WAITING_FOR_VALID_FVG_OR_OB,
    State.WAITING_FOR_RETRACE_TO_ZONE,
    State.WAITING_FOR_MICRO_CHOCH,
    State.WAITING_FOR_CONFIRMATION_CANDLE,
})


@dataclass(frozen=True)
class StateTransition:
    from_state: State
    to_state: State
    reason: str
    timestamp: datetime
    setup_id: Optional[str]
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "setup_id": self.setup_id,
            "context": self.context,
        }


class StateMachine:
    """Manages system state with validated transitions and full logging."""

    def __init__(
        self,
        initial_state: State = State.WAITING_FOR_MARKET_OPEN,
        execution_switcher: Optional[ExecutionSwitcher] = None,
    ) -> None:
        self._state = initial_state
        self._history: List[StateTransition] = []
        self._current_setup_id: Optional[str] = None
        self._execution_switcher = execution_switcher
        self._execution_decision: Optional[ExecutionDecision] = None

    @property
    def state(self) -> State:
        return self._state

    @property
    def current_setup_id(self) -> Optional[str]:
        return self._current_setup_id

    @property
    def history(self) -> List[StateTransition]:
        return list(self._history)

    # ---------------------------------------------------------------- #
    # Phase 11 — execution timeframe (5m / 1m) selection                 #
    # ---------------------------------------------------------------- #

    @property
    def active_execution_tf(self) -> Optional[ExecutionTF]:
        """Currently selected trigger timeframe.

        None if no switcher is configured. Defaults to 5m before the first
        refresh when a switcher is present.
        """
        if self._execution_switcher is None:
            return None
        if self._execution_decision is None:
            return ExecutionTF.M5
        return self._execution_decision.chosen_tf

    @property
    def execution_decision(self) -> Optional[ExecutionDecision]:
        return self._execution_decision

    def is_in_trigger_phase(self) -> bool:
        """True when the machine is in an LTF trigger state where 5m/1m matters."""
        return self._state in TRIGGER_PHASE_STATES

    def refresh_execution_tf(
        self,
        now: datetime,
        volatility_regime: str = "normal",
    ) -> Optional[ExecutionDecision]:
        """Recompute the active execution TF from the switcher.

        No-op (returns None) if no switcher was configured. When the chosen TF
        changes, the change is appended to history as a non-state transition note.
        """
        if self._execution_switcher is None:
            return None

        prev_tf = self.active_execution_tf
        decision = self._execution_switcher.decide(now, volatility_regime)
        self._execution_decision = decision

        if prev_tf is not None and prev_tf != decision.chosen_tf:
            self._history.append(StateTransition(
                from_state=self._state,
                to_state=self._state,
                reason=f"execution_tf {prev_tf.value}→{decision.chosen_tf.value}: {decision.reason}",
                timestamp=now,
                setup_id=self._current_setup_id,
                context={
                    "execution_tf_change": True,
                    "from_tf": prev_tf.value,
                    "to_tf": decision.chosen_tf.value,
                    "in_overlap": decision.in_overlap,
                    "volatility_regime": volatility_regime,
                },
            ))
        return decision

    def transition(
        self,
        to_state: State,
        reason: str,
        timestamp: datetime,
        setup_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        valid_targets = VALID_TRANSITIONS.get(self._state, [])
        if to_state not in valid_targets:
            return False

        transition = StateTransition(
            from_state=self._state,
            to_state=to_state,
            reason=reason,
            timestamp=timestamp,
            setup_id=setup_id or self._current_setup_id,
            context=context or {},
        )
        self._history.append(transition)
        self._state = to_state

        if setup_id is not None:
            self._current_setup_id = setup_id

        if to_state in (State.WAITING_FOR_MARKET_OPEN, State.DAY_LOCKED):
            self._current_setup_id = None

        return True

    def can_transition(self, to_state: State) -> bool:
        return to_state in VALID_TRANSITIONS.get(self._state, [])

    def force_state(
        self,
        to_state: State,
        reason: str,
        timestamp: datetime,
    ) -> None:
        transition = StateTransition(
            from_state=self._state,
            to_state=to_state,
            reason=f"FORCED: {reason}",
            timestamp=timestamp,
            setup_id=self._current_setup_id,
            context={"forced": True},
        )
        self._history.append(transition)
        self._state = to_state

    def get_recent_transitions(self, n: int = 10) -> List[StateTransition]:
        return self._history[-n:]

    def reset(self, initial_state: State = State.WAITING_FOR_MARKET_OPEN) -> None:
        self._state = initial_state
        self._history.clear()
        self._current_setup_id = None
        self._execution_decision = None
