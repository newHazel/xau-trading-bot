"""
Sequence runner — drives the State Machine through the SMC setup sequence.

Instead of requiring all 15 mandatory conditions to be true on one bar (which is
mathematically near-impossible), this walks the StateMachine forward one step at
a time as each condition becomes true, and REMEMBERS the steps already passed:

    HTF bias → 15m aligned → price in zone → liquidity sweep → valid FVG
    → retrace to zone → micro-CHoCH → confirmation candle → SIGNAL_READY

When SIGNAL_READY is reached, the "at-entry" gates (kill_zone, news, R:R,
no-blocking-filters, daily limits) are checked NOW, the setup is graded, and an
A/A+/B signal is emitted. The setup's sweep/FVG levels are captured when their
states are passed and reused for entry/SL/TP — so the trade reflects the real
setup, not whatever the latest bar happens to show.

Invalidation: if the HTF bias is lost mid-sequence, or the setup takes too long
(expiry), the machine resets to WAITING_FOR_HTF_BIAS. After a signal it cools
down for a few bars before hunting again.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.engine.rulebook_engine import RulebookEngine
from core.engine.signal_pipeline import PipelineContext, PipelineSignal, SignalPipeline
from core.engine.pipeline_hooks import build_default_hooks
from core.engine.state_machine import StateMachine, State

# (current state, advance-when, next state). Each gate is checked only in its state.
_SEQUENCE: List[Tuple[State, str, State]] = [
    (State.WAITING_FOR_HTF_BIAS,          "htf_bias",            State.WAITING_FOR_15M_ALIGNMENT),
    (State.WAITING_FOR_15M_ALIGNMENT,     "15m_aligned",         State.WAITING_FOR_PRICE_IN_ZONE),
    (State.WAITING_FOR_PRICE_IN_ZONE,     "price_zone",          State.WAITING_FOR_LIQUIDITY_SWEEP),
    (State.WAITING_FOR_LIQUIDITY_SWEEP,   "sweep",               State.WAITING_FOR_VALID_FVG_OR_OB),
    (State.WAITING_FOR_VALID_FVG_OR_OB,   "fvg",                 State.WAITING_FOR_RETRACE_TO_ZONE),
    (State.WAITING_FOR_RETRACE_TO_ZONE,   "retrace",             State.WAITING_FOR_MICRO_CHOCH),
    (State.WAITING_FOR_MICRO_CHOCH,       "micro_choch",         State.WAITING_FOR_CONFIRMATION_CANDLE),
    (State.WAITING_FOR_CONFIRMATION_CANDLE, "confirmation",      State.SIGNAL_READY),
]
_SEQUENCE_MAP = {s: (cond, nxt) for s, cond, nxt in _SEQUENCE}


def _advance_condition(ctx: PipelineContext, key: str) -> bool:
    if key == "htf_bias":
        return ctx.htf_bias in ("long", "short")
    if key == "15m_aligned":
        return ctx.structure_15m == ctx.direction
    if key == "price_zone":
        return ctx.price_zone in ("premium", "discount")
    if key == "sweep":
        return ctx.sweep is not None and ctx.sweep_confirmed
    if key == "fvg":
        return ctx.fvg_valid and ctx.fvg_fresh
    if key == "retrace":
        return ctx.retraced_to_zone
    if key == "micro_choch":
        return ctx.micro_choch
    if key == "confirmation":
        return ctx.confirmation_candle
    return False


class SequenceRunner:
    def __init__(
        self,
        config: Dict[str, Any],
        execution_tf: str = "15m",
        account_balance: float = 10000.0,
        setup_expiry_bars: int = 40,
        cooldown_bars: int = 8,
        tradeable_grades: Tuple[str, ...] = ("A+", "A", "B"),
        hooks: Optional[Dict[str, Any]] = None,
        setup_id_fn=None,
    ) -> None:
        self._config = config
        self._exec_tf = execution_tf
        self._rulebook = RulebookEngine(config)
        self._hooks = hooks or build_default_hooks(config, account_balance, execution_tf)
        self._sm = StateMachine(State.WAITING_FOR_HTF_BIAS)
        self._expiry = setup_expiry_bars
        self._cooldown_bars = cooldown_bars
        self._tradeable = set(tradeable_grades)
        self._setup_id_fn = setup_id_fn or self._default_setup_id

        self._bars_in_setup = 0
        self._cooldown_left = 0
        self._locked_direction: Optional[str] = None
        self._captured: Dict[str, Any] = {}
        self._counter = 0

        # Periodic re-pin of the captured FVG: prefer fresh+near zones, never stay
        # stuck on a stale/far gap waiting for an impossible retrace (the 2026-06-10
        # ~4180 miss). Same config switch as the selection scorer.
        self._repin_enabled = bool(config.get("fvg_freshness_enabled", True))
        self._repin_interval = int(config.get("fvg_repin_interval_bars", 5))
        self._repin_min_gain = float(config.get("fvg_repin_min_improvement_points", 8.0))
        self._bars_since_repin = 0

        # #6: require a real rejection AT the pinned zone for the confirmation step
        # (price tags the zone and closes back out), not just a same-colour candle.
        self._require_zone_rejection = bool(config.get("require_zone_rejection", True))

    @property
    def state(self) -> State:
        return self._sm.state

    # ------------------------------------------------------------------ #

    def on_bar(self, bar: Any, history: Any) -> Optional[PipelineSignal]:
        ctx = self._populate(bar, history)
        now = ctx.timestamp

        # Once the FVG is captured, PIN it: retrace AND entry must reference that
        # exact zone. Otherwise a newer FVG re-selected each bar can spuriously
        # satisfy "retrace" while the entry still uses the (possibly stale) captured
        # FVG — producing an entry far from price (the 08:40 bug). Re-evaluate
        # retrace as "did THIS bar's price come back to the captured FVG zone".
        fresh_fvg = ctx.fvg  # best zone smc_hook just selected this bar (may be None)
        cap_fvg = self._captured.get("fvg")
        if cap_fvg is not None:
            # Before re-evaluating retrace, optionally RE-PIN to a clearly nearer
            # same-direction zone (every repin_interval bars, before retrace fires).
            # Both retrace AND entry read self._captured["fvg"], so they stay
            # consistent — no 08:40-style entry-vs-retrace mismatch is reintroduced.
            cap_fvg = self._maybe_repin(cap_fvg, fresh_fvg, history)
            ctx.fvg = cap_fvg
            df = history.get(self._exec_tf) if isinstance(history, dict) else None
            if df is not None and getattr(df, "empty", True) is False and len(df) >= 1:
                lo = min(cap_fvg["top"], cap_fvg["bottom"])
                hi = max(cap_fvg["top"], cap_fvg["bottom"])
                rl = float(df["low"].iloc[-3:].min())
                rh = float(df["high"].iloc[-3:].max())
                ctx.retraced_to_zone = (rl <= hi and rh >= lo)
                # #6: confirmation = a REAL rejection AT the pinned zone — price tagged
                # the zone and CLOSED BACK OUT of it in-direction — not just any same-
                # colour candle (smc_hook's c<o/c>o, which fired the 4154 short mid-move
                # while price was still BELOW the zone). Evaluated against the PINNED zone
                # so entry stays consistent. require_zone_rejection=false → legacy.
                if self._require_zone_rejection:
                    o = float(df["open"].iloc[-1]); c = float(df["close"].iloc[-1])
                    h = float(df["high"].iloc[-1]); l = float(df["low"].iloc[-1])
                    if self._locked_direction == "short":
                        ctx.confirmation_candle = (h >= lo) and (c < lo) and (c < o)
                    elif self._locked_direction == "long":
                        ctx.confirmation_candle = (l <= hi) and (c > hi) and (c > o)

        # cooldown after a signal
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            if self._cooldown_left == 0:
                self._reset(now, "cooldown complete")
            return None

        state = self._sm.state

        # mid-sequence guards: expiry + bias invalidation
        if state != State.WAITING_FOR_HTF_BIAS:
            self._bars_in_setup += 1
            if self._bars_in_setup > self._expiry:
                self._reset(now, "setup expired")
                return None
            if ctx.htf_bias not in ("long", "short") or ctx.htf_bias != self._locked_direction:
                self._reset(now, "htf bias lost/flipped")
                return None

        # try to advance as far as possible this bar (a fast bar can clear several steps)
        for _ in range(len(_SEQUENCE)):
            state = self._sm.state
            if state not in _SEQUENCE_MAP:
                break
            cond_key, nxt = _SEQUENCE_MAP[state]
            if not _advance_condition(ctx, cond_key):
                break
            self._sm.transition(nxt, f"{cond_key} met", now)
            if state == State.WAITING_FOR_HTF_BIAS:
                self._locked_direction = ctx.direction
                self._bars_in_setup = 0
            if cond_key == "sweep":
                self._captured["sweep"] = ctx.sweep
            if cond_key == "fvg":
                self._captured["fvg"] = ctx.fvg

        if self._sm.state == State.SIGNAL_READY:
            return self._emit(ctx, bar, history, now)
        return None

    # ------------------------------------------------------------------ #

    def _maybe_repin(self, cap_fvg: Dict[str, Any], fresh_fvg: Optional[Dict[str, Any]],
                     history: Any) -> Dict[str, Any]:
        """Swap the pinned FVG for a clearly nearer same-direction one, at most once
        every `repin_interval` bars. Keeps the engine from waiting out a stale/far gap
        on a one-directional move. The freshest+nearest candidate is whatever smc_hook
        already selected this bar.

        CRITICAL: only re-pin while the FSM is still WAITING_FOR_RETRACE_TO_ZONE. The
        retrace gate is validated exactly once (the FSM is forward-only) and entry/
        SL/TP read the captured zone — so swapping AFTER retrace has passed would make
        the trade execute on a zone price never retraced into. That is the cross-bar
        form of the 08:40 entry-vs-retrace bug; the state guard below prevents it."""
        if not self._repin_enabled or fresh_fvg is None or fresh_fvg is cap_fvg:
            return cap_fvg
        if self._sm.state != State.WAITING_FOR_RETRACE_TO_ZONE:
            return cap_fvg
        self._bars_since_repin += 1
        if self._bars_since_repin < self._repin_interval:
            return cap_fvg
        df = history.get(self._exec_tf) if isinstance(history, dict) else None
        if df is None or getattr(df, "empty", True) is True or len(df) < 1:
            return cap_fvg
        self._bars_since_repin = 0  # consume the interval only when we truly evaluate
        price = float(df["close"].iloc[-1])

        def _near(f: Dict[str, Any]) -> float:
            lo = min(f["top"], f["bottom"]); hi = max(f["top"], f["bottom"])
            if price < lo:
                return lo - price
            if price > hi:
                return price - hi
            return 0.0

        d_cap = _near(cap_fvg)
        if d_cap == 0.0:
            return cap_fvg  # price already at/in the pinned zone — keep it
        if _near(fresh_fvg) + self._repin_min_gain < d_cap:
            self._captured["fvg"] = fresh_fvg
            return fresh_fvg
        return cap_fvg

    def _populate(self, bar: Any, history: Any) -> PipelineContext:
        ts = getattr(bar, "timestamp", None) or (bar.get("timestamp") if isinstance(bar, dict) else None)
        bidx = getattr(bar, "bar_index", None) or (bar.get("bar_index", 0) if isinstance(bar, dict) else 0)
        sym = getattr(bar, "symbol", None) or (bar.get("symbol", "XAUUSD") if isinstance(bar, dict) else "XAUUSD")
        ctx = PipelineContext(timestamp=ts, bar_index=bidx, symbol=sym)
        if self._locked_direction:
            ctx.direction = self._locked_direction
        # analysis stages (risk runs only at emit, with captured levels)
        for name in ("structure_hook", "smc_hook", "filter_hook", "indicator_hook"):
            self._hooks[name](ctx, bar, history)
        return ctx

    def _emit(self, ctx: PipelineContext, bar: Any, history: Any, now: datetime) -> Optional[PipelineSignal]:
        # reuse the captured setup levels for the risk calc
        if "sweep" in self._captured:
            ctx.sweep = self._captured["sweep"]
        if "fvg" in self._captured:
            ctx.fvg = self._captured["fvg"]
        ctx.direction = self._locked_direction or ctx.direction
        self._hooks["risk_hook"](ctx, bar, history)

        if ctx.entry is None or ctx.sl is None or ctx.tp1 is None:
            self._reset(now, "risk could not size the setup")
            return None

        # sequence conditions are TRUE (we tracked them); gates checked now.
        mandatory = {
            "htf_bias": True, "15m_aligned": True, "price_zone": True,
            "sweep": True, "sweep_confirmation": True,
            "fvg_valid": True, "fvg_freshness": True,
            "retrace_to_zone": True, "micro_choch": True, "confirmation_candle": True,
            "kill_zone": ctx.in_kill_zone,
            "news_clear": ctx.news_clear,
            "rr_minimum": ctx.rr_minimum_ok,
            "daily_limits_ok": ctx.daily_limits_ok,
            "no_blocking_filters": ctx.no_blocking_filters,
        }
        optional = SignalPipeline._build_optional(ctx)
        indicators = SignalPipeline._build_indicators(ctx)

        setup_id = self._setup_id_fn(ctx)
        decision = self._rulebook.evaluate(
            direction=ctx.direction, mandatory=mandatory, optional=optional,
            net_rr=ctx.net_rr, symbol=ctx.symbol, timestamp=now,
            setup_id=setup_id, indicators=indicators,
        )

        # enter cooldown regardless (one shot per completed sequence)
        self._sm.transition(State.SIGNAL_SENT, "signal evaluated", now)
        self._sm.transition(State.COOLDOWN, "post-signal cooldown", now)
        self._cooldown_left = self._cooldown_bars

        grade = decision.grade.grade if decision.grade else "D"
        if not decision.approved or grade not in self._tradeable:
            return None

        return PipelineSignal(
            setup_id=setup_id, direction=ctx.direction,
            entry=ctx.entry, sl=ctx.sl, tp1=ctx.tp1,
            tp2=ctx.tp2 if ctx.tp2 is not None else ctx.tp1,
            lot_size=ctx.lot_size, grade=grade,
            score=decision.grade.score if decision.grade else 0,
            timestamp=now, bar_index=ctx.bar_index,
            approved=decision.approved, decision=decision,
        )

    def _reset(self, now: datetime, reason: str) -> None:
        self._sm.force_state(State.WAITING_FOR_HTF_BIAS, reason, now)
        self._bars_in_setup = 0
        self._locked_direction = None
        self._captured = {}
        self._bars_since_repin = 0

    def _default_setup_id(self, ctx: PipelineContext) -> str:
        self._counter += 1
        ts = ctx.timestamp.strftime("%Y%m%d-%H%M") if hasattr(ctx.timestamp, "strftime") else "na"
        return f"{ctx.symbol}-{ts}-{ctx.direction.upper()}"
