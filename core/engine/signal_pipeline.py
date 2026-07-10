"""
Signal Pipeline — SKELETON (wiring scaffold).

Turns raw candles into graded trade signals by chaining the analysis stages
that already exist as standalone, tested components:

    candles → structure → SMC → filters → indicators → risk → rulebook → signal

This file owns the ORCHESTRATION and the analysis-output → rulebook-boolean
MAPPING. It does NOT implement the detectors — those are injected as stage
hooks (dependency injection). Each hook receives the shared PipelineContext and
populates it. Where a hook is not connected, the stage is skipped and the
relevant rulebook conditions default to False (so an unwired pipeline simply
produces no approved signals rather than crashing).

HOW TO WIRE (done by the caller, not here):
    pipeline = SignalPipeline(
        rulebook_engine=RulebookEngine(risk_config),
        structure_hook=my_structure_fn,   # (ctx, bar, history) -> None
        smc_hook=my_smc_fn,
        filter_hook=my_filter_fn,
        indicator_hook=my_indicator_fn,
        risk_hook=my_risk_fn,
    )
    signal = pipeline.process_bar(bar, history)

Each `*_hook` is a callable `(ctx: PipelineContext, bar, history) -> None` that
reads the candle/history and writes its results onto `ctx`. Search for
"# CONNECT:" markers below to see exactly what each stage is expected to set.

No heavy computation happens in this module — it is pure glue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from core.engine.rulebook_engine import (
    RulebookEngine,
    RulebookDecision,
    MANDATORY_CONDITIONS,
    OPTIONAL_CONDITIONS,
    INDICATOR_CONDITIONS,
)

# A stage hook reads bar/history and mutates the shared context in place.
StageHook = Callable[["PipelineContext", Any, Any], None]


def zone_allows_direction(zone: Optional[str], direction: Optional[str]) -> bool:
    """SMC premium/discount gate, matched to the trade direction.

    A LONG may enter only in DISCOUNT (lower half of the dealing range — buy a dip);
    a SHORT only in PREMIUM (upper half — sell a rally). See
    core/structure/premium_discount.py: premium = sell bias, discount = buy bias.

    The original gate (`zone in ("premium","discount")`) was True for BOTH zones and
    never compared to the direction, so a long could enter in premium and a short in
    discount — wrong-side-of-equilibrium entries whose proximal-edge limit never fills
    on a one-directional move (the live 'enters when it should NOT / never fills' bug).
    """
    if direction == "long":
        return zone == "discount"
    if direction == "short":
        return zone == "premium"
    return False


@dataclass
class PipelineContext:
    """Mutable scratchpad that accumulates each stage's output for one bar.

    Stage hooks WRITE onto this; the mapping helpers READ from it to build the
    rulebook boolean dicts. Field names below are the contract between the
    detectors you wire and the rulebook mapping in this file.
    """

    timestamp: datetime
    bar_index: int
    symbol: str = "XAUUSD"
    direction: str = "long"  # "long" / "short" — set by structure/SMC stage

    # --- structure stage (CONNECT: structure_hook) ---
    htf_bias: Optional[str] = None          # "long"/"short"/"neutral"
    structure_15m: Optional[str] = None     # aligned direction or None
    price_zone: Optional[str] = None        # "premium"/"discount"/"equilibrium"

    # --- SMC stage (CONNECT: smc_hook) ---
    sweep: Optional[Any] = None             # sweep detection result (or None)
    sweep_confirmed: bool = False
    fvg: Optional[Any] = None               # FVG result (or None)
    fvg_valid: bool = False
    fvg_fresh: bool = False
    order_block: Optional[Any] = None
    micro_choch: bool = False
    confirmation_candle: bool = False
    retraced_to_zone: bool = False

    # --- filter stage (CONNECT: filter_hook) ---
    in_kill_zone: bool = False
    news_clear: bool = True
    no_blocking_filters: bool = True
    dxy_aligned: bool = False
    overlap_session: bool = False
    clean_market_state: bool = False
    asia_liquidity_in_setup: bool = False

    # --- indicator stage (CONNECT: indicator_hook) — Phase 11 readings ---
    vwap_reading: Optional[Any] = None
    ema_reading: Optional[Any] = None
    divergence: Optional[Any] = None
    volume_profile_reading: Optional[Any] = None

    # --- risk stage (CONNECT: risk_hook) ---
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    net_rr: Optional[float] = None
    lot_size: float = 0.01
    rr_minimum_ok: bool = False
    daily_limits_ok: bool = True

    # extra fields for things like displacement strength, ob strength, etc.
    strong_displacement: bool = False
    ob_valid: bool = False
    liquidity_target_clear: bool = False
    volume_confirmation: bool = False
    multiple_confluence: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSignal:
    """Final output — shaped to be consumed directly by BacktestRunner.run()."""

    setup_id: str
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    lot_size: float
    grade: str
    score: int
    timestamp: datetime
    bar_index: int
    approved: bool
    decision: Optional[RulebookDecision] = None
    # which liquidity the captured sweep grabbed (swing_low/eql/pdl/swing_high/eqh/pdh);
    # telemetry only — lets reports/alerts break results down by sweep source.
    sweep_src: Optional[str] = None

    def to_signal_dict(self) -> Dict[str, Any]:
        """Format expected by backtesting.backtest_runner.BacktestRunner."""
        return {
            "setup_id": self.setup_id,
            "direction": self.direction,
            "entry": self.entry,
            "sl": self.sl,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "lot_size": self.lot_size,
            "timestamp": self.timestamp,
            "bar_index": self.bar_index,
            "grade": self.grade,
            "sweep_src": self.sweep_src,
        }

    def to_dict(self) -> Dict[str, Any]:
        d = self.to_signal_dict()
        d.update({"score": self.score, "approved": self.approved})
        return d


def _noop_hook(ctx: PipelineContext, bar: Any, history: Any) -> None:
    """Default stage hook — does nothing. Replace via constructor injection."""
    return None


class SignalPipeline:
    """Orchestrates the analysis stages and grades the result.

    The detector logic is injected; this class only sequences the stages and
    maps their outputs to the rulebook. Stages with no injected hook are skipped.
    """

    def __init__(
        self,
        rulebook_engine: RulebookEngine,
        *,
        structure_hook: Optional[StageHook] = None,
        smc_hook: Optional[StageHook] = None,
        filter_hook: Optional[StageHook] = None,
        indicator_hook: Optional[StageHook] = None,
        risk_hook: Optional[StageHook] = None,
        setup_id_fn: Optional[Callable[[PipelineContext], str]] = None,
        use_indicator_boosters: bool = True,
    ) -> None:
        self._rulebook = rulebook_engine
        self._structure_hook = structure_hook or _noop_hook
        self._smc_hook = smc_hook or _noop_hook
        self._filter_hook = filter_hook or _noop_hook
        self._indicator_hook = indicator_hook or _noop_hook
        self._risk_hook = risk_hook or _noop_hook
        self._setup_id_fn = setup_id_fn or self._default_setup_id
        self._use_indicator_boosters = use_indicator_boosters
        self._counter = 0

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def process_bar(self, bar: Any, history: Any = None) -> Optional[PipelineSignal]:
        """Run one bar through all stages. Returns a PipelineSignal or None.

        `bar` and `history` are passed straight to the injected hooks — their
        type is whatever your detectors expect (DataFrame, dict, custom object).
        """
        ctx = PipelineContext(
            timestamp=getattr(bar, "timestamp", None) or _dict_get(bar, "timestamp"),
            bar_index=getattr(bar, "bar_index", None) or _dict_get(bar, "bar_index", 0),
            symbol=getattr(bar, "symbol", None) or _dict_get(bar, "symbol", "XAUUSD"),
        )

        # STAGE 1-5: populate ctx via injected detectors (skipped if not wired)
        self._structure_hook(ctx, bar, history)
        self._smc_hook(ctx, bar, history)
        self._filter_hook(ctx, bar, history)
        self._indicator_hook(ctx, bar, history)
        self._risk_hook(ctx, bar, history)

        # STAGE 6: map analysis outputs → rulebook boolean dicts
        mandatory = self._build_mandatory(ctx)
        optional = self._build_optional(ctx)
        indicators = self._build_indicators(ctx) if self._use_indicator_boosters else None

        # STAGE 7: grade
        if ctx.entry is None or ctx.sl is None or ctx.tp1 is None:
            # Risk stage not wired / no setup — nothing to trade this bar.
            return None

        # Position sizing must have validated the risk against the configured limits.
        # A valid=False sizing result (SL too small / account can't afford the min lot)
        # was previously coerced to a 0.01 lot and the trade fired anyway, risking an
        # un-validated amount. Suppress instead.
        if not ctx.extra.get("sizing_valid", True):
            return None

        decision = self._rulebook.evaluate(
            direction=ctx.direction,
            mandatory=mandatory,
            optional=optional,
            net_rr=ctx.net_rr,
            symbol=ctx.symbol,
            timestamp=ctx.timestamp,
            setup_id=self._setup_id_fn(ctx),
            indicators=indicators,
        )

        # STAGE 8: build signal
        grade = decision.grade.grade if decision.grade else "D"
        score = decision.grade.score if decision.grade else 0
        return PipelineSignal(
            setup_id=self._setup_id_fn(ctx),
            direction=ctx.direction,
            entry=ctx.entry,
            sl=ctx.sl,
            tp1=ctx.tp1,
            tp2=ctx.tp2 if ctx.tp2 is not None else ctx.tp1,
            lot_size=ctx.lot_size,
            grade=grade,
            score=score,
            timestamp=ctx.timestamp,
            bar_index=ctx.bar_index,
            approved=decision.approved,
            decision=decision,
        )

    # ------------------------------------------------------------------ #
    # Analysis-output → rulebook-boolean mapping                           #
    # (This is the contract. Detectors set ctx fields; we read them here.) #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_mandatory(ctx: PipelineContext) -> Dict[str, bool]:
        m = {
            "htf_bias": ctx.htf_bias in ("long", "short"),
            "15m_aligned": ctx.structure_15m == ctx.direction,
            "price_zone": zone_allows_direction(ctx.price_zone, ctx.direction),
            "sweep": ctx.sweep is not None,
            "sweep_confirmation": ctx.sweep_confirmed,
            "fvg_valid": ctx.fvg_valid,
            "fvg_freshness": ctx.fvg_fresh,
            "kill_zone": ctx.in_kill_zone,
            "news_clear": ctx.news_clear,
            "retrace_to_zone": ctx.retraced_to_zone,
            "micro_choch": ctx.micro_choch,
            "confirmation_candle": ctx.confirmation_candle,
            "rr_minimum": ctx.rr_minimum_ok,
            "daily_limits_ok": ctx.daily_limits_ok,
            "no_blocking_filters": ctx.no_blocking_filters,
        }
        # Guard: every mandatory condition must have a key.
        return {k: bool(m.get(k, False)) for k in MANDATORY_CONDITIONS}

    @staticmethod
    def _build_optional(ctx: PipelineContext) -> Dict[str, bool]:
        o = {
            "dxy_aligned": ctx.dxy_aligned,
            "ob_valid": ctx.ob_valid,
            "strong_displacement": ctx.strong_displacement,
            "overlap_session": ctx.overlap_session,
            "liquidity_target_clear": ctx.liquidity_target_clear,
            "volume_confirmation": ctx.volume_confirmation,
            "clean_market_state": ctx.clean_market_state,
            "asia_liquidity_in_setup": ctx.asia_liquidity_in_setup,
            # GENUINE freshness of the TRADED zone (ctx.fvg is the captured/pinned zone at
            # emit, or the chosen zone on the non-pinned path) — state=='fresh', not the
            # always-true alias of the mandatory fvg_freshness. So this booster + the A+
            # freshness gate actually discriminate, and reflect the zone actually traded (#11).
            "fvg_fresh": bool(isinstance(ctx.fvg, dict) and ctx.fvg.get("state") == "fresh"),
            "multiple_confluence": ctx.multiple_confluence,
        }
        return {k: bool(o.get(k, False)) for k in OPTIONAL_CONDITIONS}

    @staticmethod
    def _build_indicators(ctx: PipelineContext) -> Dict[str, bool]:
        # Delegates to the Phase 11 bridge so the mapping logic lives in one place.
        from core.indicators.indicator_grader import build_indicator_results
        results = build_indicator_results(
            direction=ctx.direction,
            vwap=ctx.vwap_reading,
            ema=ctx.ema_reading,
            divergence=ctx.divergence,
            volume_profile=ctx.volume_profile_reading,
        )
        return {k: bool(results.get(k, False)) for k in INDICATOR_CONDITIONS}

    # ------------------------------------------------------------------ #

    def _default_setup_id(self, ctx: PipelineContext) -> str:
        self._counter += 1
        ts = ctx.timestamp.strftime("%Y%m%d-%H%M") if hasattr(ctx.timestamp, "strftime") else "00000000-0000"
        return f"XAU-{ts}-{ctx.direction.upper()}-{self._counter:04d}"


def _dict_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default
