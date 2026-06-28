"""Tests for SignalPipeline skeleton — wiring scaffold (no heavy compute)."""

import pytest
from datetime import datetime, timezone
from core.engine.signal_pipeline import (
    SignalPipeline, PipelineContext, PipelineSignal, zone_allows_direction,
)
from core.engine.rulebook_engine import RulebookEngine

CONFIG = {
    "rr_tiers": {
        "min_to_enter": 2.0,
        "required_for_grade_b": 1.5,
        "required_for_grade_a": 2.0,
        "required_for_grade_a_plus": 2.5,
    },
}

NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)


def _bar(bar_index=0):
    return {"timestamp": NOW, "bar_index": bar_index, "symbol": "XAUUSD"}


@pytest.fixture
def engine():
    return RulebookEngine(CONFIG)


# --- stub hooks: populate ctx with no real computation ---

def _full_structure(ctx, bar, history):
    ctx.direction = "long"
    ctx.htf_bias = "long"
    ctx.structure_15m = "long"
    ctx.price_zone = "discount"


def _full_smc(ctx, bar, history):
    ctx.sweep = {"ok": True}
    ctx.sweep_confirmed = True
    ctx.fvg = {"ok": True}
    ctx.fvg_valid = True
    ctx.fvg_fresh = True
    ctx.micro_choch = True
    ctx.confirmation_candle = True
    ctx.retraced_to_zone = True
    ctx.ob_valid = True
    ctx.strong_displacement = True


def _full_filters(ctx, bar, history):
    ctx.in_kill_zone = True
    ctx.news_clear = True
    ctx.no_blocking_filters = True
    ctx.dxy_aligned = True
    ctx.overlap_session = True


def _full_risk(ctx, bar, history):
    ctx.entry = 2650.0
    ctx.sl = 2640.0
    ctx.tp1 = 2670.0
    ctx.tp2 = 2685.0
    ctx.net_rr = 3.0
    ctx.rr_minimum_ok = True
    ctx.lot_size = 0.02


class TestUnwired:
    def test_noop_pipeline_returns_none(self, engine):
        # No hooks → risk stage never sets entry/sl/tp → no signal
        pipe = SignalPipeline(engine)
        assert pipe.process_bar(_bar()) is None


class TestFullyWired:
    @pytest.fixture
    def pipe(self, engine):
        return SignalPipeline(
            engine,
            structure_hook=_full_structure,
            smc_hook=_full_smc,
            filter_hook=_full_filters,
            risk_hook=_full_risk,
        )

    def test_produces_signal(self, pipe):
        sig = pipe.process_bar(_bar())
        assert sig is not None
        assert isinstance(sig, PipelineSignal)
        assert sig.approved is True

    def test_signal_shape_for_backtest(self, pipe):
        sig = pipe.process_bar(_bar(bar_index=5))
        d = sig.to_signal_dict()
        for key in ("setup_id", "direction", "entry", "sl", "tp1", "tp2", "lot_size", "bar_index"):
            assert key in d
        assert d["entry"] == 2650.0
        assert d["sl"] == 2640.0
        assert d["bar_index"] == 5

    def test_grade_assigned(self, pipe):
        sig = pipe.process_bar(_bar())
        # all mandatory pass + several optional + rr 3.0 → A or A+
        assert sig.grade in ("A", "A+")

    def test_setup_id_generated(self, pipe):
        sig = pipe.process_bar(_bar())
        assert sig.setup_id.startswith("XAU-")
        assert "LONG" in sig.setup_id


class TestMandatoryGate:
    def test_missing_mandatory_not_approved(self, engine):
        def bad_smc(ctx, bar, history):
            _full_smc(ctx, bar, history)
            ctx.sweep = None  # break one mandatory
            ctx.sweep_confirmed = False

        pipe = SignalPipeline(
            engine,
            structure_hook=_full_structure,
            smc_hook=bad_smc,
            filter_hook=_full_filters,
            risk_hook=_full_risk,
        )
        sig = pipe.process_bar(_bar())
        assert sig is not None  # signal still built (for logging/rejection)
        assert sig.approved is False


class TestZoneAllowsDirection:
    """Unit truth-table for the SMC zone gate: long only in discount, short only in
    premium. Locks the root-cause fix (the old gate accepted ANY zone)."""

    def test_long_only_in_discount(self):
        assert zone_allows_direction("discount", "long") is True
        assert zone_allows_direction("premium", "long") is False
        assert zone_allows_direction("equilibrium", "long") is False
        assert zone_allows_direction("undefined", "long") is False
        assert zone_allows_direction(None, "long") is False

    def test_short_only_in_premium(self):
        assert zone_allows_direction("premium", "short") is True
        assert zone_allows_direction("discount", "short") is False
        assert zone_allows_direction("equilibrium", "short") is False
        assert zone_allows_direction(None, "short") is False

    def test_neutral_or_none_direction_never_allowed(self):
        assert zone_allows_direction("discount", "neutral") is False
        assert zone_allows_direction("premium", None) is False


class TestZoneDirectionGateInPipeline:
    """Root-cause regression lock: a LONG in PREMIUM (or a SHORT in DISCOUNT) is on the
    wrong side of equilibrium and must NOT be approved. The pre-fix gate accepted any
    zone, letting these wrong-side setups through (the 'enters when it should NOT' bug)."""

    def _pipe(self, engine, direction, zone):
        def structure(ctx, bar, history):
            ctx.direction = direction
            ctx.htf_bias = direction
            ctx.structure_15m = direction
            ctx.price_zone = zone
        return SignalPipeline(
            engine, structure_hook=structure, smc_hook=_full_smc,
            filter_hook=_full_filters, risk_hook=_full_risk,
        )

    def test_long_in_premium_rejected(self, engine):
        sig = self._pipe(engine, "long", "premium").process_bar(_bar())
        assert sig is not None and sig.approved is False

    def test_short_in_discount_rejected(self, engine):
        sig = self._pipe(engine, "short", "discount").process_bar(_bar())
        assert sig is not None and sig.approved is False

    def test_long_in_discount_approved(self, engine):
        sig = self._pipe(engine, "long", "discount").process_bar(_bar())
        assert sig is not None and sig.approved is True

    def test_short_in_premium_approved(self, engine):
        sig = self._pipe(engine, "short", "premium").process_bar(_bar())
        assert sig is not None and sig.approved is True


class TestIndicatorToggle:
    def _indicator_hook(self, ctx, bar, history):
        from core.indicators.vwap import VWAPReading, VWAPBias
        from core.indicators.ema import EMAReading
        ctx.vwap_reading = VWAPReading(NOW, 2645, "london", VWAPBias.ABOVE, 1.0, 2650)
        ctx.ema_reading = EMAReading(NOW, 2648, 2640, 2650, "long")

    def test_indicators_boost_score(self, engine):
        pipe = SignalPipeline(
            engine,
            structure_hook=_full_structure,
            smc_hook=_full_smc,
            filter_hook=_full_filters,
            indicator_hook=self._indicator_hook,
            risk_hook=_full_risk,
            use_indicator_boosters=True,
        )
        sig = pipe.process_bar(_bar())
        assert sig.decision.grade.indicator_score > 0

    def test_boosters_disabled(self, engine):
        pipe = SignalPipeline(
            engine,
            structure_hook=_full_structure,
            smc_hook=_full_smc,
            filter_hook=_full_filters,
            indicator_hook=self._indicator_hook,
            risk_hook=_full_risk,
            use_indicator_boosters=False,
        )
        sig = pipe.process_bar(_bar())
        assert sig.decision.grade.indicator_score == 0


class TestContext:
    def test_context_defaults(self):
        ctx = PipelineContext(timestamp=NOW, bar_index=0)
        assert ctx.direction == "long"
        assert ctx.lot_size == 0.01
        assert ctx.news_clear is True
