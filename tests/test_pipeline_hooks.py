"""Tests for pipeline_hooks skeleton — construction + contract (no heavy compute)."""

import pytest
import pandas as pd
from datetime import datetime, timezone
from core.engine.signal_pipeline import PipelineContext
from core.engine.pipeline_hooks import (
    build_default_hooks, compute_atr,
    make_structure_hook, make_smc_hook, make_filter_hook,
    make_indicator_hook, make_risk_hook,
)

NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)
CFG = {"rr_tiers": {"min_to_enter": 2.0, "required_for_grade_a": 2.0,
                    "required_for_grade_a_plus": 2.5, "required_for_grade_b": 1.5}}


def _ctx():
    return PipelineContext(timestamp=NOW, bar_index=0, symbol="XAUUSDT")


class TestComputeATR:
    def test_too_short_returns_zero(self):
        df = pd.DataFrame({"open": [10] * 5, "high": [11] * 5, "low": [9] * 5, "close": [10] * 5})
        assert compute_atr(df, period=14) == 0.0

    def test_basic_atr(self):
        df = pd.DataFrame({"open": [10] * 20, "high": [12] * 20, "low": [9] * 20,
                           "close": [11] * 20, "volume": [100] * 20})
        assert compute_atr(df) == pytest.approx(3.0, abs=0.01)

    def test_none_df(self):
        assert compute_atr(None) == 0.0


class TestFactories:
    def test_build_default_hooks_returns_five(self):
        hooks = build_default_hooks(CFG)
        assert set(hooks) == {"structure_hook", "smc_hook", "filter_hook",
                              "indicator_hook", "risk_hook"}
        assert all(callable(h) for h in hooks.values())

    def test_each_factory_returns_callable(self):
        assert callable(make_structure_hook(CFG))
        assert callable(make_smc_hook(CFG))
        assert callable(make_filter_hook(CFG))
        assert callable(make_indicator_hook(CFG))
        assert callable(make_risk_hook(CFG))


class TestDefensiveEmptyHistory:
    """Hooks must no-op (not crash) when timeframes are missing."""

    @pytest.mark.parametrize("name", ["structure_hook", "smc_hook", "filter_hook",
                                       "indicator_hook", "risk_hook"])
    def test_empty_history_no_crash(self, name):
        hooks = build_default_hooks(CFG)
        ctx = _ctx()
        hooks[name](ctx, {"timestamp": NOW}, {})  # empty history dict
        # risk stage must not have produced a setup
        assert ctx.entry is None

    def test_filter_hook_sets_session_flags(self):
        # filter hook only needs ctx.timestamp, no history TFs
        hook = make_filter_hook(CFG)
        ctx = _ctx()
        hook(ctx, {}, {})
        assert isinstance(ctx.in_kill_zone, bool)
        assert ctx.news_clear is True  # placeholder default


class TestStructureHookOnTinyData:
    def test_runs_without_crash_on_small_frames(self):
        # 60 flat bars — detectors should run and not crash; bias likely neutral
        idx = pd.date_range("2026-01-01", periods=60, freq="1h", tz="UTC")
        df = pd.DataFrame({"open": 2650.0, "high": 2655.0, "low": 2645.0,
                           "close": 2650.0, "volume": 100.0}, index=idx)
        hook = make_structure_hook(CFG)
        ctx = _ctx()
        hook(ctx, {}, {"4h": df, "1h": df})
        # htf_bias should be set to something (string) or remain None — just no crash
        assert ctx.htf_bias is None or isinstance(ctx.htf_bias, str)


class TestRiskHookNetRR:
    """F1: the live risk hook's net_rr is NET of execution costs (RRCalculator), not
    gross price geometry — and the gross value is stashed for display transparency."""

    def test_net_rr_is_below_gross(self):
        from core.engine.pipeline_hooks import make_risk_hook
        idx = pd.date_range("2026-06-11", periods=40, freq="5min", tz="UTC")
        df = pd.DataFrame({"open": [4000.0] * 40, "high": [4003.0] * 40,
                           "low": [3997.0] * 40, "close": [4000.0] * 40,
                           "volume": [100] * 40}, index=idx)
        ctx = _ctx()
        ctx.direction = "long"
        ctx.sweep = {"level": 3994.0}
        ctx.fvg = {"top": 4000.0, "bottom": 3998.0}
        ctx.news_clear = True
        make_risk_hook(CFG)(ctx, {"timestamp": NOW}, {"5m": df})
        if ctx.entry is not None:                       # a trade was sized
            assert "gross_rr" in ctx.extra
            assert ctx.net_rr < ctx.extra["gross_rr"]   # costs strictly subtracted


class TestSKEntryGrid:
    """SK-System entry grid (Stefan Kassing) — flag-gated FVG-depth resolver, default OFF.
    Grid [0.50, 0.559, 0.618, 0.667]; sk_entry_grid_mode collapses it to one entry depth."""

    def test_off_returns_none(self):
        from core.engine.pipeline_hooks import _sk_grid_depth
        assert _sk_grid_depth({}) is None
        assert _sk_grid_depth({"sk_entry_grid": False}) is None

    def test_mean_is_grid_average(self):
        from core.engine.pipeline_hooks import _sk_grid_depth
        d = _sk_grid_depth({"sk_entry_grid": True})
        assert d == pytest.approx((0.50 + 0.559 + 0.618 + 0.667) / 4)

    def test_golden_picks_0618(self):
        from core.engine.pipeline_hooks import _sk_grid_depth
        assert _sk_grid_depth({"sk_entry_grid": True,
                               "sk_entry_grid_mode": "golden"}) == pytest.approx(0.618)

    def test_deep_and_shallow(self):
        from core.engine.pipeline_hooks import _sk_grid_depth
        assert _sk_grid_depth({"sk_entry_grid": True,
                               "sk_entry_grid_mode": "deep"}) == pytest.approx(0.667)
        assert _sk_grid_depth({"sk_entry_grid": True,
                               "sk_entry_grid_mode": "shallow"}) == pytest.approx(0.50)

    def test_float_mode_is_exact_depth(self):
        from core.engine.pipeline_hooks import _sk_grid_depth
        assert _sk_grid_depth({"sk_entry_grid": True,
                               "sk_entry_grid_mode": 0.72}) == pytest.approx(0.72)

    def test_custom_levels(self):
        from core.engine.pipeline_hooks import _sk_grid_depth
        assert _sk_grid_depth({"sk_entry_grid": True,
                               "sk_entry_grid_levels": [0.4, 0.6]}) == pytest.approx(0.5)

    def test_clamped_to_max(self):
        from core.engine.pipeline_hooks import _sk_grid_depth
        assert _sk_grid_depth({"sk_entry_grid": True,
                               "sk_entry_grid_mode": 5.0}) == pytest.approx(0.95)

    def test_grid_ignores_entry_depth_pct(self):
        # When both are set the SK grid WINS — entry_depth_pct is not consulted.
        from core.engine.pipeline_hooks import _sk_grid_depth
        d = _sk_grid_depth({"sk_entry_grid": True, "sk_entry_grid_mode": "deep",
                            "entry_depth_pct": 0.70})
        assert d == pytest.approx(0.667)

    def test_grid_overrides_entry_depth_pct_in_risk_hook(self):
        # entry_depth_pct=0.70 would enter at 0.70 of the gap; the SK grid (deep=0.667)
        # must override → entry lands at 0.667 of the FVG gap instead.
        from core.engine.pipeline_hooks import make_risk_hook
        idx = pd.date_range("2026-06-11", periods=40, freq="5min", tz="UTC")
        df = pd.DataFrame({"open": [4000.0] * 40, "high": [4004.0] * 40,
                           "low": [3996.0] * 40, "close": [4000.0] * 40,
                           "volume": [100] * 40}, index=idx)
        ctx = _ctx()
        ctx.direction = "long"
        ctx.sweep = {"level": 3985.0}
        ctx.fvg = {"top": 4000.0, "bottom": 3990.0}   # gap = 10
        ctx.news_clear = True
        cfg = {**CFG, "entry_depth_pct": 0.70,
               "sk_entry_grid": True, "sk_entry_grid_mode": "deep"}
        make_risk_hook(cfg)(ctx, {"timestamp": NOW}, {"5m": df})
        if ctx.entry is not None:                     # a trade was sized
            # long entry = fvg_top - depth*gap = 4000 - 0.667*10 = 3993.33
            assert ctx.entry == pytest.approx(4000.0 - 0.667 * 10.0, abs=1e-6)


class TestComputeRSI:
    """RSI powers the momentum-confirmation gate: high when rising, low when falling."""

    def test_rising_series_high_rsi(self):
        from core.engine.pipeline_hooks import compute_rsi
        assert compute_rsi(pd.DataFrame({"close": [100 + i for i in range(30)]}), 14) > 70

    def test_falling_series_low_rsi(self):
        from core.engine.pipeline_hooks import compute_rsi
        assert compute_rsi(pd.DataFrame({"close": [100 - i for i in range(30)]}), 14) < 30

    def test_too_short_returns_neutral(self):
        from core.engine.pipeline_hooks import compute_rsi
        assert compute_rsi(pd.DataFrame({"close": [100, 101]}), 14) == 50.0


class TestComputeEmaBias:
    """compute_ema_bias: exec-TF EMA50/200 trend used by the trend_gate."""

    def _df(self, closes):
        n = len(closes)
        idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
        return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                             "close": closes, "volume": [1.0] * n}, index=idx)

    def test_uptrend_is_long(self):
        from core.engine.pipeline_hooks import compute_ema_bias
        assert compute_ema_bias(self._df([100.0 + i for i in range(250)])) == "long"

    def test_downtrend_is_short(self):
        from core.engine.pipeline_hooks import compute_ema_bias
        assert compute_ema_bias(self._df([350.0 - i for i in range(250)])) == "short"

    def test_flat_is_neutral(self):
        from core.engine.pipeline_hooks import compute_ema_bias
        assert compute_ema_bias(self._df([100.0] * 250)) == "neutral"

    def test_insufficient_bars_neutral(self):
        from core.engine.pipeline_hooks import compute_ema_bias
        assert compute_ema_bias(self._df([100.0 + i for i in range(150)])) == "neutral"


class TestRejectionConfirms:
    """rejection_confirms: the REAL confirmation gate (Layer 4) — a decisive rejection
    candle that reclaims the proximal FVG edge, vs the weak body-color default."""

    def _fn(self):
        from core.engine.pipeline_hooks import rejection_confirms
        return rejection_confirms

    # FVG lo=100 hi=102 ; atr=2 k=0.3 -> min body 0.6
    def test_long_valid_rejection(self):
        # bullish, decisive body, dipped into zone (low<=hi), closed back above hi
        assert self._fn()(101.0, 103.5, 99.5, 103.0, 100.0, 102.0, True, 2.0) is True

    def test_long_weak_body_rejected(self):
        # would reclaim, but body 0.3 < 0.6*atr
        assert self._fn()(102.0, 102.5, 99.5, 102.3, 100.0, 102.0, True, 2.0) is False

    def test_long_no_reclaim_rejected(self):
        # decisive + dipped, but CLOSED inside the gap (below hi) — not reclaimed
        assert self._fn()(100.0, 101.8, 99.5, 101.5, 100.0, 102.0, True, 2.0) is False

    def test_long_breakout_no_zone_touch_rejected(self):
        # never dipped to the zone (low > hi) — a breakout, not a rejection
        assert self._fn()(103.0, 105.5, 102.5, 105.0, 100.0, 102.0, True, 2.0) is False

    def test_short_valid_rejection(self):
        # bearish, decisive, wicked up into zone (high>=lo), closed back below lo
        assert self._fn()(101.0, 102.5, 98.5, 99.0, 100.0, 102.0, False, 2.0) is True

    def test_short_bounce_continuation_rejected(self):
        # the ETH 11:50 pattern: bearish body but CLOSED above the proximal edge (in the gap)
        assert self._fn()(102.0, 102.5, 100.2, 100.5, 100.0, 102.0, False, 2.0) is False

    def test_atr_zero_falls_back_to_any_body(self):
        # atr<=0 → strength = body>0; still needs the reject geometry
        assert self._fn()(101.0, 103.5, 99.5, 103.0, 100.0, 102.0, True, 0.0) is True
