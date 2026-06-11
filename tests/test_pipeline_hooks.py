"""Tests for pipeline_hooks skeleton — construction + contract (no heavy compute)."""

import pytest
import pandas as pd
from datetime import datetime, timezone
from core.engine.signal_pipeline import PipelineContext
from core.engine.pipeline_hooks import (
    build_default_hooks, compute_atr, select_fvg,
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


class TestSelectFvgDirectionAware:
    """Fix #1: select_fvg must prefer a correct-SIDE zone over a nearer wrong-side one
    (a zone price has already passed is not actionable), but never drop to None."""

    def _df(self, price=4180.0):
        idx = pd.date_range("2026-06-10", periods=20, freq="5min", tz="UTC")
        return pd.DataFrame({"open": price, "high": price + 5, "low": price - 5,
                             "close": price, "volume": 100.0}, index=idx)

    def test_short_prefers_above_price_over_nearer_passed_zone(self):
        df = self._df(4180.0)
        ts = df.index[-2]
        near_passed = {"top": 4155, "bottom": 4150, "fvg_type": "bear", "confirm_ts": ts, "state": "fresh"}
        far_valid = {"top": 4215, "bottom": 4210, "fvg_type": "bear", "confirm_ts": ts, "state": "fresh"}
        chosen = select_fvg([near_passed, far_valid], df, {"fvg_freshness_enabled": True})
        assert chosen is far_valid  # correct-side wins despite being farther

    def test_long_prefers_below_price_over_nearer_passed_zone(self):
        df = self._df(4180.0)
        ts = df.index[-2]
        near_passed = {"top": 4210, "bottom": 4205, "fvg_type": "bull", "confirm_ts": ts, "state": "fresh"}
        far_valid = {"top": 4150, "bottom": 4145, "fvg_type": "bull", "confirm_ts": ts, "state": "fresh"}
        chosen = select_fvg([near_passed, far_valid], df, {"fvg_freshness_enabled": True})
        assert chosen is far_valid

    def test_wrong_side_only_still_returned_as_fallback(self):
        df = self._df(4180.0)
        only_passed = {"top": 4155, "bottom": 4150, "fvg_type": "bear",
                       "confirm_ts": df.index[-2], "state": "fresh"}
        assert select_fvg([only_passed], df, {"fvg_freshness_enabled": True}) is only_passed

    def test_legacy_flag_ignores_direction_scoring(self):
        df = self._df(4180.0)
        ts = df.index[-2]
        near_passed = {"top": 4155, "bottom": 4150, "fvg_type": "bear", "confirm_ts": ts, "state": "fresh"}
        far_valid = {"top": 4215, "bottom": 4210, "fvg_type": "bear", "confirm_ts": ts, "state": "fresh"}
        # legacy = newest-first = candidates[0], regardless of side
        chosen = select_fvg([near_passed, far_valid], df, {"fvg_freshness_enabled": False})
        assert chosen is near_passed
