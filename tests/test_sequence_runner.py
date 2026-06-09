"""Tests for SequenceRunner — scripted hooks drive the state sequence."""

import pytest
from datetime import datetime, timezone, timedelta
from core.engine.sequence_runner import SequenceRunner
from core.engine.state_machine import State

NOW = datetime(2026, 1, 21, 16, 0, tzinfo=timezone.utc)
CONFIG = {"rr_tiers": {"min_to_enter": 2.0, "required_for_grade_b": 1.5,
                       "required_for_grade_a": 2.0, "required_for_grade_a_plus": 2.5}}


class _Script:
    """Holds ctx field values to apply on the next bar(s)."""
    def __init__(self):
        self.fields = {}
    def apply(self, ctx, bar, history):
        for k, v in self.fields.items():
            setattr(ctx, k, v)


def _hooks(script, risk_ok=True):
    def risk_hook(ctx, bar, history):
        if risk_ok and ctx.sweep is not None and ctx.fvg is not None:
            ctx.entry, ctx.sl, ctx.tp1, ctx.tp2 = 2650.0, 2640.0, 2670.0, 2685.0
            ctx.net_rr, ctx.rr_minimum_ok, ctx.lot_size = 3.0, True, 0.01
    return {
        "structure_hook": script.apply, "smc_hook": script.apply,
        "filter_hook": script.apply, "indicator_hook": lambda c, b, h: None,
        "risk_hook": risk_hook,
    }


def _bar(i):
    return {"timestamp": NOW + timedelta(minutes=15 * i), "bar_index": i, "symbol": "XAUUSD"}


def _runner(script, **kw):
    return SequenceRunner(CONFIG, hooks=_hooks(script), **kw)


class TestSequenceProgression:
    def test_advances_step_by_step(self):
        s = _Script()
        r = _runner(s)
        assert r.state == State.WAITING_FOR_HTF_BIAS

        s.fields.update({"htf_bias": "long", "direction": "long"})
        r.on_bar(_bar(0), {})
        assert r.state == State.WAITING_FOR_15M_ALIGNMENT

        s.fields["structure_15m"] = "long"
        r.on_bar(_bar(1), {})
        assert r.state == State.WAITING_FOR_PRICE_IN_ZONE

        s.fields["price_zone"] = "discount"
        r.on_bar(_bar(2), {})
        assert r.state == State.WAITING_FOR_LIQUIDITY_SWEEP

        s.fields.update({"sweep": {"level": 2640}, "sweep_confirmed": True})
        r.on_bar(_bar(3), {})
        assert r.state == State.WAITING_FOR_VALID_FVG_OR_OB

    def test_full_sequence_emits_signal(self):
        s = _Script()
        r = _runner(s)
        # set the whole setup up front; the runner zooms through states then emits
        s.fields.update({
            "htf_bias": "long", "direction": "long", "structure_15m": "long",
            "price_zone": "discount", "sweep": {"level": 2640}, "sweep_confirmed": True,
            "fvg_valid": True, "fvg_fresh": True, "fvg": {"top": 2648, "bottom": 2644},
            "retraced_to_zone": True, "micro_choch": True, "confirmation_candle": True,
            "in_kill_zone": True, "news_clear": True, "no_blocking_filters": True,
            "daily_limits_ok": True,
        })
        sig = r.on_bar(_bar(0), {})
        assert sig is not None
        assert sig.approved is True
        assert sig.direction == "long"
        assert sig.entry == 2650.0  # value set by the stub risk hook


class TestInvalidation:
    def test_bias_lost_resets(self):
        s = _Script()
        r = _runner(s)
        s.fields.update({"htf_bias": "long", "direction": "long"})
        r.on_bar(_bar(0), {})
        assert r.state == State.WAITING_FOR_15M_ALIGNMENT
        # bias lost on the next bar → reset
        s.fields["htf_bias"] = "neutral"
        r.on_bar(_bar(1), {})
        assert r.state == State.WAITING_FOR_HTF_BIAS

    def test_expiry_prevents_stale_completion(self):
        # An incomplete setup must never emit; expiry keeps resetting it so it
        # never festers into a late state.
        s = _Script()
        r = _runner(s, setup_expiry_bars=3)
        s.fields.update({"htf_bias": "long", "direction": "long",
                         "structure_15m": "long"})  # gets to 15M then PRICE_IN_ZONE, then stalls
        early = {State.WAITING_FOR_HTF_BIAS, State.WAITING_FOR_15M_ALIGNMENT,
                 State.WAITING_FOR_PRICE_IN_ZONE}
        for i in range(12):
            sig = r.on_bar(_bar(i), {})
            assert sig is None                 # never completes
        assert r.state in early                # never advanced past the stall point


class TestCooldown:
    def test_cooldown_after_signal(self):
        s = _Script()
        r = _runner(s, cooldown_bars=3)
        s.fields.update({
            "htf_bias": "long", "direction": "long", "structure_15m": "long",
            "price_zone": "discount", "sweep": {"level": 2640}, "sweep_confirmed": True,
            "fvg_valid": True, "fvg_fresh": True, "fvg": {"top": 2648, "bottom": 2644},
            "retraced_to_zone": True, "micro_choch": True, "confirmation_candle": True,
            "in_kill_zone": True, "news_clear": True, "no_blocking_filters": True,
            "daily_limits_ok": True,
        })
        sig = r.on_bar(_bar(0), {})
        assert sig is not None
        assert r.state == State.COOLDOWN
        # during cooldown, no new signals even if setup still valid
        assert r.on_bar(_bar(1), {}) is None
        assert r.on_bar(_bar(2), {}) is None
        r.on_bar(_bar(3), {})  # cooldown ends → back to hunting
        assert r.state == State.WAITING_FOR_HTF_BIAS


class TestNoSignalWhenGatesFail:
    def test_no_killzone_no_approved_signal(self):
        s = _Script()
        r = _runner(s)
        s.fields.update({
            "htf_bias": "long", "direction": "long", "structure_15m": "long",
            "price_zone": "discount", "sweep": {"level": 2640}, "sweep_confirmed": True,
            "fvg_valid": True, "fvg_fresh": True, "fvg": {"top": 2648, "bottom": 2644},
            "retraced_to_zone": True, "micro_choch": True, "confirmation_candle": True,
            "in_kill_zone": False, "news_clear": True, "no_blocking_filters": True,
            "daily_limits_ok": True,
        })
        sig = r.on_bar(_bar(0), {})
        assert sig is None  # kill_zone gate fails → not approved
