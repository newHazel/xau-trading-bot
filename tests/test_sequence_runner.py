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


class TestRepinGating:
    """The periodic FVG re-pin must only fire BEFORE the retrace gate is validated.
    Re-pinning after retrace passed would let entry/SL/TP use a zone that retrace was
    never validated against (the cross-bar form of the 08:40 entry-vs-retrace bug)."""

    def _df(self, price):
        import pandas as pd
        return {"15m": pd.DataFrame({"close": [price]},
                                    index=pd.to_datetime(["2026-06-10T14:00:00Z"]))}

    FAR = {"top": 4242.0, "bottom": 4238.0}   # ~58 pts above price 4180
    NEAR = {"top": 4184.0, "bottom": 4182.0}  # ~2 pts above price 4180

    def test_repins_to_nearer_zone_while_waiting_for_retrace(self):
        r = _runner(_Script())
        r._sm.force_state(State.WAITING_FOR_RETRACE_TO_ZONE, "test", NOW)
        r._captured["fvg"] = self.FAR
        r._bars_since_repin = r._repin_interval - 1  # next call hits the interval
        out = r._maybe_repin(self.FAR, self.NEAR, self._df(4180.0))
        assert out is self.NEAR
        assert r._captured["fvg"] is self.NEAR  # entry will now use the nearer zone

    def test_does_not_repin_after_retrace_passed(self):
        # FSM has advanced past retrace → the validated zone must be frozen.
        r = _runner(_Script())
        r._sm.force_state(State.WAITING_FOR_MICRO_CHOCH, "test", NOW)
        r._captured["fvg"] = self.FAR
        r._bars_since_repin = 999  # interval long satisfied — only the state guard matters
        out = r._maybe_repin(self.FAR, self.NEAR, self._df(4180.0))
        assert out is self.FAR
        assert r._captured["fvg"] is self.FAR  # entry == retrace-validated zone

    def test_disabled_flag_never_repins(self):
        r = SequenceRunner({**CONFIG, "fvg_freshness_enabled": False}, hooks=_hooks(_Script()))
        r._sm.force_state(State.WAITING_FOR_RETRACE_TO_ZONE, "test", NOW)
        r._captured["fvg"] = self.FAR
        r._bars_since_repin = 999
        out = r._maybe_repin(self.FAR, self.NEAR, self._df(4180.0))
        assert out is self.FAR  # legacy behavior: pinned zone never swapped


class TestZoneRejectionConfirmation:
    """Fix #6: the confirmation step requires a REAL rejection at the pinned zone
    (price TAGS the zone and CLOSES BACK OUT in-direction), not just a same-colour
    candle — so it cannot fire while price is still outside/passing through the zone."""

    ZONE = {"top": 4158.1, "bottom": 4154.3}  # short setup

    def _df(self, o, h, l, c):
        import pandas as pd
        idx = pd.date_range("2026-06-10", periods=5, freq="15min", tz="UTC")
        return {"15m": pd.DataFrame({"open": [4156.0] * 4 + [o], "high": [4157.0] * 4 + [h],
                                     "low": [4153.0] * 4 + [l], "close": [4155.0] * 4 + [c]}, index=idx)}

    def _armed(self):
        s = _Script()
        s.fields.update({"htf_bias": "short", "direction": "short"})
        r = _runner(s)  # default config → require_zone_rejection True
        r._locked_direction = "short"
        r._captured["fvg"] = dict(self.ZONE)
        r._captured["sweep"] = {"level": 4158}
        r._sm.force_state(State.WAITING_FOR_CONFIRMATION_CANDLE, "test", NOW)
        return r

    def test_real_rejection_advances(self):
        r = self._armed()
        # short rejection: high tags zone (>=4154.3), close back below (<4154.3), bearish
        r.on_bar(_bar(1), self._df(o=4156.0, h=4159.0, l=4150.0, c=4150.0))
        assert r.state != State.WAITING_FOR_CONFIRMATION_CANDLE  # advanced past confirmation

    def test_bearish_candle_without_tag_does_not_advance(self):
        r = self._armed()
        # bearish candle, but price never tags up into the zone (the body-colour trap)
        r.on_bar(_bar(1), self._df(o=4148.0, h=4150.0, l=4145.0, c=4146.0))
        assert r.state == State.WAITING_FOR_CONFIRMATION_CANDLE  # no real rejection → stuck
