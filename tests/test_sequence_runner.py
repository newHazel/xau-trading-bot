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


class TestCooldownAfterApproval:
    """#5 (default ON): cooldown burns only when a signal is actually SENT. A
    completed-but-rejected (non-tradeable) sequence resets to hunting instead of
    muting a real follow-up for the cooldown window."""

    REJECTED = {  # full setup but kill_zone False → completes yet is NOT tradeable
        "htf_bias": "long", "direction": "long", "structure_15m": "long",
        "price_zone": "discount", "sweep": {"level": 2640}, "sweep_confirmed": True,
        "fvg_valid": True, "fvg_fresh": True, "fvg": {"top": 2648, "bottom": 2644},
        "retraced_to_zone": True, "micro_choch": True, "confirmation_candle": True,
        "in_kill_zone": False, "news_clear": True, "no_blocking_filters": True,
        "daily_limits_ok": True,
    }

    def test_default_resets_after_reject_not_cooldown(self):
        s = _Script(); s.fields.update(self.REJECTED)
        r = _runner(s)  # #5 default ON
        assert r.on_bar(_bar(0), {}) is None
        assert r.state == State.WAITING_FOR_HTF_BIAS  # reset to hunting, no cooldown

    def test_flag_off_restores_legacy_cooldown(self):
        s = _Script(); s.fields.update(self.REJECTED)
        r = SequenceRunner({**CONFIG, "cooldown_after_approval_only": False}, hooks=_hooks(s))
        assert r.on_bar(_bar(0), {}) is None
        assert r.state == State.COOLDOWN  # legacy: cooldown even after a reject


class TestNearMiss:
    """A completed setup rejected only at the kill-zone gate is recorded as a
    near-miss with the blocking reason; an INCOMPLETE setup is not."""

    OFF_KILLZONE = {
        "htf_bias": "long", "direction": "long", "structure_15m": "long",
        "price_zone": "discount", "sweep": {"level": 2640}, "sweep_confirmed": True,
        "fvg_valid": True, "fvg_fresh": True, "fvg": {"top": 2648, "bottom": 2644},
        "retraced_to_zone": True, "micro_choch": True, "confirmation_candle": True,
        "in_kill_zone": False, "news_clear": True, "no_blocking_filters": True,
        "daily_limits_ok": True,
    }

    def test_completed_but_off_killzone_is_near_miss(self):
        s = _Script(); s.fields.update(self.OFF_KILLZONE)
        r = _runner(s)
        assert r.on_bar(_bar(0), {}) is None         # rejected, no signal
        nm = r.last_near_miss
        assert nm is not None and "kill-zone" in nm["reason"]
        assert nm["direction"] == "long"

    def test_incomplete_setup_is_not_a_near_miss(self):
        s = _Script(); s.fields.update({"htf_bias": "long", "direction": "long"})
        r = _runner(s)
        r.on_bar(_bar(0), {})
        assert r.last_near_miss is None              # sequence never completed

    def test_ignore_kill_zone_bypasses_the_gate(self):
        # same off-kill-zone setup, but with the experiment flag the gate no longer blocks
        s = _Script(); s.fields.update(self.OFF_KILLZONE)
        r = SequenceRunner({**CONFIG, "ignore_kill_zone": True}, hooks=_hooks(s))
        sig = r.on_bar(_bar(0), {})
        nm = r.last_near_miss
        assert sig is not None or (nm is not None and "kill-zone" not in nm["reason"])


class TestMomentumGate:
    """ctx.extra['momentum_ok'] gates the emit: a completed setup with momentum_ok False
    is blocked (near-miss 'momentum not confirmed'); True or absent passes."""

    FULL = {
        "htf_bias": "long", "direction": "long", "structure_15m": "long",
        "price_zone": "discount", "sweep": {"level": 2640}, "sweep_confirmed": True,
        "fvg_valid": True, "fvg_fresh": True, "fvg": {"top": 2648, "bottom": 2644},
        "retraced_to_zone": True, "micro_choch": True, "confirmation_candle": True,
        "in_kill_zone": True, "news_clear": True, "no_blocking_filters": True,
        "daily_limits_ok": True,
    }

    def test_momentum_false_blocks(self):
        s = _Script(); s.fields.update(self.FULL); s.fields["extra"] = {"momentum_ok": False}
        r = _runner(s)
        assert r.on_bar(_bar(0), {}) is None
        nm = r.last_near_miss
        assert nm is not None and "momentum" in nm["reason"]

    def test_momentum_true_passes(self):
        s = _Script(); s.fields.update(self.FULL); s.fields["extra"] = {"momentum_ok": True}
        r = _runner(s)
        assert r.on_bar(_bar(0), {}) is not None

    def test_momentum_absent_passes(self):
        s = _Script(); s.fields.update(self.FULL)  # no extra → momentum_ok absent → True
        r = _runner(s)
        assert r.on_bar(_bar(0), {}) is not None


class TestPriceSanityGate:
    """Don't fire a trade whose SL is ALREADY breached by the current closed price —
    the captured FVG was blown through, so it's a dead-on-arrival loss (the 2026-06-16
    05:35 long: entry 4324.87 / SL 4321.90 while price had already closed 4319.53)."""

    SETUP = {  # full long setup that would otherwise be approved
        "htf_bias": "long", "direction": "long", "structure_15m": "long",
        "price_zone": "discount", "sweep": {"level": 2640}, "sweep_confirmed": True,
        "fvg_valid": True, "fvg_fresh": True, "fvg": {"top": 2648, "bottom": 2644},
        "retraced_to_zone": True, "micro_choch": True, "confirmation_candle": True,
        "in_kill_zone": True, "news_clear": True, "no_blocking_filters": True,
        "daily_limits_ok": True,
    }

    def _hist(self, last_close):
        import pandas as pd
        idx = pd.date_range("2026-06-16", periods=5, freq="15min", tz="UTC")
        return {"15m": pd.DataFrame({"open": [last_close] * 5, "high": [last_close + 2] * 5,
                                     "low": [last_close - 2] * 5, "close": [last_close] * 5}, index=idx)}

    def test_doa_long_skipped(self):  # risk hook sets entry 2650 / SL 2640
        s = _Script(); s.fields.update(self.SETUP)
        r = SequenceRunner({**CONFIG, "price_sanity_gate": True}, hooks=_hooks(s))
        sig = r.on_bar(_bar(0), self._hist(2630.0))   # price 2630 < SL 2640 → DOA
        assert sig is None
        assert r.last_near_miss and "DOA" in r.last_near_miss["reason"]

    def test_normal_long_fires(self):
        s = _Script(); s.fields.update(self.SETUP)
        r = SequenceRunner({**CONFIG, "price_sanity_gate": True}, hooks=_hooks(s))
        assert r.on_bar(_bar(0), self._hist(2655.0)) is not None   # price 2655 > SL → ok

    def test_gate_off_still_fires_doa(self):
        s = _Script(); s.fields.update(self.SETUP)
        r = SequenceRunner({**CONFIG, "price_sanity_gate": False}, hooks=_hooks(s))
        assert r.on_bar(_bar(0), self._hist(2630.0)) is not None   # gate off → legacy
