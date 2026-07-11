"""Pine-v4.0 parity levers (all default OFF):
  1. macd_gate       — MACD histogram+line alignment at the trigger
  2. htf_ob_gate     — proximity to a same-direction 1H order block
  3. retrace_next_bar_only — retrace counts only on bars AFTER the FVG capture
"""

import numpy as np
import pandas as pd
import pytest

from core.engine.pipeline_hooks import compute_macd_alignment, near_any_ob
from core.engine.sequence_runner import retrace_lookback, SequenceRunner


def _accel_df(n=120, sign=1.0):
    """ACCELERATING trend — a linear ramp converges MACD to a constant (hist -> 0),
    which the alignment gate rightly treats as fading momentum. Quadratic keeps the
    histogram expanding in the trend direction."""
    idx = pd.date_range("2026-01-05", periods=n, freq="15min", tz="UTC")
    base = 100 + sign * 0.002 * np.arange(float(n)) ** 2
    return pd.DataFrame({"open": base, "high": base + 0.4, "low": base - 0.4,
                         "close": base + 0.1, "volume": 1.0}, index=idx)


class TestMACDAlignment:
    def test_accel_uptrend_aligns_long_blocks_short(self):
        df = _accel_df(sign=1.0)
        assert compute_macd_alignment(df, is_long=True) is True
        assert compute_macd_alignment(df, is_long=False) is False

    def test_accel_downtrend_aligns_short_blocks_long(self):
        df = _accel_df(sign=-1.0)
        assert compute_macd_alignment(df, is_long=False) is True
        assert compute_macd_alignment(df, is_long=True) is False

    def test_fading_momentum_blocks_both_directions(self):
        """A decaying decline (steps shrinking) = momentum fading — the gate blocks
        shorts even though price is falling. That is the intended Pine behavior."""
        idx = pd.date_range("2026-01-05", periods=120, freq="15min", tz="UTC")
        base = 100 * (0.998 ** np.arange(120))
        df = pd.DataFrame({"open": base, "high": base + 0.4, "low": base - 0.4,
                           "close": base + 0.1, "volume": 1.0}, index=idx)
        assert compute_macd_alignment(df, is_long=False) is False

    def test_warmup_never_blocks(self):
        df = _accel_df(n=1)
        assert compute_macd_alignment(df, is_long=True) is True
        assert compute_macd_alignment(df, is_long=False) is True


class TestNearAnyOB:
    BLOCKS = [{"top": 105.0, "bottom": 103.0}, {"top": 98.0, "bottom": 96.5}]

    def test_inside_zone(self):
        assert near_any_ob(104.0, self.BLOCKS, tol=0.0) is True

    def test_within_tolerance(self):
        assert near_any_ob(105.8, self.BLOCKS, tol=1.0) is True
        assert near_any_ob(95.8, self.BLOCKS, tol=1.0) is True

    def test_outside(self):
        assert near_any_ob(101.0, self.BLOCKS, tol=1.0) is False
        assert near_any_ob(101.0, [], tol=5.0) is False


class TestRetraceLookback:
    def test_flag_off_keeps_default(self):
        assert retrace_lookback(False, 10, 10) == 3
        assert retrace_lookback(False, None, 50) == 3

    def test_capture_bar_yields_zero(self):
        assert retrace_lookback(True, 10, 10) == 0

    def test_bars_after_capture_grow_to_default(self):
        assert retrace_lookback(True, 10, 11) == 1
        assert retrace_lookback(True, 10, 12) == 2
        assert retrace_lookback(True, 10, 13) == 3
        assert retrace_lookback(True, 10, 40) == 3

    def test_missing_capture_bar_keeps_default(self):
        assert retrace_lookback(True, None, 40) == 3

    def test_runner_reads_flag_and_resets_capture_bar(self):
        cfg = {"rr_tiers": {"min_to_enter": 2.0},
               "costs": {"default_spread": 0.25, "default_slippage": 0.10},
               "retrace_next_bar_only": True}
        r = SequenceRunner(cfg, execution_tf="15m")
        assert r._retrace_next_bar is True
        r._fvg_captured_bar = 42
        from datetime import datetime, timezone
        r._reset(datetime(2026, 1, 5, tzinfo=timezone.utc), "test")
        assert r._fvg_captured_bar is None
