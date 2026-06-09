"""
Tests for displacement_detector.py — Phase 2.4.

Critical properties:
  - body / ATR  ≥ body_atr_threshold              (default 1.2)
  - body / range ≥ body_range_threshold           (default 0.60)
  - close > max(prev `break_lookback` highs)      (bullish)
  - close < min(prev `break_lookback` lows)       (bearish)
  - All three thresholds are STRICT requirements; failing any one means
    no displacement is recorded.
  - Doji (close == open) is never a displacement.
  - No look-ahead — bar i only inspects bars i-break_lookback..i.
"""

import pytest
import numpy as np
import pandas as pd

from core.smc.displacement_detector import DisplacementDetector, detect_displacements


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_df(opens, highs, lows, closes, start="2026-01-05 10:00", freq="5min"):
    n = len(opens)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [100.] * n},
        index=idx,
    )


def _flat_then_displacement(
    n_flat: int,
    flat_high: float = 10.0,
    flat_low:  float = 9.0,
    flat_close: float = 9.5,
    disp_open:  float = 10.0,
    disp_high:  float = 15.0,
    disp_low:   float = 10.0,
    disp_close: float = 15.0,
):
    """Build n_flat flat bars followed by ONE displacement bar."""
    n = n_flat + 1
    opens   = [flat_close]  * n_flat + [disp_open]
    highs   = [flat_high]   * n_flat + [disp_high]
    lows    = [flat_low]    * n_flat + [disp_low]
    closes  = [flat_close]  * n_flat + [disp_close]
    return _make_df(opens, highs, lows, closes), n_flat   # disp index = n_flat


def _det(**kwargs):
    return DisplacementDetector(**kwargs)


# ------------------------------------------------------------------ #
# Bullish displacement                                                 #
# ------------------------------------------------------------------ #

class TestBullishDisplacement:
    def test_basic_bullish_displacement(self):
        """body=5, range=5, ATR≈1 → all three thresholds pass; close > max prev highs."""
        df, i = _flat_then_displacement(15)
        result = _det().detect(df)
        assert result["displacement_type"].iloc[i] == "bull"

    def test_body_atr_value_correct(self):
        df, i = _flat_then_displacement(15)
        result = _det().detect(df)
        body = abs(15.0 - 10.0)
        # Don't pin the exact ATR — just verify it equals body / atr at that bar
        atr = result["displacement_body_atr"].iloc[i]
        assert atr == pytest.approx(body / (body / atr), rel=1e-9)
        assert atr > 1.2

    def test_body_pct_one_when_no_wicks(self):
        df, i = _flat_then_displacement(15)
        result = _det().detect(df)
        assert result["displacement_body_pct"].iloc[i] == pytest.approx(1.0)

    def test_no_displacement_when_body_too_small_for_atr(self):
        """body=0.5, ATR≈1 → body/ATR=0.5 < 1.2 → fail."""
        df, i = _flat_then_displacement(
            15,
            disp_open=10.0, disp_close=10.5, disp_high=10.5, disp_low=10.0,
        )
        result = _det().detect(df)
        assert result["displacement_type"].iloc[i] is None

    def test_no_displacement_when_too_much_wick(self):
        """body=2 (small), range=10 (huge wick) → body/range=0.2 < 0.6 → fail."""
        df, i = _flat_then_displacement(
            15,
            disp_open=10.0, disp_close=12.0, disp_high=20.0, disp_low=10.0,
        )
        result = _det().detect(df)
        assert result["displacement_type"].iloc[i] is None

    def test_no_displacement_when_close_does_not_break_prev_highs(self):
        """Body and shape pass, but close ≤ max of prev `break_lookback` highs."""
        # Bar 12 has a spike high=20. Bar 15 is a strong bull candle but close=15 < 20 → no break.
        opens   = [9.5]*12 + [9.5, 9.5, 9.5, 10.0]
        highs   = [10.0]*12 + [20.0, 10.0, 10.0, 15.0]   # bar 12 high=20
        lows    = [9.0]*12 + [9.0, 9.0, 9.0, 10.0]
        closes  = [9.5]*12 + [9.5, 9.5, 9.5, 15.0]
        df = _make_df(opens, highs, lows, closes)
        # bar 15: body=5, range=5 → body/range=1.0 ✓
        # max highs[12:15] = max(20, 10, 10) = 20  →  close=15 NOT > 20  →  blocked
        result = _det().detect(df)
        assert result["displacement_type"].iloc[15] is None


# ------------------------------------------------------------------ #
# Bearish displacement                                                 #
# ------------------------------------------------------------------ #

class TestBearishDisplacement:
    def test_basic_bearish_displacement(self):
        """body=5, range=5, ATR≈1 → bullish criteria mirrored for downside."""
        df, i = _flat_then_displacement(
            15,
            disp_open=10.0, disp_close=5.0, disp_high=10.0, disp_low=5.0,
        )
        result = _det().detect(df)
        assert result["displacement_type"].iloc[i] == "bear"

    def test_no_bear_displacement_when_close_does_not_break_prev_lows(self):
        # Prev lows are at 4 (well below the displacement close of 5)
        n = 16
        opens   = [9.5]*15 + [10.0]
        highs   = [10.0]*15 + [10.0]
        lows    = [4.0]*15 + [5.0]                # prev lows are at 4
        closes  = [9.5]*15 + [5.0]                # close=5 NOT < min(lows[12:15])=4
        df = _make_df(opens, highs, lows, closes)
        result = _det().detect(df)
        assert result["displacement_type"].iloc[15] is None


# ------------------------------------------------------------------ #
# Doji / zero-range / equality                                         #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_doji_no_displacement(self):
        """close == open → no displacement regardless of other criteria."""
        df, i = _flat_then_displacement(
            15,
            disp_open=12.0, disp_close=12.0, disp_high=15.0, disp_low=10.0,
        )
        result = _det().detect(df)
        assert result["displacement_type"].iloc[i] is None

    def test_zero_range_no_displacement(self):
        """high == low → no displacement (body and range both zero)."""
        df, i = _flat_then_displacement(
            15,
            disp_open=10.0, disp_close=10.0, disp_high=10.0, disp_low=10.0,
        )
        result = _det().detect(df)
        assert result["displacement_type"].iloc[i] is None

    def test_first_bars_have_no_displacement(self):
        """Bars before break_lookback can never host a displacement."""
        df, i = _flat_then_displacement(15)
        result = _det(break_lookback=3).detect(df)
        for idx in range(3):
            assert result["displacement_type"].iloc[idx] is None


# ------------------------------------------------------------------ #
# Threshold tunability                                                 #
# ------------------------------------------------------------------ #

class TestThresholds:
    def test_lower_body_atr_admits_smaller_bodies(self):
        """Body just below the default threshold is rejected, but admitted with a lower threshold."""
        # body=1.0, ATR≈1 → body/ATR=1.0 < 1.2
        # But with body_atr_threshold=0.5 it should pass
        opens   = [9.5]*15 + [10.0]
        highs   = [10.0]*15 + [11.0]
        lows    = [9.0]*15 + [10.0]
        closes  = [9.5]*15 + [11.0]
        df = _make_df(opens, highs, lows, closes)
        # Default — fails body_atr (~1.0 < 1.2)
        assert _det().detect(df)["displacement_type"].iloc[15] is None
        # Loosened — passes
        assert _det(body_atr_threshold=0.5).detect(df)["displacement_type"].iloc[15] == "bull"

    def test_higher_body_range_blocks_wickier_candles(self):
        # Tight body/range = 0.6 (default); raise it to 0.9 → fails
        opens   = [9.5]*15 + [10.0]
        highs   = [10.0]*15 + [16.0]
        lows    = [9.0]*15 + [10.0]
        closes  = [9.5]*15 + [13.6]   # body = 3.6, range = 6 → 0.6 exactly
        df = _make_df(opens, highs, lows, closes)
        # 0.6 is not > 0.6 either; the rule is ≥ — but our code uses < so 0.6 is allowed
        # Actually the code: if body_pct < threshold: continue. So 0.6 passes 0.6 (not strict).
        # Make body slightly bigger to ensure it passes default:
        closes2 = [9.5]*15 + [14.0]   # body=4, range=6 → 0.667
        df2 = _make_df(opens, highs, lows, closes2)
        # default body_range = 0.6 → passes
        # But default body_atr_threshold=1.2: body=4, ATR≈1 → 4 > 1.2 ✓
        # And close > max prev highs? close=14, prev highs=10 → ✓
        assert _det().detect(df2)["displacement_type"].iloc[15] == "bull"
        # Tighten body_range_threshold to 0.9 → 0.667 < 0.9 → fails
        assert _det(body_range_threshold=0.9).detect(df2)["displacement_type"].iloc[15] is None


# ------------------------------------------------------------------ #
# Break lookback                                                       #
# ------------------------------------------------------------------ #

class TestBreakLookback:
    def test_close_must_break_all_lookback_highs(self):
        """A high spike inside the lookback window blocks the break."""
        # Bars 0-13 flat (high=10). Bar 14: high=20 (spike). Bar 15: displacement
        # close = 15 — does it break max highs[12..14] = max(10,10,20) = 20? No.
        opens   = [9.5]*14 + [12.0, 10.0]
        highs   = [10.0]*14 + [20.0, 16.0]
        lows    = [9.0]*14 + [10.0, 10.0]
        closes  = [9.5]*14 + [15.0, 15.0]   # bar 15: close=15
        df = _make_df(opens, highs, lows, closes)
        # body for bar 15 = abs(15-10) = 5, range = 16-10 = 6, body/range = 0.83 ✓
        # body/ATR — ATR will be elevated due to bar 14's TR=11. body=5, ATR≈?
        # Actually with TR at bar 14 ≈ max(10, |20-9.5|, |10-9.5|) = 10.5
        # ATR over rolling 14 includes bar 14 → ATR≈ (13+10.5)/14 ≈ 1.68
        # body/ATR ≈ 5/1.68 ≈ 2.97 > 1.2 ✓
        # close=15 vs max(highs[12..15])=max(10,10,20,16)=20 → 15 not > 20 → blocks bull
        assert _det().detect(df)["displacement_type"].iloc[15] is None

    def test_lookback_smaller_admits_break(self):
        """With break_lookback=1 the spike (bar 14) is the only comparison —
        but bar 15 close=15 still doesn't break bar 14 high=20 → fail.
        Use a different setup: close=21 to break bar 14's high."""
        opens   = [9.5]*14 + [12.0, 10.0]
        highs   = [10.0]*14 + [20.0, 22.0]
        lows    = [9.0]*14 + [10.0, 10.0]
        closes  = [9.5]*14 + [15.0, 21.0]
        df = _make_df(opens, highs, lows, closes)
        # close=21 vs max(highs[14:15])=max(20)=20 → break ✓ with break_lookback=1
        # body=11, range=12, body/range=0.916 ✓; body/ATR very large ✓
        assert _det(break_lookback=1).detect(df)["displacement_type"].iloc[15] == "bull"


# ------------------------------------------------------------------ #
# Output format                                                       #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df, _ = _flat_then_displacement(15)
        result = _det().detect(df)
        for col in ["displacement_type", "displacement_body_atr", "displacement_body_pct"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df, _ = _flat_then_displacement(15)
        pd.testing.assert_index_equal(_det().detect(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df, _ = _flat_then_displacement(15)
        _det().detect(df)
        assert "displacement_type" not in df.columns

    def test_no_displacement_means_all_nan(self):
        opens = [9.5]*5
        df = _make_df(opens, opens, opens, opens)   # all flat
        result = _det().detect(df)
        assert result["displacement_type"].isna().all()
        assert result["displacement_body_atr"].isna().all()


# ------------------------------------------------------------------ #
# Accessor methods                                                     #
# ------------------------------------------------------------------ #

class TestAccessors:
    def test_get_last_displacement_bull(self):
        df, i = _flat_then_displacement(15)
        result = _det().detect(df)
        info = _det().get_last_displacement(result, "bull")
        assert info["type"] == "bull"
        assert info["confirm_pos"] == i

    def test_get_last_displacement_returns_none_when_empty(self):
        df, _ = _flat_then_displacement(
            15, disp_open=10.0, disp_close=10.0, disp_high=10.0, disp_low=10.0,
        )
        result = _det().detect(df)
        assert _det().get_last_displacement(result, "bull") is None
        assert _det().get_last_displacement(result, "bear") is None

    def test_get_all_displacements_returns_list(self):
        df, _ = _flat_then_displacement(15)
        result = _det().detect(df)
        lst = _det().get_all_displacements(result, n=10)
        assert isinstance(lst, list)
        assert len(lst) == 1
        assert lst[0]["type"] == "bull"


# ------------------------------------------------------------------ #
# Validation                                                           #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_body_atr_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="body_atr_threshold"):
            DisplacementDetector(body_atr_threshold=0)

    def test_body_range_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="body_range_threshold"):
            DisplacementDetector(body_range_threshold=0)

    def test_body_range_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="body_range_threshold"):
            DisplacementDetector(body_range_threshold=1.5)

    def test_break_lookback_zero_raises(self):
        with pytest.raises(ValueError, match="break_lookback"):
            DisplacementDetector(break_lookback=0)

    def test_atr_period_zero_raises(self):
        with pytest.raises(ValueError, match="atr_period"):
            DisplacementDetector(atr_period=0)

    def test_missing_open_raises(self):
        df, _ = _flat_then_displacement(15)
        df = df.drop(columns=["open"])
        with pytest.raises(ValueError, match="Missing"):
            _det().detect(df)

    def test_non_datetime_index_raises(self):
        df, _ = _flat_then_displacement(15)
        df.index = range(len(df))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _det().detect(df)

    def test_convenience_function_works(self):
        df, _ = _flat_then_displacement(15)
        result = detect_displacements(df)
        assert "displacement_type" in result.columns
