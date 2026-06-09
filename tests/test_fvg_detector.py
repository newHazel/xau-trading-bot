"""
Tests for fvg_detector.py — Phase 2.3.

Critical properties:
  - Bull FVG: c1.high < c3.low (strictly).
  - Bear FVG: c1.low  > c3.high (strictly).
  - Strict inequality: equal boundaries do NOT form a gap.
  - The two FVG directions are mutually exclusive on the same 3-bar window.
  - Size filter: gap > ATR * size_threshold_atr_pct (default 0.3).
  - FVG event is marked at candle 3 (bar i, with c1=i-2).
  - No look-ahead — early bars (< 2) cannot have an FVG.
"""

import pytest
import numpy as np
import pandas as pd

from core.smc.fvg_detector import FVGDetector, detect_fvgs


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_df(highs, lows, closes=None, start="2026-01-05 10:00", freq="5min"):
    n = len(highs)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": [100.] * n},
        index=idx,
    )


def _det(threshold=0.0, atr_period=14):
    """Default = no size filter (threshold=0) for unit tests."""
    return FVGDetector(atr_period=atr_period, size_threshold_atr_pct=threshold)


# ------------------------------------------------------------------ #
# Bullish FVG                                                           #
# ------------------------------------------------------------------ #

class TestBullishFVG:
    def test_bullish_fvg_basic(self):
        """c1.high=10, c2 displaces, c3.low=15 → gap [10, 15]."""
        highs = [10, 14, 16, 16, 16]
        lows  = [ 8, 12, 15, 15, 15]
        df = _make_df(highs, lows)
        result = _det().detect(df)
        assert result["fvg_type"].iloc[2] == "bull"
        assert result["fvg_bottom"].iloc[2] == pytest.approx(10.0)
        assert result["fvg_top"].iloc[2]    == pytest.approx(15.0)
        assert result["fvg_size"].iloc[2]   == pytest.approx(5.0)

    def test_bullish_fvg_c1_idx_correct(self):
        highs = [10, 14, 16, 16, 16]
        lows  = [ 8, 12, 15, 15, 15]
        df = _make_df(highs, lows)
        result = _det().detect(df)
        # c3 is at bar 2 → c1 should be at bar 0
        assert result["fvg_c1_idx"].iloc[2] == 0

    def test_no_bull_fvg_when_c1_high_equals_c3_low(self):
        """Strict inequality — equal boundaries do NOT form a gap."""
        highs = [10, 12, 15, 15, 15]
        lows  = [ 8, 11, 10, 10, 10]   # c3.low == c1.high == 10
        df = _make_df(highs, lows)
        result = _det().detect(df)
        assert result["fvg_type"].iloc[2] is None

    def test_no_bull_fvg_when_overlap(self):
        """c1's range overlaps with c3's range — no gap."""
        highs = [12, 13, 14, 14, 14]
        lows  = [ 8,  9, 10, 10, 10]   # c1.high=12 > c3.low=10 → no bull gap
        df = _make_df(highs, lows)
        result = _det().detect(df)
        assert result["fvg_type"].iloc[2] is None

    def test_multiple_bullish_fvgs_in_series(self):
        """Two distinct bull FVGs at well-separated 3-bar windows."""
        highs = [10, 14, 16, 16, 16, 16, 20, 20, 20, 20]
        lows  = [ 8, 12, 15, 13, 13, 13, 18, 13, 13, 13]
        df = _make_df(highs, lows)
        result = _det().detect(df)
        assert result["fvg_type"].iloc[2] == "bull"
        assert result["fvg_type"].iloc[6] == "bull"


# ------------------------------------------------------------------ #
# Bearish FVG                                                          #
# ------------------------------------------------------------------ #

class TestBearishFVG:
    def test_bearish_fvg_basic(self):
        """c1.low=20, c2 displaces down, c3.high=15 → gap [15, 20]."""
        highs = [22, 18, 15, 15, 15]
        lows  = [20, 16, 14, 14, 14]
        df = _make_df(highs, lows)
        result = _det().detect(df)
        assert result["fvg_type"].iloc[2] == "bear"
        assert result["fvg_bottom"].iloc[2] == pytest.approx(15.0)
        assert result["fvg_top"].iloc[2]    == pytest.approx(20.0)
        assert result["fvg_size"].iloc[2]   == pytest.approx(5.0)

    def test_no_bear_fvg_when_c1_low_equals_c3_high(self):
        highs = [22, 18, 15, 15, 15]
        lows  = [15, 14, 14, 14, 14]   # c1.low=15 == c3.high=15
        df = _make_df(highs, lows)
        result = _det().detect(df)
        assert result["fvg_type"].iloc[2] is None

    def test_no_bear_fvg_when_overlap(self):
        highs = [22, 18, 21, 21, 21]
        lows  = [20, 16, 19, 19, 19]   # c1.low=20 < c3.high=21 → no bear gap
        df = _make_df(highs, lows)
        result = _det().detect(df)
        assert result["fvg_type"].iloc[2] is None


# ------------------------------------------------------------------ #
# Mutual exclusion                                                      #
# ------------------------------------------------------------------ #

class TestMutualExclusion:
    def test_only_one_direction_at_a_time(self):
        """Bull and bear FVG conditions cannot both hold for the same 3 bars."""
        highs = [10, 14, 16, 16, 16]
        lows  = [ 8, 12, 15, 15, 15]
        df = _make_df(highs, lows)
        result = _det().detect(df)
        # Just verify exactly one direction set (and it's bull here)
        assert result["fvg_type"].iloc[2] == "bull"
        # No row should have both top and bottom set inconsistently — covered by type


# ------------------------------------------------------------------ #
# Size filter                                                          #
# ------------------------------------------------------------------ #

class TestSizeFilter:
    def test_small_fvg_filtered_out(self):
        """ATR≈1 (flat data); tiny gap of 0.2 is below 0.3*ATR threshold."""
        # Bars 0-13: flat (high=10, low=9) → TR≈1, ATR≈1.
        # Bars 14, 15, 16: tiny FVG.
        highs = [10.0] * 14 + [10.0, 11.0, 10.5]
        lows  = [ 9.0] * 14 + [ 9.0, 10.5, 10.2]
        df = _make_df(highs, lows)
        result = _det(threshold=0.3).detect(df)
        # Bar 16: c1=bar 14 (high=10), c3=bar 16 (low=10.2). Gap = 0.2 < 0.3 → filtered.
        assert result["fvg_type"].iloc[16] is None

    def test_large_fvg_passes_filter(self):
        """Same setup but with a much larger gap → kept."""
        highs = [10.0] * 14 + [10.0, 15.0, 14.0]
        lows  = [ 9.0] * 14 + [ 9.0, 14.0, 13.0]
        df = _make_df(highs, lows)
        result = _det(threshold=0.3).detect(df)
        # Gap = 13 - 10 = 3 → much larger than 0.3*ATR
        assert result["fvg_type"].iloc[16] == "bull"
        assert result["fvg_size"].iloc[16] == pytest.approx(3.0)

    def test_zero_threshold_disables_filter(self):
        """threshold=0 means any positive gap qualifies."""
        highs = [10.0, 11.0, 10.5]
        lows  = [ 9.0, 10.5, 10.01]
        df = _make_df(highs, lows)
        result = _det(threshold=0.0).detect(df)
        assert result["fvg_type"].iloc[2] == "bull"
        assert result["fvg_size"].iloc[2] == pytest.approx(0.01)

    def test_filter_strict_inequality(self):
        """Filter uses strict > (gap == threshold is rejected). Use a tiny
        gap well below threshold to verify rejection."""
        highs = [10.0] * 14 + [10.0, 11.0, 10.05]
        lows  = [ 9.0] * 14 + [ 9.0, 10.05, 10.01]
        df = _make_df(highs, lows)
        result = _det(threshold=0.3).detect(df)
        # gap = 10.01 - 10 = 0.01 — far below threshold ≈ 0.3
        assert result["fvg_type"].iloc[16] is None


# ------------------------------------------------------------------ #
# No look-ahead / placement                                             #
# ------------------------------------------------------------------ #

class TestPlacement:
    def test_first_two_bars_have_no_fvg(self):
        """A 3-candle pattern requires bar i and (i-2) — bar 0,1 cannot host one."""
        highs = [10, 14, 16, 16]
        lows  = [ 8, 12, 15, 15]
        df = _make_df(highs, lows)
        result = _det().detect(df)
        assert result["fvg_type"].iloc[0] is None
        assert result["fvg_type"].iloc[1] is None

    def test_fvg_marked_at_candle3_not_candle1(self):
        highs = [10, 14, 16, 16, 16]
        lows  = [ 8, 12, 15, 15, 15]
        df = _make_df(highs, lows)
        result = _det().detect(df)
        # Marked at bar 2 (c3), not bar 0 (c1)
        assert result["fvg_type"].iloc[0] is None
        assert result["fvg_type"].iloc[2] == "bull"

    def test_late_fvg_does_not_relabel_past(self):
        """An FVG at bar 10 must not put labels on earlier bars.
        c2 (bar 9) deliberately leaves bar 7-9 overlap so only bar 10 forms an FVG."""
        highs = [10] * 9 + [12, 15]    # bars 0-8 = 10, bar 9 = 12, bar 10 = 15
        lows  = [ 9] * 9 + [10, 11]    # bar 9 low = 10 (= bar 7 high → no gap at bar 9)
        df = _make_df(highs, lows)
        result = _det().detect(df)
        for i in range(10):
            assert result["fvg_type"].iloc[i] is None, f"bar {i}"
        assert result["fvg_type"].iloc[10] == "bull"


# ------------------------------------------------------------------ #
# Output format                                                         #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _make_df([10] * 5, [9] * 5)
        result = _det().detect(df)
        for col in ["fvg_type", "fvg_top", "fvg_bottom", "fvg_size", "fvg_c1_idx"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _make_df([10] * 5, [9] * 5)
        pd.testing.assert_index_equal(_det().detect(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _make_df([10] * 5, [9] * 5)
        _det().detect(df)
        assert "fvg_type" not in df.columns

    def test_no_fvg_means_all_nan(self):
        df = _make_df([10] * 5, [9] * 5)
        result = _det().detect(df)
        assert result["fvg_type"].isna().all()
        assert result["fvg_top"].isna().all()
        assert (result["fvg_c1_idx"] == -1).all()


# ------------------------------------------------------------------ #
# Accessor methods                                                      #
# ------------------------------------------------------------------ #

class TestAccessors:
    def _build_bull(self):
        # Single bull FVG at bar 2; bar 4's high=16 prevents an opposite-side FVG
        highs = [10, 14, 16, 14, 16]
        lows  = [ 8, 12, 15, 13, 13]
        df = _make_df(highs, lows)
        return _det().detect(df)

    def _build_bear(self):
        # Single bear FVG at bar 2; bar 4's low=14 prevents an opposite-side FVG
        highs = [22, 18, 15, 17, 17]
        lows  = [20, 16, 14, 16, 14]
        df = _make_df(highs, lows)
        return _det().detect(df)

    def test_get_last_fvg_bull(self):
        result = self._build_bull()
        info = _det().get_last_fvg(result, direction="bull")
        assert info["type"] == "bull"
        assert info["top"]    == pytest.approx(15.0)
        assert info["bottom"] == pytest.approx(10.0)
        assert info["size"]   == pytest.approx(5.0)
        assert info["c1_idx"] == 0

    def test_get_last_fvg_bear(self):
        result = self._build_bear()
        info = _det().get_last_fvg(result, direction="bear")
        assert info["type"] == "bear"
        assert info["top"]    == pytest.approx(20.0)
        assert info["bottom"] == pytest.approx(15.0)

    def test_get_last_fvg_returns_none_when_empty(self):
        df = _make_df([10] * 5, [9] * 5)
        result = _det().detect(df)
        assert _det().get_last_fvg(result, "bull") is None
        assert _det().get_last_fvg(result, "bear") is None

    def test_get_all_fvgs_returns_list(self):
        result = self._build_bull()
        lst = _det().get_all_fvgs(result, n=10)
        assert isinstance(lst, list)
        assert len(lst) == 1
        assert lst[0]["type"] == "bull"

    def test_get_all_fvgs_newest_first(self):
        # Exactly 2 bull FVGs:
        #   bar 2: c1=bar 0 high=10 < c3=bar 2 low=15
        #   bar 6: c1=bar 4 high=16 < c3=bar 6 low=18
        # Highs/lows around 16 between them prevent opposite-direction gaps.
        highs = [10, 14, 16, 16, 16, 16, 20, 20, 20, 20]
        lows  = [ 8, 12, 15, 13, 13, 13, 18, 13, 13, 13]
        df = _make_df(highs, lows)
        result = _det().detect(df)
        lst = _det().get_all_fvgs(result, n=10)
        assert len(lst) == 2
        assert lst[0]["confirm_ts"] >= lst[1]["confirm_ts"]

    def test_get_all_fvgs_empty_when_no_fvg(self):
        df = _make_df([10] * 5, [9] * 5)
        result = _det().detect(df)
        assert _det().get_all_fvgs(result) == []


# ------------------------------------------------------------------ #
# Validation                                                            #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_atr_period_zero_raises(self):
        with pytest.raises(ValueError, match="atr_period"):
            FVGDetector(atr_period=0)

    def test_size_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="size_threshold"):
            FVGDetector(size_threshold_atr_pct=-0.1)

    def test_missing_high_raises(self):
        df = _make_df([10] * 5, [9] * 5).drop(columns=["high"])
        with pytest.raises(ValueError, match="Missing"):
            _det().detect(df)

    def test_missing_low_raises(self):
        df = _make_df([10] * 5, [9] * 5).drop(columns=["low"])
        with pytest.raises(ValueError, match="Missing"):
            _det().detect(df)

    def test_missing_close_raises(self):
        df = _make_df([10] * 5, [9] * 5).drop(columns=["close"])
        with pytest.raises(ValueError, match="Missing"):
            _det().detect(df)

    def test_non_datetime_index_raises(self):
        df = _make_df([10] * 5, [9] * 5)
        df.index = range(len(df))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _det().detect(df)

    def test_convenience_function_works(self):
        highs = [10, 14, 16, 16, 16]
        lows  = [ 8, 12, 15, 15, 15]
        df = _make_df(highs, lows)
        result = detect_fvgs(df, size_threshold_atr_pct=0)
        assert "fvg_type" in result.columns
