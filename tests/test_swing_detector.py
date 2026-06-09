"""
Tests for swing_detector.py — Phase 1.1.

Critical property: no look-ahead.
Every swing is confirmed only AFTER lag candles close.
A spike in the future must never affect past swing labels.
"""

import pytest
import numpy as np
import pandas as pd

from core.structure.swing_detector import SwingDetector, detect_swings, DEFAULT_FRACTAL_WINDOWS


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

def _make_df(
    highs: list,
    lows:  list,
    start: str = "2026-01-05 10:00",
    freq:  str = "5min",
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from explicit high/low lists."""
    assert len(highs) == len(lows)
    n   = len(highs)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    mid = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame(
        {
            "open":   mid,
            "high":   highs,
            "low":    lows,
            "close":  mid,
            "volume": [100.0] * n,
        },
        index=idx,
    )


def _det(windows: dict = None) -> SwingDetector:
    return SwingDetector(fractal_windows=windows or DEFAULT_FRACTAL_WINDOWS)


# ------------------------------------------------------------------ #
# Basic swing detection                                                #
# ------------------------------------------------------------------ #

class TestBasicSwingDetection:
    def test_single_peak_detected_as_swing_high(self):
        # window=5, lag=2. Peak at bar 4, confirmed at bar 6.
        highs = [10, 11, 12, 13, 20, 13, 12, 11, 10, 9]
        lows  = [ 8,  9, 10, 11, 18, 11, 10,  9,  8, 7]
        df    = _make_df(highs, lows)
        result = _det().detect(df, "5m")

        assert result["swing_high"].notna().sum() == 1
        swing_val = result["swing_high"].dropna().iloc[0]
        assert swing_val == pytest.approx(20.0)

    def test_single_trough_detected_as_swing_low(self):
        highs = [20, 19, 18, 17, 10, 17, 18, 19, 20, 21]
        lows  = [18, 17, 16, 15,  8, 15, 16, 17, 18, 19]
        df    = _make_df(highs, lows)
        result = _det().detect(df, "5m")

        assert result["swing_low"].notna().sum() == 1
        swing_val = result["swing_low"].dropna().iloc[0]
        assert swing_val == pytest.approx(8.0)

    def test_flat_series_has_no_swings(self):
        highs = [10.0] * 15
        lows  = [ 9.0] * 15
        df    = _make_df(highs, lows)
        result = _det().detect(df, "5m")
        assert result["swing_high"].notna().sum() == 0
        assert result["swing_low"].notna().sum()  == 0

    def test_multiple_swings_detected(self):
        # window=5, lag=2.
        # Peak  at bar  3 (high=20): window bars 1-5 = [11,12,20,12,11] → strict max ✓ confirmed at 5
        # Trough at bar 7 (low=1):   window bars 5-9 = [11,10,1,4,5]   → strict min ✓ confirmed at 9
        # Peak  at bar 12 (high=25): window bars 10-14 = [8,9,25,9,8]  → strict max ✓ confirmed at 14
        highs = [10, 11, 12, 20, 12, 11, 10,  4,  5,  8,  9, 10, 25, 10,  9,  8,  7]
        lows  = [ 8,  9, 10, 18, 10,  9,  8,  1,  3,  6,  7,  8, 23,  8,  7,  6,  5]
        df    = _make_df(highs, lows)
        result = _det().detect(df, "5m")
        assert result["swing_high"].notna().sum() >= 2
        assert result["swing_low"].notna().sum()  >= 1

    def test_swing_high_idx_points_to_correct_bar(self):
        # Peak at bar 4, window=5, confirmed at bar 6
        highs = [10, 11, 12, 13, 20, 13, 12, 11, 10, 9]
        lows  = [ 8,  9, 10, 11, 18, 11, 10,  9,  8, 7]
        df    = _make_df(highs, lows)
        result = _det().detect(df, "5m")

        confirm_row = result[result["swing_high"].notna()].iloc[0]
        swing_bar_idx = int(confirm_row["swing_high_idx"])
        assert df["high"].iloc[swing_bar_idx] == pytest.approx(20.0)


# ------------------------------------------------------------------ #
# No look-ahead — confirmation lag                                     #
# ------------------------------------------------------------------ #

class TestNoLookahead:
    """
    Core guarantee: a swing is only visible at the confirmation bar,
    not before. Future bars must not affect past labels.
    """

    def test_swing_marked_at_confirmation_bar_not_swing_bar(self):
        # window=5, lag=2. Swing bar=4, confirmation bar=6.
        highs = [10, 11, 12, 13, 20, 13, 12, 11, 10, 9]
        lows  = [ 8,  9, 10, 11, 18, 11, 10,  9,  8, 7]
        df    = _make_df(highs, lows)
        result = _det().detect(df, "5m")

        # Row 4 (the actual swing bar) must NOT have swing_high marked
        assert np.isnan(result["swing_high"].iloc[4])
        # Row 5 (one bar after swing) must NOT be marked
        assert np.isnan(result["swing_high"].iloc[5])
        # Row 6 (lag=2 bars after swing=4) MUST be marked
        assert not np.isnan(result["swing_high"].iloc[6])

    def test_window3_lag1_confirmed_one_bar_later(self):
        # window=3, lag=1. Swing bar=3, confirmed at bar=4.
        highs = [10, 11, 12, 20, 12, 11, 10, 9]
        lows  = [ 8,  9, 10, 18, 10,  9,  8, 7]
        df    = _make_df(highs, lows, freq="1min")
        result = _det().detect(df, "1m")

        assert np.isnan(result["swing_high"].iloc[3])     # swing bar — not yet confirmed
        assert not np.isnan(result["swing_high"].iloc[4]) # confirmed 1 bar later

    def test_no_swing_in_future_affects_past_labels(self):
        """
        Inject a massive spike at the very last bar.
        Bars before it must have no swing label.
        """
        highs = [10, 11, 10, 11, 10, 11, 10, 11, 10, 99_000]
        lows  = [ 8,  9,  8,  9,  8,  9,  8,  9,  8, 98_998]
        df    = _make_df(highs, lows)
        result = _det().detect(df, "5m")

        # No bar before index 7 (confirm window for spike) should see 99_000
        for i in range(7):
            assert np.isnan(result["swing_high"].iloc[i]) or result["swing_high"].iloc[i] < 99_000

    def test_bars_before_first_confirmation_have_no_swings(self):
        # window=5, lag=2 → first possible confirmation is bar index 4
        highs = [10, 20, 10, 15, 10, 15, 10]
        lows  = [ 8, 18,  8, 13,  8, 13,  8]
        df    = _make_df(highs, lows)
        result = _det().detect(df, "5m")

        # Bars 0-3 cannot have any confirmed swing (need lag=2 bars after swing bar)
        for i in range(4):
            assert np.isnan(result["swing_high"].iloc[i])
            assert np.isnan(result["swing_low"].iloc[i])

    def test_truncated_series_no_swing_at_end(self):
        """
        The very last bar (index n-1) can never be a confirmed swing
        because it has no bars after it to serve as confirmation candles.
        window=5, lag=2 → a swing at bar i needs bar i+2 to exist and close.
        """
        highs = [10, 11, 12, 20, 12, 11, 10]   # peak at bar 3, confirmed at bar 5
        lows  = [ 8,  9, 10, 18, 10,  9,  8]
        df    = _make_df(highs, lows)
        result = _det().detect(df, "5m")
        # bar 6 = last bar → must be NaN
        assert np.isnan(result["swing_high"].iloc[-1])
        assert np.isnan(result["swing_low"].iloc[-1])


# ------------------------------------------------------------------ #
# Timeframe-specific window sizes                                      #
# ------------------------------------------------------------------ #

class TestTimeframeWindows:
    def test_1m_uses_window3(self):
        # window=3, lag=1. Peak at bar 2, confirmed at bar 3.
        highs = [10, 11, 20, 11, 10, 9, 8, 7]
        lows  = [ 8,  9, 18,  9,  8, 7, 6, 5]
        df    = _make_df(highs, lows, freq="1min")
        result = _det().detect(df, "1m")
        assert not np.isnan(result["swing_high"].iloc[3])

    def test_15m_uses_window5(self):
        # Same as 5m — window=5, lag=2
        highs = [10, 11, 12, 13, 20, 13, 12, 11, 10, 9]
        lows  = [ 8,  9, 10, 11, 18, 11, 10,  9,  8, 7]
        df    = _make_df(highs, lows, freq="15min")
        result = _det().detect(df, "15m")
        assert result["swing_high"].notna().sum() == 1
        assert not np.isnan(result["swing_high"].iloc[6])

    def test_4h_uses_window3(self):
        highs = [10, 11, 20, 11, 10, 9, 8, 7]
        lows  = [ 8,  9, 18,  9,  8, 7, 6, 5]
        df    = _make_df(highs, lows, freq="4h")
        result = _det().detect(df, "4h")
        assert not np.isnan(result["swing_high"].iloc[3])


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_swing_columns(self):
        df = _make_df([10]*10, [8]*10)
        result = _det().detect(df, "5m")
        for col in ["swing_high", "swing_low", "swing_high_idx", "swing_low_idx"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _make_df([10]*10, [8]*10)
        result = _det().detect(df, "5m")
        pd.testing.assert_index_equal(result.index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _make_df([10]*10, [8]*10)
        result = _det().detect(df, "5m")
        assert "swing_high" not in df.columns

    def test_swing_high_idx_is_neg1_when_no_swing(self):
        df = _make_df([10]*10, [8]*10)
        result = _det().detect(df, "5m")
        assert (result["swing_high_idx"] == -1).all()


# ------------------------------------------------------------------ #
# Accessor methods                                                     #
# ------------------------------------------------------------------ #

class TestAccessors:
    def _result(self) -> pd.DataFrame:
        highs = [10, 11, 12, 13, 20, 13, 12, 11, 10, 9,
                 10, 11, 12, 13, 25, 13, 12, 11, 10, 9]
        lows  = [ 8,  9, 10, 11, 18, 11, 10,  9,  8, 7,
                   8,  9, 10, 11, 23, 11, 10,  9,  8, 7]
        df = _make_df(highs, lows)
        return _det().detect(df, "5m")

    def test_get_last_swing_high(self):
        result = self._result()
        val = _det().get_last_swing_high(result)
        assert val == pytest.approx(25.0)   # last swing high is 25

    def test_get_last_swing_low(self):
        highs = [20, 19, 18, 17, 10, 17, 18, 19, 20, 21,
                 20, 19, 18, 17,  5, 17, 18, 19, 20, 21]
        lows  = [18, 17, 16, 15,  8, 15, 16, 17, 18, 19,
                 18, 17, 16, 15,  3, 15, 16, 17, 18, 19]
        df = _make_df(highs, lows)
        result = _det().detect(df, "5m")
        val = _det().get_last_swing_low(result)
        assert val == pytest.approx(3.0)

    def test_get_last_swing_high_before_idx(self):
        result = self._result()
        # before_idx=8 means we only see the first swing (20), not the second (25)
        val = _det().get_last_swing_high(result, before_idx=8)
        assert val == pytest.approx(20.0)

    def test_get_last_swing_high_none_when_empty(self):
        df = _make_df([10]*5, [8]*5)
        result = _det().detect(df, "5m")
        assert _det().get_last_swing_high(result) is None

    def test_get_recent_swings_returns_dataframe(self):
        result = self._result()
        swings = _det().get_recent_swings(result, n=3)
        assert isinstance(swings, pd.DataFrame)
        assert len(swings) <= 3

    def test_get_recent_swings_newest_first(self):
        result = self._result()
        swings = _det().get_recent_swings(result, n=10)
        if len(swings) > 1:
            assert swings["confirm_ts"].iloc[0] >= swings["confirm_ts"].iloc[1]


# ------------------------------------------------------------------ #
# Validation / Error handling                                          #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_unknown_timeframe_raises(self):
        df = _make_df([10]*10, [8]*10)
        with pytest.raises(ValueError, match="No fractal window"):
            _det().detect(df, "3m")

    def test_even_window_raises(self):
        with pytest.raises(ValueError, match="odd number"):
            SwingDetector({"5m": 4}).detect(_make_df([10]*10, [8]*10), "5m")

    def test_missing_high_column_raises(self):
        df = _make_df([10]*10, [8]*10).drop(columns=["high"])
        with pytest.raises(ValueError, match="Missing columns"):
            _det().detect(df, "5m")

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame(
            {"high": [10]*5, "low": [8]*5},
            index=range(5),
        )
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _det().detect(df, "5m")

    def test_convenience_function_works(self):
        highs = [10, 11, 12, 13, 20, 13, 12, 11, 10, 9]
        lows  = [ 8,  9, 10, 11, 18, 11, 10,  9,  8, 7]
        df    = _make_df(highs, lows)
        result = detect_swings(df, "5m")
        assert "swing_high" in result.columns
