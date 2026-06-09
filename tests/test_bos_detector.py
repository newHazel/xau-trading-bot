"""
Tests for bos_detector.py — Phase 1.3.

Critical properties:
  - Close only: a wick beyond the level is NOT a BOS.
  - Close must be STRICTLY greater (bull) / less (bear) than the level.
  - Each confirmed swing level is consumed by the first BOS it produces;
    a second break of the same level does not fire again.
  - No look-ahead: BOS can only be detected at bars where the swing level
    was already confirmed.
  - No BOS before any swing is confirmed.
"""

import pytest
import numpy as np
import pandas as pd

from core.structure.swing_detector import SwingDetector
from core.structure.bos_detector import BOSDetector, detect_bos


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_raw_df(highs, lows, closes=None, start="2026-01-05 10:00", freq="5min"):
    n = len(highs)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    mid = closes
    return pd.DataFrame(
        {"open": mid, "high": highs, "low": lows, "close": mid, "volume": [100.0] * n},
        index=idx,
    )


def _make_swing_df(n=20, start="2026-01-05 10:00", freq="5min", close_val=9.0):
    """Minimal DataFrame with swing columns (all NaN / −1) for injection tests."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open":           [9.0]    * n,
            "high":           [10.0]   * n,
            "low":            [8.0]    * n,
            "close":          [close_val] * n,
            "volume":         [100.0]  * n,
            "swing_high":     [np.nan] * n,
            "swing_low":      [np.nan] * n,
            "swing_high_idx": [-1] * n,
            "swing_low_idx":  [-1] * n,
        },
        index=idx,
    )


def _set_sh(df, pos, price, swing_bar=None):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("swing_high")]     = float(price)
    df.iloc[pos, df.columns.get_loc("swing_high_idx")] = swing_bar if swing_bar is not None else max(0, pos - 2)
    return df


def _set_sl(df, pos, price, swing_bar=None):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("swing_low")]     = float(price)
    df.iloc[pos, df.columns.get_loc("swing_low_idx")] = swing_bar if swing_bar is not None else max(0, pos - 2)
    return df


def _set_close(df, pos, close):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("close")] = float(close)
    return df


def _det():
    return BOSDetector()


def _full_pipeline(highs, lows, closes=None, tf="5m", freq="5min"):
    """Run SwingDetector then BOSDetector on synthetic OHLCV data."""
    df = _make_raw_df(highs, lows, closes, freq=freq)
    with_swings = SwingDetector().detect(df, tf)
    return BOSDetector().detect(with_swings)


# ------------------------------------------------------------------ #
# Basic BOS detection                                                  #
# ------------------------------------------------------------------ #

class TestBasicBOS:
    def test_bullish_bos_when_close_strictly_above_swing_high(self):
        df = _make_swing_df()
        df = _set_sh(df, 4, 20.0)
        df = _set_close(df, 8, 21.0)       # close 21 > 20 → BOS bull
        result = _det().detect(df)
        assert result["bos_bull"].iloc[8] == pytest.approx(20.0)

    def test_bearish_bos_when_close_strictly_below_swing_low(self):
        df = _make_swing_df()
        df = _set_sl(df, 4, 5.0)
        df = _set_close(df, 8, 4.0)        # close 4 < 5 → BOS bear
        result = _det().detect(df)
        assert result["bos_bear"].iloc[8] == pytest.approx(5.0)

    def test_no_bos_when_close_equals_swing_high(self):
        df = _make_swing_df()
        df = _set_sh(df, 4, 20.0)
        df = _set_close(df, 8, 20.0)       # equal — NOT a BOS
        result = _det().detect(df)
        assert result["bos_bull"].isna().all()

    def test_no_bos_when_close_equals_swing_low(self):
        df = _make_swing_df()
        df = _set_sl(df, 4, 5.0)
        df = _set_close(df, 8, 5.0)        # equal — NOT a BOS
        result = _det().detect(df)
        assert result["bos_bear"].isna().all()

    def test_no_bos_when_close_below_swing_high(self):
        df = _make_swing_df()
        df = _set_sh(df, 4, 20.0)
        df = _set_close(df, 8, 19.0)
        result = _det().detect(df)
        assert result["bos_bull"].isna().all()

    def test_no_bos_when_close_above_swing_low(self):
        df = _make_swing_df()
        df = _set_sl(df, 4, 5.0)
        df = _set_close(df, 8, 6.0)
        result = _det().detect(df)
        assert result["bos_bear"].isna().all()

    def test_bos_ref_bar_stores_swing_bar_index(self):
        df = _make_swing_df()
        df = _set_sh(df, 4, 20.0, swing_bar=2)   # swing was at bar 2
        df = _set_close(df, 8, 21.0)
        result = _det().detect(df)
        assert result["bos_bull_ref_bar"].iloc[8] == 2

    def test_bos_ref_bar_stores_swing_low_bar_index(self):
        df = _make_swing_df()
        df = _set_sl(df, 4, 5.0, swing_bar=2)
        df = _set_close(df, 8, 4.0)
        result = _det().detect(df)
        assert result["bos_bear_ref_bar"].iloc[8] == 2


# ------------------------------------------------------------------ #
# Close-only rule                                                      #
# ------------------------------------------------------------------ #

class TestCloseOnly:
    def test_wick_above_swing_high_does_not_trigger_bos(self):
        """High > swing level but close < level → no BOS."""
        df = _make_swing_df()
        df = _set_sh(df, 4, 20.0)
        # bar 8: high=22 (wick breaks), close=19 (close stays below)
        df.iloc[8, df.columns.get_loc("high")]  = 22.0
        df.iloc[8, df.columns.get_loc("close")] = 19.0
        result = _det().detect(df)
        assert result["bos_bull"].isna().all()

    def test_wick_below_swing_low_does_not_trigger_bos(self):
        """Low < swing level but close > level → no BOS."""
        df = _make_swing_df()
        df = _set_sl(df, 4, 5.0)
        df.iloc[8, df.columns.get_loc("low")]   = 2.0
        df.iloc[8, df.columns.get_loc("close")] = 6.0
        result = _det().detect(df)
        assert result["bos_bear"].isna().all()


# ------------------------------------------------------------------ #
# Level consumption (one BOS per level)                                #
# ------------------------------------------------------------------ #

class TestLevelConsumption:
    def test_same_level_not_broken_twice(self):
        """After a BOS fires, the level is consumed. Same level can't fire again."""
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)
        df = _set_close(df, 5, 21.0)   # BOS at bar 5
        df = _set_close(df, 6, 22.0)   # close still above 20 — must NOT fire again
        result = _det().detect(df)
        # Only bar 5 should have bos_bull
        assert result["bos_bull"].notna().sum() == 1
        assert result["bos_bull"].iloc[5] == pytest.approx(20.0)
        assert np.isnan(result["bos_bull"].iloc[6])

    def test_new_swing_after_bos_enables_new_bos(self):
        """After BOS consumed the first level, a new swing at a higher level
        enables a second BOS when that level is broken."""
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)      # pending = 20
        df = _set_close(df, 5, 21.0)   # BOS at bar 5; level 20 consumed
        df = _set_sh(df, 7, 25.0)      # new swing; pending = 25
        df = _set_close(df, 10, 26.0)  # BOS at bar 10
        result = _det().detect(df)
        assert result["bos_bull"].notna().sum() == 2
        assert result["bos_bull"].iloc[5]  == pytest.approx(20.0)
        assert result["bos_bull"].iloc[10] == pytest.approx(25.0)

    def test_second_bos_uses_new_level_not_old(self):
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)
        df = _set_close(df, 5, 21.0)   # BOS at bar 5; level 20 consumed
        df = _set_sh(df, 7, 25.0)      # new level = 25
        df = _set_close(df, 10, 26.0)  # BOS at bar 10 breaks 25, not 20
        result = _det().detect(df)
        assert result["bos_bull"].iloc[10] == pytest.approx(25.0)


# ------------------------------------------------------------------ #
# No look-ahead                                                        #
# ------------------------------------------------------------------ #

class TestNoLookahead:
    def test_no_bos_before_swing_confirmed(self):
        """Bars before any swing confirmation must have no BOS."""
        df = _make_swing_df()
        df = _set_sh(df, 10, 20.0)
        df = _set_close(df, 12, 21.0)   # BOS at bar 12
        result = _det().detect(df)
        for i in range(10):
            assert np.isnan(result["bos_bull"].iloc[i]), f"bar {i} should not have BOS"

    def test_no_bos_at_swing_confirmation_bar_if_close_not_above(self):
        """The bar that confirms the swing must not trigger a spurious BOS
        unless its close is actually above the level."""
        df = _make_swing_df()
        df = _set_sh(df, 5, 20.0)
        # close at bar 5 is 9.0 (default) — well below 20 → no BOS
        result = _det().detect(df)
        assert np.isnan(result["bos_bull"].iloc[5])

    def test_future_close_does_not_affect_past_bars(self):
        """A massive close at a late bar must not put BOS labels on early bars."""
        df = _make_swing_df()
        df = _set_sh(df, 4, 20.0)
        df = _set_close(df, 15, 99_000.0)  # enormous close at bar 15
        result = _det().detect(df)
        # Bars 0-14 except bar 15 must not have BOS bull
        for i in range(15):
            assert np.isnan(result["bos_bull"].iloc[i]) or i == 15, \
                f"bar {i} should not have BOS"

    def test_bos_at_confirmation_bar_when_close_already_above(self):
        """Edge case: swing confirmed at bar X and bar X's close is above it.
        This is a valid BOS — the level was confirmed AND broken on the same bar."""
        df = _make_swing_df()
        df = _set_sh(df, 5, 20.0)
        df = _set_close(df, 5, 21.0)   # confirmed AND broken on same bar
        result = _det().detect(df)
        assert result["bos_bull"].iloc[5] == pytest.approx(20.0)


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _make_swing_df()
        result = _det().detect(df)
        for col in ["bos_bull", "bos_bear", "bos_bull_ref_bar", "bos_bear_ref_bar"]:
            assert col in result.columns

    def test_existing_columns_preserved(self):
        df = _make_swing_df()
        result = _det().detect(df)
        for col in ["swing_high", "swing_low", "close", "high", "low"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _make_swing_df()
        pd.testing.assert_index_equal(_det().detect(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _make_swing_df()
        _det().detect(df)
        assert "bos_bull" not in df.columns

    def test_bos_nan_when_no_bos(self):
        df = _make_swing_df()
        result = _det().detect(df)
        assert result["bos_bull"].isna().all()
        assert result["bos_bear"].isna().all()

    def test_ref_bar_minus1_when_no_bos(self):
        df = _make_swing_df()
        result = _det().detect(df)
        assert (result["bos_bull_ref_bar"] == -1).all()
        assert (result["bos_bear_ref_bar"] == -1).all()


# ------------------------------------------------------------------ #
# Accessor methods                                                      #
# ------------------------------------------------------------------ #

class TestAccessors:
    def _build_bull(self):
        df = _make_swing_df()
        df = _set_sh(df, 4, 20.0, swing_bar=2)
        df = _set_close(df, 8, 21.0)
        return _det().detect(df)

    def _build_bear(self):
        df = _make_swing_df()
        df = _set_sl(df, 4, 5.0, swing_bar=2)
        df = _set_close(df, 8, 4.0)
        return _det().detect(df)

    def test_get_last_bos_bull_returns_dict(self):
        result = self._build_bull()
        bos = _det().get_last_bos(result, "bull")
        assert isinstance(bos, dict)

    def test_get_last_bos_bull_level(self):
        result = self._build_bull()
        bos = _det().get_last_bos(result, "bull")
        assert bos["level"] == pytest.approx(20.0)

    def test_get_last_bos_bull_direction(self):
        result = self._build_bull()
        bos = _det().get_last_bos(result, "bull")
        assert bos["direction"] == "bull"

    def test_get_last_bos_bull_swing_bar(self):
        result = self._build_bull()
        bos = _det().get_last_bos(result, "bull")
        assert bos["swing_bar"] == 2

    def test_get_last_bos_bear_level(self):
        result = self._build_bear()
        bos = _det().get_last_bos(result, "bear")
        assert bos["level"] == pytest.approx(5.0)

    def test_get_last_bos_returns_none_when_empty(self):
        df = _make_swing_df()
        result = _det().detect(df)
        assert _det().get_last_bos(result, "bull") is None
        assert _det().get_last_bos(result, "bear") is None

    def test_get_all_bos_returns_list(self):
        result = self._build_bull()
        bos_list = _det().get_all_bos(result)
        assert isinstance(bos_list, list)

    def test_get_all_bos_newest_first(self):
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)
        df = _set_close(df, 5, 21.0)
        df = _set_sh(df, 7, 25.0)
        df = _set_close(df, 10, 26.0)
        result = _det().detect(df)
        bos_list = _det().get_all_bos(result, n=10)
        assert len(bos_list) == 2
        assert bos_list[0]["confirm_ts"] >= bos_list[1]["confirm_ts"]

    def test_get_all_bos_respects_n(self):
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)
        df = _set_close(df, 5, 21.0)
        df = _set_sh(df, 7, 25.0)
        df = _set_close(df, 10, 26.0)
        result = _det().detect(df)
        bos_list = _det().get_all_bos(result, n=1)
        assert len(bos_list) == 1

    def test_get_all_bos_has_required_keys(self):
        result = self._build_bull()
        bos_list = _det().get_all_bos(result, n=10)
        for entry in bos_list:
            for key in ("confirm_ts", "direction", "level", "swing_bar"):
                assert key in entry

    def test_get_all_bos_empty_list_when_no_bos(self):
        df = _make_swing_df()
        result = _det().detect(df)
        assert _det().get_all_bos(result) == []


# ------------------------------------------------------------------ #
# Validation / Error handling                                          #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_missing_close_column_raises(self):
        df = _make_swing_df().drop(columns=["close"])
        with pytest.raises(ValueError, match="Missing columns"):
            _det().detect(df)

    def test_missing_swing_columns_raises(self):
        idx = pd.date_range("2026-01-05", periods=5, freq="5min", tz="UTC")
        df = pd.DataFrame({"close": [9.0] * 5, "high": [10.] * 5, "low": [8.] * 5}, index=idx)
        with pytest.raises(ValueError, match="Missing columns"):
            _det().detect(df)

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame(
            {
                "close":          [9.] * 5,
                "swing_high":     [np.nan] * 5,
                "swing_low":      [np.nan] * 5,
                "swing_high_idx": [-1] * 5,
                "swing_low_idx":  [-1] * 5,
            },
            index=range(5),
        )
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _det().detect(df)

    def test_convenience_function_works(self):
        df = _make_swing_df()
        result = detect_bos(df)
        assert "bos_bull" in result.columns


# ------------------------------------------------------------------ #
# Integration with SwingDetector                                       #
# ------------------------------------------------------------------ #

class TestIntegration:
    def test_bullish_bos_detected_in_full_pipeline(self):
        """
        window=5, lag=2. Swing high at bar 4 (high=20), confirmed at bar 6.
        Bar 8: close = (22+20)/2 = 21 > 20 → BOS bull.
        """
        highs  = [10, 11, 12, 13, 20, 13, 12, 11, 22, 21, 20, 19]
        lows   = [ 8,  9, 10, 11, 18, 11, 10,  9, 20, 19, 18, 17]
        result = _full_pipeline(highs, lows)
        assert result["bos_bull"].notna().any()
        # BOS fires at bar 8 (close=21 > 20)
        assert result["bos_bull"].iloc[8] == pytest.approx(20.0)

    def test_bearish_bos_detected_in_full_pipeline(self):
        """
        Swing low at bar 4 (low=5), confirmed at bar 6.
        Bar 8: close = (4+1)/2 = 2.5 < 5 → BOS bear.
        """
        highs  = [20, 19, 18, 17, 10, 17, 18, 19,  4, 17, 18, 19]
        lows   = [18, 17, 16, 15,  5, 15, 16, 17,  1, 15, 16, 17]
        result = _full_pipeline(highs, lows)
        assert result["bos_bear"].notna().any()
        assert result["bos_bear"].iloc[8] == pytest.approx(5.0)

    def test_wick_does_not_trigger_bos_in_full_pipeline(self):
        """
        Swing high = 20 confirmed at bar 6.
        Bar 7: high=22 (wick above) but close explicitly set below 20.
        """
        highs = [10, 11, 12, 13, 20, 13, 22, 13, 12, 11, 10]
        lows  = [ 8,  9, 10, 11, 18, 11, 11, 11, 10,  9,  8]
        # close at bar 6 = (22+11)/2 = 16.5 < 20 → no BOS
        result = _full_pipeline(highs, lows)
        assert result["bos_bull"].isna().all()

    def test_no_bos_in_flat_market(self):
        highs = [10.0] * 20
        lows  = [8.0]  * 20
        result = _full_pipeline(highs, lows)
        assert result["bos_bull"].isna().all()
        assert result["bos_bear"].isna().all()

    def test_bos_ref_bar_points_to_swing_bar_in_pipeline(self):
        """bos_bull_ref_bar must point to the swing BAR (not confirmation bar)."""
        highs  = [10, 11, 12, 13, 20, 13, 12, 11, 22, 21, 20, 19]
        lows   = [ 8,  9, 10, 11, 18, 11, 10,  9, 20, 19, 18, 17]
        result = _full_pipeline(highs, lows)
        bos_row = result[result["bos_bull"].notna()].iloc[0]
        # swing_high_idx stored by SwingDetector is the swing bar (bar 4)
        assert int(bos_row["bos_bull_ref_bar"]) == 4

    def test_1m_timeframe_bos(self):
        """window=3, lag=1. Swing high at bar 2 (high=20), confirmed at bar 3.
        Bar 4: close=(21+20)/2=20.5 > 20 → BOS bull."""
        highs = [10, 11, 20, 11, 21, 20, 11, 10]
        lows  = [ 8,  9, 18,  9, 20, 18,  9,  8]
        result = _full_pipeline(highs, lows, tf="1m", freq="1min")
        assert result["bos_bull"].notna().any()
