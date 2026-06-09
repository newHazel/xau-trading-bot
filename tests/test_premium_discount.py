"""
Tests for premium_discount.py — Phase 1.5.

Critical properties:
  - Zone is "undefined" until BOTH a swing high AND swing low are confirmed.
  - Equilibrium = (range_high + range_low) / 2 exactly.
  - close > equilibrium → "premium"; close < equilibrium → "discount".
  - close == equilibrium → "equilibrium".
  - Range reference updates when a new swing is confirmed (carries forward).
  - Inverted range (last SH ≤ last SL price) → "undefined".
  - No look-ahead: range uses only swings confirmed at or before each bar.
"""

import pytest
import numpy as np
import pandas as pd

from core.structure.swing_detector import SwingDetector
from core.structure.premium_discount import PremiumDiscountAnalyzer, analyze_premium_discount


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_raw_df(highs, lows, closes=None, start="2026-01-05 10:00", freq="5min"):
    n = len(highs)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": [100.] * n},
        index=idx,
    )


def _make_swing_df(n=20, start="2026-01-05 10:00", freq="5min", close_val=9.0):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open":           [9.0]      * n,
            "high":           [10.0]     * n,
            "low":            [8.0]      * n,
            "close":          [close_val] * n,
            "volume":         [100.0]    * n,
            "swing_high":     [np.nan]   * n,
            "swing_low":      [np.nan]   * n,
            "swing_high_idx": [-1]       * n,
            "swing_low_idx":  [-1]       * n,
        },
        index=idx,
    )


def _set_sh(df, pos, price):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("swing_high")]     = float(price)
    df.iloc[pos, df.columns.get_loc("swing_high_idx")] = max(0, pos - 2)
    return df


def _set_sl(df, pos, price):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("swing_low")]     = float(price)
    df.iloc[pos, df.columns.get_loc("swing_low_idx")] = max(0, pos - 2)
    return df


def _set_close(df, pos, close):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("close")] = float(close)
    return df


def _pd():
    return PremiumDiscountAnalyzer()


def _full_pipeline(highs, lows, closes=None, tf="5m", freq="5min"):
    df = _make_raw_df(highs, lows, closes, freq=freq)
    with_swings = SwingDetector().detect(df, tf)
    return PremiumDiscountAnalyzer().analyze(with_swings)


# ------------------------------------------------------------------ #
# Basic zone detection                                                 #
# ------------------------------------------------------------------ #

class TestBasicZoneDetection:
    def _base(self):
        """SH=20, SL=10 → eq=15. Close controls zone."""
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)   # range_high = 20
        df = _set_sl(df, 6, 10.0)   # range_low  = 10, eq = 15
        return df

    def test_close_above_equilibrium_is_premium(self):
        df = self._base()
        df = _set_close(df, 8, 16.0)   # 16 > 15 → premium
        result = _pd().analyze(df)
        assert result["pd_zone"].iloc[8] == "premium"

    def test_close_below_equilibrium_is_discount(self):
        df = self._base()
        df = _set_close(df, 8, 14.0)   # 14 < 15 → discount
        result = _pd().analyze(df)
        assert result["pd_zone"].iloc[8] == "discount"

    def test_close_at_equilibrium_is_equilibrium(self):
        df = self._base()
        df = _set_close(df, 8, 15.0)   # exactly 15 → equilibrium
        result = _pd().analyze(df)
        assert result["pd_zone"].iloc[8] == "equilibrium"

    def test_zone_undefined_before_both_swings_confirmed(self):
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)      # only swing high, no swing low yet
        result = _pd().analyze(df)
        for i in range(4):
            assert result["pd_zone"].iloc[i] == "undefined"
        # After SH confirmed but before SL → still undefined
        assert result["pd_zone"].iloc[3] == "undefined"

    def test_zone_undefined_with_only_swing_low(self):
        df = _make_swing_df()
        df = _set_sl(df, 5, 10.0)      # only swing low, no swing high
        result = _pd().analyze(df)
        assert (result["pd_zone"] == "undefined").all()

    def test_zone_undefined_with_only_swing_high(self):
        df = _make_swing_df()
        df = _set_sh(df, 5, 20.0)
        result = _pd().analyze(df)
        assert (result["pd_zone"] == "undefined").all()

    def test_zone_begins_after_both_swings_confirmed(self):
        df = self._base()
        df = _set_close(df, 7, 16.0)
        result = _pd().analyze(df)
        # Bars 0-5 (before SL at bar 6) → undefined
        for i in range(6):
            assert result["pd_zone"].iloc[i] == "undefined", f"bar {i}"
        # Bar 6 onward → defined (SH at 3, SL at 6)
        assert result["pd_zone"].iloc[6] != "undefined"


# ------------------------------------------------------------------ #
# Equilibrium and range computation                                    #
# ------------------------------------------------------------------ #

class TestRangeComputation:
    def test_equilibrium_is_exact_midpoint(self):
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)
        df = _set_sl(df, 6, 10.0)
        result = _pd().analyze(df)
        assert result["pd_equilibrium"].iloc[8] == pytest.approx(15.0)

    def test_range_high_set_to_last_swing_high(self):
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)
        df = _set_sl(df, 6, 10.0)
        result = _pd().analyze(df)
        assert result["pd_range_high"].iloc[8] == pytest.approx(20.0)

    def test_range_low_set_to_last_swing_low(self):
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)
        df = _set_sl(df, 6, 10.0)
        result = _pd().analyze(df)
        assert result["pd_range_low"].iloc[8] == pytest.approx(10.0)

    def test_range_updates_when_new_swing_high_confirmed(self):
        """New swing high at bar 12 updates range_high from 20 to 25."""
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)
        df = _set_sl(df, 6, 10.0)
        df = _set_sh(df, 12, 25.0)   # new SH → new equilibrium = (25+10)/2 = 17.5
        result = _pd().analyze(df)
        assert result["pd_range_high"].iloc[13]  == pytest.approx(25.0)
        assert result["pd_equilibrium"].iloc[13] == pytest.approx(17.5)

    def test_range_updates_when_new_swing_low_confirmed(self):
        """New swing low at bar 12 updates range_low from 10 to 5."""
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)
        df = _set_sl(df, 6, 10.0)
        df = _set_sl(df, 12, 5.0)    # new SL → eq = (20+5)/2 = 12.5
        result = _pd().analyze(df)
        assert result["pd_range_low"].iloc[13]   == pytest.approx(5.0)
        assert result["pd_equilibrium"].iloc[13] == pytest.approx(12.5)

    def test_range_carried_forward_between_swings(self):
        """Once established, the range is available on every subsequent bar."""
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)
        df = _set_sl(df, 6, 10.0)
        result = _pd().analyze(df)
        for i in range(6, len(df)):
            assert result["pd_equilibrium"].iloc[i] == pytest.approx(15.0), f"bar {i}"

    def test_inverted_range_is_undefined(self):
        """If last SH price ≤ last SL price, zone must be undefined."""
        df = _make_swing_df()
        df = _set_sh(df, 3, 10.0)   # SH price = 10
        df = _set_sl(df, 6, 20.0)   # SL price = 20  → SH <= SL → inverted
        result = _pd().analyze(df)
        assert result["pd_zone"].iloc[8] == "undefined"
        assert np.isnan(result["pd_equilibrium"].iloc[8])

    def test_non_integer_equilibrium(self):
        """SH=21, SL=10 → eq=15.5."""
        df = _make_swing_df()
        df = _set_sh(df, 3, 21.0)
        df = _set_sl(df, 6, 10.0)
        result = _pd().analyze(df)
        assert result["pd_equilibrium"].iloc[8] == pytest.approx(15.5)


# ------------------------------------------------------------------ #
# Zone tracking over time                                              #
# ------------------------------------------------------------------ #

class TestZoneTracking:
    def test_zone_changes_as_close_moves(self):
        """Same range, different closes at different bars produce different zones."""
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)
        df = _set_sl(df, 4, 10.0)
        df = _set_close(df, 6, 17.0)    # premium
        df = _set_close(df, 7, 13.0)    # discount
        df = _set_close(df, 8, 15.0)    # equilibrium
        result = _pd().analyze(df)
        assert result["pd_zone"].iloc[6] == "premium"
        assert result["pd_zone"].iloc[7] == "discount"
        assert result["pd_zone"].iloc[8] == "equilibrium"

    def test_zone_flips_after_range_update(self):
        """Close that was 'discount' under old range may become 'premium' after
        range_low drops (new lower swing low)."""
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)
        df = _set_sl(df, 4, 10.0)    # eq = 15; close=9 → default → 9 < 15 = discount
        df = _set_sl(df, 8, 2.0)     # new SL → eq = (20+2)/2 = 11; close=9 < 11 = discount still
        # Let's set close to 12 so it flips
        df = _set_close(df, 5, 9.0)   # old eq=15: 9 < 15 → discount
        df = _set_close(df, 9, 12.0)  # new eq=11: 12 > 11 → premium
        result = _pd().analyze(df)
        assert result["pd_zone"].iloc[5] == "discount"
        assert result["pd_zone"].iloc[9] == "premium"


# ------------------------------------------------------------------ #
# No look-ahead                                                        #
# ------------------------------------------------------------------ #

class TestNoLookahead:
    def test_bars_before_second_swing_have_undefined_zone(self):
        df = _make_swing_df()
        df = _set_sh(df, 5, 20.0)    # SH at bar 5
        df = _set_sl(df, 10, 10.0)   # SL at bar 10 → range established at bar 10
        result = _pd().analyze(df)
        for i in range(10):
            assert result["pd_zone"].iloc[i] == "undefined", f"bar {i}"

    def test_future_swing_does_not_affect_past_equilibrium(self):
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)
        df = _set_sl(df, 6, 10.0)    # eq = 15 from bar 6
        df = _set_sh(df, 15, 30.0)   # new SH at bar 15 → eq becomes 20
        result = _pd().analyze(df)
        # Bars 6-14 should use eq=15, not eq=20
        for i in range(6, 15):
            assert result["pd_equilibrium"].iloc[i] == pytest.approx(15.0), f"bar {i}"
        # Bar 15+ should use eq=20
        assert result["pd_equilibrium"].iloc[15] == pytest.approx(20.0)


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _make_swing_df()
        result = _pd().analyze(df)
        for col in ["pd_range_high", "pd_range_low", "pd_equilibrium", "pd_zone"]:
            assert col in result.columns

    def test_existing_columns_preserved(self):
        df = _make_swing_df()
        result = _pd().analyze(df)
        for col in ["swing_high", "swing_low", "close", "high", "low"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _make_swing_df()
        pd.testing.assert_index_equal(_pd().analyze(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _make_swing_df()
        _pd().analyze(df)
        assert "pd_zone" not in df.columns

    def test_pd_zone_on_every_bar(self):
        df = _make_swing_df()
        result = _pd().analyze(df)
        assert result["pd_zone"].notna().all()
        assert result["pd_zone"].isin(["premium", "discount", "equilibrium", "undefined"]).all()

    def test_range_nan_before_established(self):
        df = _make_swing_df()
        result = _pd().analyze(df)
        assert result["pd_range_high"].isna().all()
        assert result["pd_range_low"].isna().all()
        assert result["pd_equilibrium"].isna().all()


# ------------------------------------------------------------------ #
# Accessor methods                                                      #
# ------------------------------------------------------------------ #

class TestAccessors:
    def _build(self, close_val=16.0):
        df = _make_swing_df(close_val=close_val)
        df = _set_sh(df, 3, 20.0)
        df = _set_sl(df, 6, 10.0)
        return _pd().analyze(df)

    def test_get_current_zone_premium(self):
        assert _pd().get_current_zone(self._build(close_val=16.0)) == "premium"

    def test_get_current_zone_discount(self):
        assert _pd().get_current_zone(self._build(close_val=14.0)) == "discount"

    def test_get_current_zone_undefined_when_no_range(self):
        df = _make_swing_df()
        result = _pd().analyze(df)
        assert _pd().get_current_zone(result) == "undefined"

    def test_get_equilibrium_returns_midpoint(self):
        eq = _pd().get_equilibrium(self._build())
        assert eq == pytest.approx(15.0)

    def test_get_equilibrium_returns_none_when_not_established(self):
        df = _make_swing_df()
        result = _pd().analyze(df)
        assert _pd().get_equilibrium(result) is None

    def test_get_range_returns_dict(self):
        rng = _pd().get_range(self._build())
        assert isinstance(rng, dict)

    def test_get_range_keys(self):
        rng = _pd().get_range(self._build())
        for key in ("range_high", "range_low", "equilibrium", "range_size"):
            assert key in rng

    def test_get_range_values(self):
        rng = _pd().get_range(self._build())
        assert rng["range_high"]  == pytest.approx(20.0)
        assert rng["range_low"]   == pytest.approx(10.0)
        assert rng["equilibrium"] == pytest.approx(15.0)
        assert rng["range_size"]  == pytest.approx(10.0)

    def test_get_range_returns_none_when_not_established(self):
        df = _make_swing_df()
        result = _pd().analyze(df)
        assert _pd().get_range(result) is None


# ------------------------------------------------------------------ #
# Validation / Error handling                                          #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_missing_close_raises(self):
        df = _make_swing_df().drop(columns=["close"])
        with pytest.raises(ValueError, match="Missing columns"):
            _pd().analyze(df)

    def test_missing_swing_high_raises(self):
        df = _make_swing_df().drop(columns=["swing_high"])
        with pytest.raises(ValueError, match="Missing columns"):
            _pd().analyze(df)

    def test_missing_swing_low_raises(self):
        df = _make_swing_df().drop(columns=["swing_low"])
        with pytest.raises(ValueError, match="Missing columns"):
            _pd().analyze(df)

    def test_non_datetime_index_raises(self):
        df = _make_swing_df()
        df.index = range(len(df))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _pd().analyze(df)

    def test_convenience_function_works(self):
        df = _make_swing_df()
        result = analyze_premium_discount(df)
        assert "pd_zone" in result.columns


# ------------------------------------------------------------------ #
# Integration with SwingDetector                                       #
# ------------------------------------------------------------------ #

class TestIntegration:
    def test_premium_zone_in_full_pipeline(self):
        """
        window=5, lag=2.
        Swing high at bar 4 (high=20), confirmed at bar 6.
        Swing low  at bar 9 (low=10),  confirmed at bar 11.
        eq = (20+10)/2 = 15.
        Bar 12: close = (18+16)/2 = 17 > 15 → premium.
        """
        highs = [10,11,12,13,20,13,12,15,16,17,15,14,18,14,15]
        lows  = [ 8, 9,10,11,18,11,10,13,14,10,13,12,16,12,13]
        result = _full_pipeline(highs, lows)
        assert result["pd_zone"].iloc[12] == "premium"

    def test_discount_zone_in_full_pipeline(self):
        """Bar 13 in same data: close = (14+12)/2 = 13 < 15 → discount."""
        highs = [10,11,12,13,20,13,12,15,16,17,15,14,18,14,15]
        lows  = [ 8, 9,10,11,18,11,10,13,14,10,13,12,16,12,13]
        result = _full_pipeline(highs, lows)
        assert result["pd_zone"].iloc[13] == "discount"

    def test_equilibrium_correct_in_full_pipeline(self):
        highs = [10,11,12,13,20,13,12,15,16,17,15,14,18,14,15]
        lows  = [ 8, 9,10,11,18,11,10,13,14,10,13,12,16,12,13]
        result = _full_pipeline(highs, lows)
        # SH=20 confirmed at bar 6; SL=10 confirmed at bar 8 → eq=15 from bar 8.
        # At bar 11 a new SH=17 is confirmed (from swing bar 9) → eq=(17+10)/2=13.5.
        assert result["pd_equilibrium"].iloc[8]  == pytest.approx(15.0)
        assert result["pd_equilibrium"].iloc[12] == pytest.approx(13.5)

    def test_undefined_before_both_swings_in_full_pipeline(self):
        highs = [10,11,12,13,20,13,12,15,16,17,15,14,18,14,15]
        lows  = [ 8, 9,10,11,18,11,10,13,14,10,13,12,16,12,13]
        result = _full_pipeline(highs, lows)
        # SH confirmed at bar 6, SL confirmed at bar 8 → range starts at bar 8.
        # Bars 0-7 must be undefined.
        for i in range(8):
            assert result["pd_zone"].iloc[i] == "undefined", f"bar {i}"

    def test_range_updates_after_new_swing_in_pipeline(self):
        """A second swing high of 25 updates the equilibrium after bar 6."""
        highs = [10,11,12,13,20,13,12,15,16,17,15,14,25,14,13,12,11,10, 9, 8]
        lows  = [ 8, 9,10,11,18,11,10,13,14,10,13,12,23,12,11,10, 9, 8, 7, 6]
        result = _full_pipeline(highs, lows)
        rng = _pd().get_range(result)
        assert rng is not None
        # Range high should be updated to 25 at some point
        assert result["pd_range_high"].dropna().max() == pytest.approx(25.0)

    def test_flat_series_zone_undefined(self):
        highs = [10.] * 20
        lows  = [8.]  * 20
        result = _full_pipeline(highs, lows)
        assert (result["pd_zone"] == "undefined").all()

    def test_1m_timeframe_integration(self):
        """window=3, lag=1."""
        highs = [10,11,20,11,10, 9,11,12,10, 5,12,13,18,13,12,11]
        lows  = [ 8, 9,18, 9, 8, 7, 9,10, 8, 3,10,11,16,11,10, 9]
        result = _full_pipeline(highs, lows, tf="1m", freq="1min")
        # At some point both a swing high and swing low are confirmed
        assert (result["pd_zone"] != "undefined").any()
