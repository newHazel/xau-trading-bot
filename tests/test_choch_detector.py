"""
Tests for choch_detector.py — Phase 1.4.

Critical properties:
  - CHoCH bull fires ONLY when bias is "bearish" and close > swing high.
  - CHoCH bear fires ONLY when bias is "bullish" and close < swing low.
  - No CHoCH when bias is "neutral".
  - No CHoCH when the break ALIGNS with the bias (that is BOS, not CHoCH).
  - Close only: wick beyond the level is NOT a CHoCH.
  - Strictly beyond: close == level is NOT a CHoCH.
  - Each level is consumed after one CHoCH; same level cannot fire twice.
  - No look-ahead.
"""

import pytest
import numpy as np
import pandas as pd

from core.structure.swing_detector import SwingDetector
from core.structure.market_structure import MarketStructure
from core.structure.choch_detector import CHoCHDetector, detect_choch


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


def _make_swing_df(n=25, start="2026-01-05 10:00", freq="5min",
                   bias="neutral", close_val=9.0):
    """Minimal swing-annotated DataFrame with structure_bias column."""
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
            "structure_bias": [bias]     * n,
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


def _set_bias(df, from_pos, bias, to_pos=None):
    df = df.copy()
    end = to_pos if to_pos is not None else len(df)
    df.iloc[from_pos:end, df.columns.get_loc("structure_bias")] = bias
    return df


def _det():
    return CHoCHDetector()


def _full_pipeline(highs, lows, closes=None, tf="5m", freq="5min"):
    df          = _make_raw_df(highs, lows, closes, freq=freq)
    with_swings = SwingDetector().detect(df, tf)
    with_struct = MarketStructure().classify(with_swings)
    return CHoCHDetector().detect(with_struct)


# ------------------------------------------------------------------ #
# Basic CHoCH detection                                                #
# ------------------------------------------------------------------ #

class TestBasicCHoCH:
    def test_choch_bull_fires_in_bearish_market(self):
        """Bearish bias + close strictly above swing high → CHoCH bull."""
        df = _make_swing_df(bias="bearish")
        df = _set_sh(df, 4, 20.0, swing_bar=2)
        df = _set_close(df, 8, 21.0)
        result = _det().detect(df)
        assert result["choch_bull"].iloc[8] == pytest.approx(20.0)

    def test_choch_bear_fires_in_bullish_market(self):
        """Bullish bias + close strictly below swing low → CHoCH bear."""
        df = _make_swing_df(bias="bullish")
        df = _set_sl(df, 4, 5.0, swing_bar=2)
        df = _set_close(df, 8, 4.0)
        result = _det().detect(df)
        assert result["choch_bear"].iloc[8] == pytest.approx(5.0)

    def test_no_choch_bull_in_bullish_market(self):
        """Bullish bias + close above swing high → BOS (not CHoCH). CHoCH must not fire."""
        df = _make_swing_df(bias="bullish")
        df = _set_sh(df, 4, 20.0)
        df = _set_close(df, 8, 21.0)
        result = _det().detect(df)
        assert result["choch_bull"].isna().all()

    def test_no_choch_bear_in_bearish_market(self):
        """Bearish bias + close below swing low → BOS (not CHoCH). CHoCH must not fire."""
        df = _make_swing_df(bias="bearish")
        df = _set_sl(df, 4, 5.0)
        df = _set_close(df, 8, 4.0)
        result = _det().detect(df)
        assert result["choch_bear"].isna().all()

    def test_no_choch_when_bias_neutral(self):
        """Neutral bias → CHoCH cannot fire regardless of price action."""
        df = _make_swing_df(bias="neutral")
        df = _set_sh(df, 4, 20.0)
        df = _set_close(df, 8, 21.0)
        result = _det().detect(df)
        assert result["choch_bull"].isna().all()
        assert result["choch_bear"].isna().all()

    def test_choch_bull_ref_bar_stores_swing_bar(self):
        df = _make_swing_df(bias="bearish")
        df = _set_sh(df, 4, 20.0, swing_bar=2)
        df = _set_close(df, 8, 21.0)
        result = _det().detect(df)
        assert result["choch_bull_ref_bar"].iloc[8] == 2

    def test_choch_bear_ref_bar_stores_swing_bar(self):
        df = _make_swing_df(bias="bullish")
        df = _set_sl(df, 4, 5.0, swing_bar=2)
        df = _set_close(df, 8, 4.0)
        result = _det().detect(df)
        assert result["choch_bear_ref_bar"].iloc[8] == 2


# ------------------------------------------------------------------ #
# Close-only rule                                                      #
# ------------------------------------------------------------------ #

class TestCloseOnly:
    def test_wick_above_swing_high_does_not_trigger_choch(self):
        """High > level but close < level → no CHoCH bull."""
        df = _make_swing_df(bias="bearish")
        df = _set_sh(df, 4, 20.0)
        df.iloc[8, df.columns.get_loc("high")]  = 22.0
        df.iloc[8, df.columns.get_loc("close")] = 19.0
        result = _det().detect(df)
        assert result["choch_bull"].isna().all()

    def test_wick_below_swing_low_does_not_trigger_choch(self):
        """Low < level but close > level → no CHoCH bear."""
        df = _make_swing_df(bias="bullish")
        df = _set_sl(df, 4, 5.0)
        df.iloc[8, df.columns.get_loc("low")]   = 2.0
        df.iloc[8, df.columns.get_loc("close")] = 6.0
        result = _det().detect(df)
        assert result["choch_bear"].isna().all()

    def test_no_choch_bull_when_close_equals_swing_high(self):
        df = _make_swing_df(bias="bearish")
        df = _set_sh(df, 4, 20.0)
        df = _set_close(df, 8, 20.0)    # equal — NOT a CHoCH
        result = _det().detect(df)
        assert result["choch_bull"].isna().all()

    def test_no_choch_bear_when_close_equals_swing_low(self):
        df = _make_swing_df(bias="bullish")
        df = _set_sl(df, 4, 5.0)
        df = _set_close(df, 8, 5.0)     # equal — NOT a CHoCH
        result = _det().detect(df)
        assert result["choch_bear"].isna().all()


# ------------------------------------------------------------------ #
# Level consumption                                                    #
# ------------------------------------------------------------------ #

class TestLevelConsumption:
    def test_same_level_cannot_trigger_choch_twice(self):
        df = _make_swing_df(bias="bearish")
        df = _set_sh(df, 2, 20.0)
        df = _set_close(df, 5, 21.0)    # CHoCH bull at bar 5
        df = _set_close(df, 6, 22.0)    # close still above 20 — must NOT fire again
        result = _det().detect(df)
        assert result["choch_bull"].notna().sum() == 1
        assert result["choch_bull"].iloc[5] == pytest.approx(20.0)
        assert np.isnan(result["choch_bull"].iloc[6])

    def test_new_swing_enables_new_choch_after_consumption(self):
        df = _make_swing_df(bias="bearish")
        df = _set_sh(df, 2, 20.0)
        df = _set_close(df, 5, 21.0)    # CHoCH bull; 20 consumed
        df = _set_sh(df, 8, 25.0)       # new swing confirmed
        df = _set_close(df, 12, 26.0)   # CHoCH bull on 25
        result = _det().detect(df)
        assert result["choch_bull"].notna().sum() == 2
        assert result["choch_bull"].iloc[12] == pytest.approx(25.0)


# ------------------------------------------------------------------ #
# Bias sensitivity                                                     #
# ------------------------------------------------------------------ #

class TestBiasSensitivity:
    def test_choch_fires_only_at_bar_where_bias_is_correct(self):
        """Bias changes mid-series. CHoCH fires only after bias flips to 'bearish'."""
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)
        # Bars 0-9 neutral, bars 10+ bearish
        df = _set_bias(df, 10, "bearish")
        df = _set_close(df, 14, 21.0)   # close > 20, bias=bearish → CHoCH bull
        result = _det().detect(df)
        assert result["choch_bull"].iloc[14] == pytest.approx(20.0)
        # No CHoCH before bar 10 (bias was neutral)
        assert result["choch_bull"].iloc[:10].isna().all()

    def test_no_choch_before_bias_turns(self):
        """Close breaks the swing high BEFORE the bias turns bearish → no CHoCH yet."""
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)
        df = _set_close(df, 6, 21.0)    # close > 20 but bias still "neutral" → no CHoCH
        df = _set_bias(df, 10, "bearish")
        result = _det().detect(df)
        assert np.isnan(result["choch_bull"].iloc[6])


# ------------------------------------------------------------------ #
# No look-ahead                                                        #
# ------------------------------------------------------------------ #

class TestNoLookahead:
    def test_no_choch_before_swing_confirmed(self):
        df = _make_swing_df(bias="bearish")
        df = _set_sh(df, 10, 20.0)     # swing confirmed at bar 10
        df = _set_close(df, 12, 21.0)  # CHoCH at bar 12
        result = _det().detect(df)
        for i in range(10):
            assert np.isnan(result["choch_bull"].iloc[i]), f"bar {i} should be NaN"

    def test_choch_at_confirmation_bar_when_close_already_above(self):
        """If swing confirmed at bar X AND bar X's close is above it → CHoCH fires."""
        df = _make_swing_df(bias="bearish")
        df = _set_sh(df, 5, 20.0)
        df = _set_close(df, 5, 21.0)   # same bar: confirmed AND broken
        result = _det().detect(df)
        assert result["choch_bull"].iloc[5] == pytest.approx(20.0)

    def test_future_break_does_not_affect_past_bars(self):
        df = _make_swing_df(bias="bearish")
        df = _set_sh(df, 4, 20.0)
        df = _set_close(df, 18, 99_000.0)
        result = _det().detect(df)
        for i in range(18):
            assert np.isnan(result["choch_bull"].iloc[i]), f"bar {i}"


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _make_swing_df()
        result = _det().detect(df)
        for col in ["choch_bull", "choch_bear", "choch_bull_ref_bar", "choch_bear_ref_bar"]:
            assert col in result.columns

    def test_existing_columns_preserved(self):
        df = _make_swing_df()
        result = _det().detect(df)
        for col in ["swing_high", "swing_low", "structure_bias", "close"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _make_swing_df()
        pd.testing.assert_index_equal(_det().detect(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _make_swing_df()
        _det().detect(df)
        assert "choch_bull" not in df.columns

    def test_choch_nan_when_no_event(self):
        df = _make_swing_df()
        result = _det().detect(df)
        assert result["choch_bull"].isna().all()
        assert result["choch_bear"].isna().all()

    def test_ref_bar_minus1_when_no_event(self):
        df = _make_swing_df()
        result = _det().detect(df)
        assert (result["choch_bull_ref_bar"] == -1).all()
        assert (result["choch_bear_ref_bar"] == -1).all()


# ------------------------------------------------------------------ #
# Accessor methods                                                      #
# ------------------------------------------------------------------ #

class TestAccessors:
    def _build_bull(self):
        df = _make_swing_df(bias="bearish")
        df = _set_sh(df, 4, 20.0, swing_bar=2)
        df = _set_close(df, 8, 21.0)
        return _det().detect(df)

    def _build_bear(self):
        df = _make_swing_df(bias="bullish")
        df = _set_sl(df, 4, 5.0, swing_bar=2)
        df = _set_close(df, 8, 4.0)
        return _det().detect(df)

    def test_get_last_choch_bull_returns_dict(self):
        assert isinstance(_det().get_last_choch(self._build_bull(), "bull"), dict)

    def test_get_last_choch_bull_level(self):
        choch = _det().get_last_choch(self._build_bull(), "bull")
        assert choch["level"] == pytest.approx(20.0)

    def test_get_last_choch_bull_direction(self):
        choch = _det().get_last_choch(self._build_bull(), "bull")
        assert choch["direction"] == "bull"

    def test_get_last_choch_bull_swing_bar(self):
        choch = _det().get_last_choch(self._build_bull(), "bull")
        assert choch["swing_bar"] == 2

    def test_get_last_choch_bear_level(self):
        choch = _det().get_last_choch(self._build_bear(), "bear")
        assert choch["level"] == pytest.approx(5.0)

    def test_get_last_choch_returns_none_when_empty(self):
        df = _make_swing_df()
        result = _det().detect(df)
        assert _det().get_last_choch(result, "bull") is None
        assert _det().get_last_choch(result, "bear") is None

    def test_get_all_choch_returns_list(self):
        assert isinstance(_det().get_all_choch(self._build_bull()), list)

    def test_get_all_choch_newest_first(self):
        df = _make_swing_df(bias="bearish")
        df = _set_sh(df, 2, 20.0)
        df = _set_close(df, 5, 21.0)
        df = _set_sh(df, 8, 25.0)
        df = _set_close(df, 12, 26.0)
        result = _det().detect(df)
        lst = _det().get_all_choch(result, n=10)
        assert len(lst) == 2
        assert lst[0]["confirm_ts"] >= lst[1]["confirm_ts"]

    def test_get_all_choch_respects_n(self):
        df = _make_swing_df(bias="bearish")
        df = _set_sh(df, 2, 20.0)
        df = _set_close(df, 5, 21.0)
        df = _set_sh(df, 8, 25.0)
        df = _set_close(df, 12, 26.0)
        result = _det().detect(df)
        assert len(_det().get_all_choch(result, n=1)) == 1

    def test_get_all_choch_empty_list_when_no_event(self):
        df = _make_swing_df()
        result = _det().detect(df)
        assert _det().get_all_choch(result) == []


# ------------------------------------------------------------------ #
# Validation / Error handling                                          #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_missing_structure_bias_raises(self):
        df = _make_swing_df().drop(columns=["structure_bias"])
        with pytest.raises(ValueError, match="Missing columns"):
            _det().detect(df)

    def test_missing_close_raises(self):
        df = _make_swing_df().drop(columns=["close"])
        with pytest.raises(ValueError, match="Missing columns"):
            _det().detect(df)

    def test_missing_swing_columns_raises(self):
        idx = pd.date_range("2026-01-05", periods=5, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {"close": [9.] * 5, "structure_bias": ["neutral"] * 5},
            index=idx,
        )
        with pytest.raises(ValueError, match="Missing columns"):
            _det().detect(df)

    def test_non_datetime_index_raises(self):
        df = _make_swing_df()
        df.index = range(len(df))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _det().detect(df)

    def test_convenience_function_works(self):
        df = _make_swing_df()
        result = detect_choch(df)
        assert "choch_bull" in result.columns


# ------------------------------------------------------------------ #
# Integration with full pipeline (1.1 → 1.2 → 1.4)                   #
# ------------------------------------------------------------------ #

class TestIntegration:
    def test_choch_bull_after_bearish_sequence(self):
        """
        Build a bearish market (LH + LL), then fire a CHoCH bull.

        window=5, lag=2:
          Peak 1  at bar  4 (high=25): confirmed at bar  6  — first high (no label)
          Trough 1 at bar  8 (low=8) : confirmed at bar 10  — first low  (no label)
          Peak 2  at bar 12 (high=20): confirmed at bar 14  → LH
          Trough 2 at bar 16 (low=5) : confirmed at bar 18  → LL  → bias = bearish

        Bar 20: close = (22+20)/2 = 21 > 20 (pending_sh from bar 14) → CHoCH bull.
        """
        highs = [10,11,12,13,25,13,12,11,10,11,12,13,20,13,12,11,10,11,12,13,22,21,20,19]
        lows  = [ 8, 9,10,11,23,11,10, 9, 8, 9,10,11,18,11,10, 9, 5, 9,10,11,20,19,18,17]
        result = _full_pipeline(highs, lows)
        assert result["choch_bull"].notna().any()
        assert result["choch_bull"].iloc[20] == pytest.approx(20.0)

    def test_choch_bear_after_bullish_sequence(self):
        """
        Build a bullish market (HH + HL), then fire a CHoCH bear.

        Peak 1  at bar  4 (high=20): confirmed at bar  6
        Trough 1 at bar  8 (low=5) : confirmed at bar 10
        Peak 2  at bar 12 (high=25): confirmed at bar 14  → HH
        Trough 2 at bar 16 (low=8) : confirmed at bar 18  → HL  → bias = bullish

        Bar 20: close = (6+4)/2 = 5 < 8 (pending_sl from bar 18) → CHoCH bear.
        """
        highs = [10,11,12,13,20,13,12,11,10,11,12,13,25,13,12,11,10,11,12,13, 6,11,12,13]
        lows  = [ 8, 9,10,11,18,11,10, 9, 5, 9,10,11,23,11,10, 9, 8, 9,10,11, 4, 9,10,11]
        result = _full_pipeline(highs, lows)
        assert result["choch_bear"].notna().any()
        assert result["choch_bear"].iloc[20] == pytest.approx(8.0)

    def test_no_choch_when_break_aligns_with_bullish_bias(self):
        """Bullish market + close > swing high = BOS bull, NOT CHoCH. choch_bull stays NaN."""
        highs = [10,11,12,13,20,13,12,11,10,11,12,13,25,13,12,11,10,11,12,13,28,21,20,19]
        lows  = [ 8, 9,10,11,18,11,10, 9, 5, 9,10,11,23,11,10, 9, 8, 9,10,11,26,19,18,17]
        result = _full_pipeline(highs, lows)
        # Bullish bias established; break upward → BOS not CHoCH
        assert result["choch_bull"].isna().all()

    def test_choch_bull_level_matches_broken_swing_high(self):
        highs = [10,11,12,13,25,13,12,11,10,11,12,13,20,13,12,11,10,11,12,13,22,21,20,19]
        lows  = [ 8, 9,10,11,23,11,10, 9, 8, 9,10,11,18,11,10, 9, 5, 9,10,11,20,19,18,17]
        result = _full_pipeline(highs, lows)
        choch_row = result[result["choch_bull"].notna()].iloc[0]
        assert choch_row["choch_bull"] == pytest.approx(20.0)

    def test_choch_bull_ref_bar_points_to_swing_bar(self):
        highs = [10,11,12,13,25,13,12,11,10,11,12,13,20,13,12,11,10,11,12,13,22,21,20,19]
        lows  = [ 8, 9,10,11,23,11,10, 9, 8, 9,10,11,18,11,10, 9, 5, 9,10,11,20,19,18,17]
        result = _full_pipeline(highs, lows)
        choch_row = result[result["choch_bull"].notna()].iloc[0]
        # swing bar for the LH (high=20 confirmed at bar 14) is bar 12
        assert int(choch_row["choch_bull_ref_bar"]) == 12

    def test_no_choch_in_flat_market(self):
        highs = [10.] * 25
        lows  = [8.]  * 25
        result = _full_pipeline(highs, lows)
        assert result["choch_bull"].isna().all()
        assert result["choch_bear"].isna().all()

    def test_1m_timeframe_choch(self):
        """window=3, lag=1. Bearish setup then CHoCH bull on 1m data."""
        # Peak 1 at bar 2 (high=25) confirmed at bar 3 — first high
        # Trough 1 at bar 5 (low=8) confirmed at bar 6 — first low
        # Peak 2 at bar 8 (high=20) confirmed at bar 9 → LH
        # Trough 2 at bar 11 (low=5) confirmed at bar 12 → LL → bearish
        # Bar 13: close = (22+20)/2 = 21 > 20 → CHoCH bull
        highs = [10,11,25,11,10, 9,11,12,20,12,11,10, 9,22,11,10]
        lows  = [ 8, 9,23, 9, 8, 7, 9,10,18,10, 9, 5, 7,20, 9, 8]
        result = _full_pipeline(highs, lows, tf="1m", freq="1min")
        assert result["choch_bull"].notna().any()
