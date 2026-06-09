"""
Tests for sweep_detector.py — Phase 2.2.

Critical properties:
  - Bearish sweep: HIGH > level AND a CLOSE strictly < level within `window`
    candles (counting the wick candle).
  - Bullish sweep: mirror — LOW < level AND CLOSE > level within `window`.
  - Strict inequality: close == level is NOT a sweep.
  - Wick alone (close stays on the wrong side and the window expires)
    is NOT a sweep.
  - Each level VALUE produces only one sweep per direction; only when the
    level value changes can it be swept again.
  - No look-ahead: pending sweeps never see future bars they shouldn't.
"""

import pytest
import numpy as np
import pandas as pd

from core.structure.swing_detector import SwingDetector
from core.smc.liquidity_detector import LiquidityDetector
from core.smc.sweep_detector import SweepDetector, detect_sweeps


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_complete_df(n=20, start="2026-01-05 10:00", freq="5min"):
    """A blank DataFrame with all columns the SweepDetector requires."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open":           [9.0]    * n,
            "high":           [10.0]   * n,
            "low":            [8.0]    * n,
            "close":          [9.0]    * n,
            "volume":         [100.0]  * n,
            "swing_high":     [np.nan] * n,
            "swing_low":      [np.nan] * n,
            "swing_high_idx": [-1]     * n,
            "swing_low_idx":  [-1]     * n,
            "eqh_level":      [np.nan] * n,
            "eqh_count":      [0]      * n,
            "eql_level":      [np.nan] * n,
            "eql_count":      [0]      * n,
            "pdh":            [np.nan] * n,
            "pdl":            [np.nan] * n,
        },
        index=idx,
    )


def _set_sh(df, pos, level, swing_bar=None):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("swing_high")]     = float(level)
    df.iloc[pos, df.columns.get_loc("swing_high_idx")] = swing_bar if swing_bar is not None else max(0, pos - 2)
    return df


def _set_sl(df, pos, level, swing_bar=None):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("swing_low")]     = float(level)
    df.iloc[pos, df.columns.get_loc("swing_low_idx")] = swing_bar if swing_bar is not None else max(0, pos - 2)
    return df


def _set_ohlc(df, pos, h=None, l=None, c=None, o=None):
    df = df.copy()
    if h is not None: df.iloc[pos, df.columns.get_loc("high")]  = float(h)
    if l is not None: df.iloc[pos, df.columns.get_loc("low")]   = float(l)
    if c is not None: df.iloc[pos, df.columns.get_loc("close")] = float(c)
    if o is not None: df.iloc[pos, df.columns.get_loc("open")]  = float(o)
    return df


def _set_eqh(df, from_pos, level, count=2):
    df = df.copy()
    df.iloc[from_pos:, df.columns.get_loc("eqh_level")] = float(level)
    df.iloc[from_pos:, df.columns.get_loc("eqh_count")] = int(count)
    return df


def _set_eql(df, from_pos, level, count=2):
    df = df.copy()
    df.iloc[from_pos:, df.columns.get_loc("eql_level")] = float(level)
    df.iloc[from_pos:, df.columns.get_loc("eql_count")] = int(count)
    return df


def _set_pdh(df, from_pos, level):
    df = df.copy()
    df.iloc[from_pos:, df.columns.get_loc("pdh")] = float(level)
    return df


def _set_pdl(df, from_pos, level):
    df = df.copy()
    df.iloc[from_pos:, df.columns.get_loc("pdl")] = float(level)
    return df


def _det(window=5):
    return SweepDetector(window=window)


# ------------------------------------------------------------------ #
# Basic sweep detection (single-bar)                                   #
# ------------------------------------------------------------------ #

class TestSingleBarSweep:
    def test_bearish_sweep_swing_high(self):
        """Wick above swing high + close strictly below — sweep on the same bar."""
        df = _make_complete_df()
        df = _set_sh(df, 4, 100.0, swing_bar=2)
        df = _set_ohlc(df, 8, h=102.0, c=98.0, l=97.0)
        result = _det().detect(df)
        assert result["sweep_bear_level"].iloc[8] == pytest.approx(100.0)
        assert result["sweep_bear_type"].iloc[8] == "swing_high"
        assert result["sweep_bear_wick_bar"].iloc[8] == 8

    def test_bullish_sweep_swing_low(self):
        # swing_low below default low (=8) so default lows on intervening bars
        # don't pre-pierce the level
        df = _make_complete_df()
        df = _set_sl(df, 4, 5.0, swing_bar=2)
        df = _set_ohlc(df, 8, l=3.0, c=7.0, h=8.0)
        result = _det().detect(df)
        assert result["sweep_bull_level"].iloc[8] == pytest.approx(5.0)
        assert result["sweep_bull_type"].iloc[8] == "swing_low"
        assert result["sweep_bull_wick_bar"].iloc[8] == 8

    def test_no_sweep_when_high_does_not_pierce(self):
        """High stays at-or-below the level — never a sweep candidate."""
        df = _make_complete_df()
        df = _set_sh(df, 4, 100.0)
        df = _set_ohlc(df, 8, h=99.0, c=98.0, l=97.0)
        result = _det().detect(df)
        assert np.isnan(result["sweep_bear_level"].iloc[8])

    def test_no_sweep_when_close_equals_level(self):
        """Close == level is NOT a confirmation (strict inequality)."""
        df = _make_complete_df()
        df = _set_sh(df, 4, 100.0)
        df = _set_ohlc(df, 8, h=102.0, c=100.0, l=99.5)
        result = _det().detect(df)
        assert np.isnan(result["sweep_bear_level"].iloc[8])

    def test_no_sweep_when_high_equals_level(self):
        """High == level is NOT a wick (strict inequality)."""
        df = _make_complete_df()
        df = _set_sh(df, 4, 100.0)
        df = _set_ohlc(df, 8, h=100.0, c=98.0, l=97.0)
        result = _det().detect(df)
        assert np.isnan(result["sweep_bear_level"].iloc[8])


# ------------------------------------------------------------------ #
# Multi-bar confirmation                                               #
# ------------------------------------------------------------------ #

class TestMultiBarConfirmation:
    def test_sweep_confirmed_one_bar_after_wick(self):
        """Wick at bar X (close still above), close back at bar X+1."""
        df = _make_complete_df()
        df = _set_sh(df, 4, 100.0)
        df = _set_ohlc(df, 8, h=102.0, c=101.0, l=100.5)   # wick — pending
        df = _set_ohlc(df, 9, h=101.5, c=99.0, l=98.0)     # close back below
        result = _det().detect(df)
        assert np.isnan(result["sweep_bear_level"].iloc[8])
        assert result["sweep_bear_level"].iloc[9] == pytest.approx(100.0)
        assert result["sweep_bear_wick_bar"].iloc[9] == 8

    def test_sweep_confirmed_at_window_boundary(self):
        """Wick at bar 10, close back at bar 14 (age=4 with window=5 → still valid)."""
        df = _make_complete_df(n=30)
        df = _set_sh(df, 4, 100.0)
        df = _set_ohlc(df, 10, h=102.0, c=101.0, l=100.5)  # wick
        for i in range(11, 14):
            df = _set_ohlc(df, i, h=101.5, c=101.0, l=100.5)  # stay above
        df = _set_ohlc(df, 14, h=101.5, c=99.0, l=98.5)    # close back
        result = _det(window=5).detect(df)
        assert result["sweep_bear_level"].iloc[14] == pytest.approx(100.0)
        assert result["sweep_bear_wick_bar"].iloc[14] == 10

    def test_sweep_expires_outside_window(self):
        """Same setup but close back at bar 15 (age=5, equal to window) → expired."""
        df = _make_complete_df(n=30)
        df = _set_sh(df, 4, 100.0)
        df = _set_ohlc(df, 10, h=102.0, c=101.0, l=100.5)
        for i in range(11, 15):
            df = _set_ohlc(df, i, h=101.5, c=101.0, l=100.5)
        df = _set_ohlc(df, 15, h=101.5, c=99.0, l=98.5)    # too late
        result = _det(window=5).detect(df)
        assert np.isnan(result["sweep_bear_level"].iloc[15])

    def test_window_one_means_single_bar_only(self):
        """window=1 → only single-bar sweeps (no multi-bar confirmation)."""
        df = _make_complete_df()
        df = _set_sh(df, 4, 100.0)
        df = _set_ohlc(df, 8, h=102.0, c=101.0, l=100.5)   # wick, no close back yet
        df = _set_ohlc(df, 9, h=101.5, c=99.0, l=98.5)     # close back at age=1, but window=1
        result = _det(window=1).detect(df)
        assert np.isnan(result["sweep_bear_level"].iloc[9])

    def test_bullish_sweep_confirmed_at_later_bar(self):
        # swing_low=5 (below default low=8) → defaults don't pre-pierce
        df = _make_complete_df()
        df = _set_sl(df, 4, 5.0)
        df = _set_ohlc(df, 8, l=3.0, c=4.0, h=5.5)         # wick low, close still below
        df = _set_ohlc(df, 9, l=4.0, c=7.0, h=8.0)         # close back above
        result = _det().detect(df)
        assert result["sweep_bull_level"].iloc[9] == pytest.approx(5.0)
        assert result["sweep_bull_wick_bar"].iloc[9] == 8


# ------------------------------------------------------------------ #
# Level consumption                                                    #
# ------------------------------------------------------------------ #

class TestLevelConsumption:
    def test_same_swing_level_not_swept_twice(self):
        df = _make_complete_df()
        df = _set_sh(df, 4, 100.0)
        df = _set_ohlc(df, 8, h=102.0, c=98.0, l=97.0)     # sweep on 100
        # Bar 12: another wick + close back on the SAME level — must NOT fire
        df = _set_ohlc(df, 12, h=103.0, c=98.0, l=97.0)
        result = _det().detect(df)
        assert result["sweep_bear_level"].notna().sum() == 1
        assert np.isnan(result["sweep_bear_level"].iloc[12])

    def test_new_swing_level_enables_new_sweep(self):
        df = _make_complete_df()
        df = _set_sh(df, 4, 100.0)
        df = _set_ohlc(df, 8, h=102.0, c=98.0, l=97.0)     # sweep on 100
        df = _set_sh(df, 14, 110.0)                        # new swing high → reset swept flag
        df = _set_ohlc(df, 16, h=112.0, c=108.0, l=107.0)  # sweep on 110
        result = _det().detect(df)
        assert result["sweep_bear_level"].notna().sum() == 2
        assert result["sweep_bear_level"].iloc[8]  == pytest.approx(100.0)
        assert result["sweep_bear_level"].iloc[16] == pytest.approx(110.0)


# ------------------------------------------------------------------ #
# Additional level types                                               #
# ------------------------------------------------------------------ #

class TestEQHEQLPDHPDL:
    def test_bearish_sweep_eqh(self):
        df = _make_complete_df()
        df = _set_eqh(df, from_pos=4, level=100.0)
        df = _set_ohlc(df, 8, h=102.0, c=98.0, l=97.0)
        result = _det().detect(df)
        assert result["sweep_bear_level"].iloc[8] == pytest.approx(100.0)
        assert result["sweep_bear_type"].iloc[8] == "eqh"

    def test_bullish_sweep_eql(self):
        df = _make_complete_df()
        df = _set_eql(df, from_pos=4, level=50.0)
        df = _set_ohlc(df, 8, l=48.0, c=52.0, h=53.0)
        result = _det().detect(df)
        assert result["sweep_bull_level"].iloc[8] == pytest.approx(50.0)
        assert result["sweep_bull_type"].iloc[8] == "eql"

    def test_bearish_sweep_pdh(self):
        df = _make_complete_df()
        df = _set_pdh(df, from_pos=4, level=100.0)
        df = _set_ohlc(df, 8, h=102.0, c=98.0, l=97.0)
        result = _det().detect(df)
        assert result["sweep_bear_level"].iloc[8] == pytest.approx(100.0)
        assert result["sweep_bear_type"].iloc[8] == "pdh"

    def test_bullish_sweep_pdl(self):
        df = _make_complete_df()
        df = _set_pdl(df, from_pos=4, level=50.0)
        df = _set_ohlc(df, 8, l=48.0, c=52.0, h=53.0)
        result = _det().detect(df)
        assert result["sweep_bull_level"].iloc[8] == pytest.approx(50.0)
        assert result["sweep_bull_type"].iloc[8] == "pdl"

    def test_eqh_value_change_resets_swept_flag(self):
        df = _make_complete_df()
        df = _set_eqh(df, from_pos=4, level=100.0)
        df = _set_ohlc(df, 8, h=102.0, c=98.0, l=97.0)     # sweep on 100
        # EQH cluster updates to 105 from bar 12
        df = _set_eqh(df, from_pos=12, level=105.0)
        df = _set_ohlc(df, 14, h=107.0, c=103.0, l=102.0)  # sweep on 105
        result = _det().detect(df)
        assert result["sweep_bear_level"].notna().sum() == 2
        assert result["sweep_bear_level"].iloc[14] == pytest.approx(105.0)


# ------------------------------------------------------------------ #
# No look-ahead                                                        #
# ------------------------------------------------------------------ #

class TestNoLookahead:
    def test_no_sweep_before_level_active(self):
        """Bar 5 has wick + close-back, but the swing isn't confirmed until bar 10."""
        df = _make_complete_df()
        df = _set_sh(df, 10, 100.0)
        df = _set_ohlc(df, 5, h=102.0, c=98.0, l=97.0)
        result = _det().detect(df)
        assert np.isnan(result["sweep_bear_level"].iloc[5])

    def test_future_wick_does_not_affect_past_bars(self):
        df = _make_complete_df()
        df = _set_sh(df, 4, 100.0)
        df = _set_ohlc(df, 15, h=102.0, c=98.0, l=97.0)
        result = _det().detect(df)
        # Only bar 15 must show the sweep
        for i in range(15):
            assert np.isnan(result["sweep_bear_level"].iloc[i]), f"bar {i}"
        assert result["sweep_bear_level"].iloc[15] == pytest.approx(100.0)


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _make_complete_df()
        result = _det().detect(df)
        for col in [
            "sweep_bull_level", "sweep_bull_type", "sweep_bull_wick_bar",
            "sweep_bear_level", "sweep_bear_type", "sweep_bear_wick_bar",
        ]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _make_complete_df()
        pd.testing.assert_index_equal(_det().detect(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _make_complete_df()
        _det().detect(df)
        assert "sweep_bear_level" not in df.columns

    def test_no_sweep_means_all_nan(self):
        df = _make_complete_df()
        result = _det().detect(df)
        assert result["sweep_bull_level"].isna().all()
        assert result["sweep_bear_level"].isna().all()
        assert (result["sweep_bear_wick_bar"] == -1).all()

    def test_existing_columns_preserved(self):
        df = _make_complete_df()
        result = _det().detect(df)
        for col in ["high", "low", "close", "swing_high", "eqh_level", "pdh"]:
            assert col in result.columns


# ------------------------------------------------------------------ #
# Accessor                                                             #
# ------------------------------------------------------------------ #

class TestAccessors:
    def _build_bear(self):
        df = _make_complete_df()
        df = _set_sh(df, 4, 100.0, swing_bar=2)
        df = _set_ohlc(df, 8, h=102.0, c=98.0, l=97.0)
        return _det().detect(df)

    def test_get_last_sweep_bear_returns_dict(self):
        result = self._build_bear()
        info = _det().get_last_sweep(result, direction="bear")
        assert isinstance(info, dict)

    def test_get_last_sweep_bear_level(self):
        result = self._build_bear()
        info = _det().get_last_sweep(result, direction="bear")
        assert info["level"] == pytest.approx(100.0)

    def test_get_last_sweep_bear_type(self):
        result = self._build_bear()
        info = _det().get_last_sweep(result, direction="bear")
        assert info["type"] == "swing_high"

    def test_get_last_sweep_returns_none_when_empty(self):
        df = _make_complete_df()
        result = _det().detect(df)
        assert _det().get_last_sweep(result, direction="bear") is None
        assert _det().get_last_sweep(result, direction="bull") is None


# ------------------------------------------------------------------ #
# Validation                                                           #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_window_zero_raises(self):
        with pytest.raises(ValueError, match="window"):
            SweepDetector(window=0)

    def test_window_negative_raises(self):
        with pytest.raises(ValueError, match="window"):
            SweepDetector(window=-1)

    def test_missing_swing_columns_raises(self):
        df = _make_complete_df().drop(columns=["swing_high"])
        with pytest.raises(ValueError, match="Missing"):
            _det().detect(df)

    def test_missing_eqh_column_raises(self):
        df = _make_complete_df().drop(columns=["eqh_level"])
        with pytest.raises(ValueError, match="Missing"):
            _det().detect(df)

    def test_missing_pdh_column_raises(self):
        df = _make_complete_df().drop(columns=["pdh"])
        with pytest.raises(ValueError, match="Missing"):
            _det().detect(df)

    def test_non_datetime_index_raises(self):
        df = _make_complete_df()
        df.index = range(len(df))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _det().detect(df)

    def test_convenience_function_works(self):
        df = _make_complete_df()
        result = detect_sweeps(df)
        assert "sweep_bear_level" in result.columns


# ------------------------------------------------------------------ #
# Integration with full pipeline                                       #
# ------------------------------------------------------------------ #

class TestIntegration:
    def _full_pipeline(self, highs, lows, closes, freq="5min", tf="5m"):
        n = len(highs)
        idx = pd.date_range("2026-01-05 10:00", periods=n, freq=freq,
                            tz="UTC", name="timestamp")
        df = pd.DataFrame(
            {"open": closes, "high": highs, "low": lows, "close": closes,
             "volume": [100.] * n},
            index=idx,
        )
        with_swings = SwingDetector().detect(df, tf)
        with_liq    = LiquidityDetector().detect(with_swings)
        return SweepDetector().detect(with_liq)

    def test_bearish_sweep_swing_high_in_full_pipeline(self):
        """Swing high at bar 4 (high=20) confirmed at bar 6.
        Bar 8: high=22 (wick above), close=19 (back below) → bear sweep."""
        highs   = [10, 11, 12, 13, 20, 13, 12, 11, 22, 13, 12, 11, 10]
        lows    = [ 8,  9, 10, 11, 18, 11, 10,  9, 18, 11, 10,  9,  8]
        closes  = [ 9, 10, 11, 12, 19, 12, 11, 10, 19,  12, 11, 10,  9]
        result  = self._full_pipeline(highs, lows, closes)
        assert result["sweep_bear_level"].iloc[8] == pytest.approx(20.0)
        assert result["sweep_bear_type"].iloc[8] == "swing_high"

    def test_bullish_sweep_swing_low_in_full_pipeline(self):
        """Swing low at bar 4 (low=5) confirmed at bar 6.
        Bar 8: low=3 (wick below), close=6 (back above) → bull sweep."""
        highs   = [20, 19, 18, 17, 10, 17, 18, 19,  7, 17, 18, 19, 20]
        lows    = [18, 17, 16, 15,  5, 15, 16, 17,  3, 15, 16, 17, 18]
        closes  = [19, 18, 17, 16,  6, 16, 17, 18,  6, 16, 17, 18, 19]
        result  = self._full_pipeline(highs, lows, closes)
        assert result["sweep_bull_level"].iloc[8] == pytest.approx(5.0)
        assert result["sweep_bull_type"].iloc[8] == "swing_low"

    def test_no_sweep_in_flat_market(self):
        n = 20
        result = self._full_pipeline([10.0] * n, [8.0] * n, [9.0] * n)
        assert result["sweep_bear_level"].isna().all()
        assert result["sweep_bull_level"].isna().all()
