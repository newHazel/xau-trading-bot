"""
Tests for liquidity_detector.py — Phase 2.1.

Critical properties:
  EQH/EQL
    - Two confirmed swings within tolerance form an equal-level cluster.
    - Cluster level = running average of all matching swings.
    - Count = number of swings in the cluster.
    - Outside tolerance → no cluster.
    - Once formed, the level is carried forward on every subsequent bar
      until replaced by a new cluster.
    - Only confirmed swings are used (no look-ahead).

  PDH/PDL
    - Computed per UTC calendar day.
    - PDH/PDL for day N = high/low of day N-1.
    - First day has no previous day → NaN.
    - Future-day data never affects past bars.
"""

import pytest
import numpy as np
import pandas as pd

from core.structure.swing_detector import SwingDetector
from core.smc.liquidity_detector import LiquidityDetector, detect_liquidity


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


def _make_swing_df(n=20, start="2026-01-05 10:00", freq="5min"):
    """Minimal DataFrame with swing columns (NaN/−1) — closes match midpoint."""
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
        },
        index=idx,
    )


def _set_sh(df, pos, price):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("swing_high")]     = float(price)
    df.iloc[pos, df.columns.get_loc("swing_high_idx")] = max(0, pos - 2)
    df.iloc[pos, df.columns.get_loc("high")]           = float(price)
    return df


def _set_sl(df, pos, price):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("swing_low")]     = float(price)
    df.iloc[pos, df.columns.get_loc("swing_low_idx")] = max(0, pos - 2)
    df.iloc[pos, df.columns.get_loc("low")]           = float(price)
    return df


def _ld(tol_pct=None):
    return LiquidityDetector(tol_pct)


def _full_pipeline(highs, lows, tf="5m", freq="5min", tol_pct=None):
    df = _make_raw_df(highs, lows, freq=freq)
    with_swings = SwingDetector().detect(df, tf)
    return LiquidityDetector(tol_pct).detect(with_swings)


# ------------------------------------------------------------------ #
# EQH detection                                                        #
# ------------------------------------------------------------------ #

class TestEQHDetection:
    def test_two_equal_highs_form_eqh(self):
        """Two swing highs within tolerance → eqh_level set, count=2."""
        df = _make_swing_df()
        df = _set_sh(df, 4, 100.00)
        df = _set_sh(df, 12, 100.05)   # within 0.1% of 100 (= 0.1)
        result = _ld().detect(df)
        assert not np.isnan(result["eqh_level"].iloc[12])
        assert result["eqh_count"].iloc[12] == 2

    def test_eqh_level_is_average(self):
        df = _make_swing_df()
        df = _set_sh(df, 4, 100.00)
        df = _set_sh(df, 12, 100.04)
        result = _ld().detect(df)
        assert result["eqh_level"].iloc[12] == pytest.approx(100.02)

    def test_two_unequal_highs_no_eqh(self):
        """Two swings far apart → no EQH cluster."""
        df = _make_swing_df()
        df = _set_sh(df, 4, 100.00)
        df = _set_sh(df, 12, 110.00)    # 10% away — way outside tol
        result = _ld().detect(df)
        assert np.isnan(result["eqh_level"].iloc[12])
        assert result["eqh_count"].iloc[12] == 0

    def test_three_equal_highs_count_3(self):
        df = _make_swing_df(n=25)
        df = _set_sh(df, 3, 100.00)
        df = _set_sh(df, 10, 100.05)
        df = _set_sh(df, 17, 99.96)
        result = _ld().detect(df)
        assert result["eqh_count"].iloc[17] == 3

    def test_eqh_level_average_of_three(self):
        # All three swings must match the LATEST swing's tolerance window:
        # at 99.96, tol = 0.09996 → 100.00 (diff 0.04) and 100.04 (diff 0.08) both match.
        df = _make_swing_df(n=25)
        df = _set_sh(df, 3, 100.00)
        df = _set_sh(df, 10, 100.04)
        df = _set_sh(df, 17, 99.96)
        result = _ld().detect(df)
        # average = (100 + 100.04 + 99.96) / 3 = 100.0
        assert result["eqh_level"].iloc[17] == pytest.approx(100.0)

    def test_first_swing_high_has_no_eqh(self):
        """A single swing cannot form an EQH (need ≥ 2)."""
        df = _make_swing_df()
        df = _set_sh(df, 4, 100.00)
        result = _ld().detect(df)
        assert np.isnan(result["eqh_level"].iloc[4])
        assert result["eqh_count"].iloc[4] == 0

    def test_eqh_carried_forward_after_formation(self):
        """Once EQH forms at bar X, every bar ≥ X has the same level."""
        df = _make_swing_df()
        df = _set_sh(df, 4, 100.00)
        df = _set_sh(df, 10, 100.04)
        result = _ld().detect(df)
        eqh_at_10 = result["eqh_level"].iloc[10]
        for i in range(10, len(df)):
            assert result["eqh_level"].iloc[i] == pytest.approx(eqh_at_10), f"bar {i}"

    def test_tolerance_boundary_inside(self):
        """Exactly at tolerance boundary → considered equal (≤, not <)."""
        df = _make_swing_df()
        df = _set_sh(df, 4, 100.00)
        df = _set_sh(df, 12, 100.10)   # diff = 0.10 = exactly 0.1% of 100
        result = _ld().detect(df)
        assert result["eqh_count"].iloc[12] == 2

    def test_tolerance_boundary_outside(self):
        df = _make_swing_df()
        df = _set_sh(df, 4, 100.00)
        df = _set_sh(df, 12, 100.11)   # diff = 0.11 > 0.1% of 100
        result = _ld().detect(df)
        assert result["eqh_count"].iloc[12] == 0

    def test_custom_tolerance_widens_match(self):
        df = _make_swing_df()
        df = _set_sh(df, 4, 100.00)
        df = _set_sh(df, 12, 100.50)   # diff = 0.5; default tol=0.1% rejects
        result_default = _ld().detect(df)
        assert result_default["eqh_count"].iloc[12] == 0

        result_loose = _ld(tol_pct=1.0).detect(df)   # 1% → 1.0 absolute
        assert result_loose["eqh_count"].iloc[12] == 2


# ------------------------------------------------------------------ #
# EQL detection (mirror of EQH)                                        #
# ------------------------------------------------------------------ #

class TestEQLDetection:
    def test_two_equal_lows_form_eql(self):
        df = _make_swing_df()
        df = _set_sl(df, 4, 50.00)
        df = _set_sl(df, 12, 50.03)
        result = _ld().detect(df)
        assert result["eql_count"].iloc[12] == 2

    def test_eql_level_is_average(self):
        df = _make_swing_df()
        df = _set_sl(df, 4, 50.00)
        df = _set_sl(df, 12, 50.02)
        result = _ld().detect(df)
        assert result["eql_level"].iloc[12] == pytest.approx(50.01)

    def test_unequal_lows_no_eql(self):
        df = _make_swing_df()
        df = _set_sl(df, 4, 50.00)
        df = _set_sl(df, 12, 60.00)
        result = _ld().detect(df)
        assert np.isnan(result["eql_level"].iloc[12])

    def test_eql_independent_of_eqh(self):
        """EQH and EQL track separately; presence of one doesn't trigger the other."""
        df = _make_swing_df()
        df = _set_sh(df, 4, 100.00)
        df = _set_sh(df, 12, 100.05)   # forms EQH
        # No swing lows at all → no EQL
        result = _ld().detect(df)
        assert result["eqh_count"].iloc[12] == 2
        assert result["eql_count"].iloc[12] == 0
        assert np.isnan(result["eql_level"].iloc[12])


# ------------------------------------------------------------------ #
# PDH / PDL                                                            #
# ------------------------------------------------------------------ #

class TestPDHPDL:
    def _two_day_df(self):
        """Day 1 (2026-01-05): high=100, low=80. Day 2 (2026-01-06): high=120, low=70."""
        idx = pd.date_range("2026-01-05 00:00", periods=48, freq="1h",
                            tz="UTC", name="timestamp")
        n = 48
        highs = [90.0] * n
        lows  = [85.0] * n
        # Day 1 = bars 0-23, day 2 = bars 24-47
        highs[10] = 100.0      # day 1 high
        lows[15]  = 80.0       # day 1 low
        highs[30] = 120.0      # day 2 high
        lows[35]  = 70.0       # day 2 low
        df = pd.DataFrame(
            {
                "open":           highs,
                "high":           highs,
                "low":            lows,
                "close":          lows,
                "volume":         [100.0] * n,
                "swing_high":     [np.nan] * n,
                "swing_low":      [np.nan] * n,
                "swing_high_idx": [-1] * n,
                "swing_low_idx":  [-1] * n,
            },
            index=idx,
        )
        return df

    def test_pdh_set_to_previous_day_high(self):
        df = self._two_day_df()
        result = _ld().detect(df)
        # Bar at 2026-01-06 12:00 → pdh should be day 1 high = 100
        assert result["pdh"].iloc[24] == pytest.approx(100.0)
        assert result["pdh"].iloc[36] == pytest.approx(100.0)

    def test_pdl_set_to_previous_day_low(self):
        df = self._two_day_df()
        result = _ld().detect(df)
        assert result["pdl"].iloc[24] == pytest.approx(80.0)
        assert result["pdl"].iloc[36] == pytest.approx(80.0)

    def test_first_day_pdh_is_nan(self):
        df = self._two_day_df()
        result = _ld().detect(df)
        for i in range(24):                 # all of day 1
            assert np.isnan(result["pdh"].iloc[i]), f"bar {i}"
            assert np.isnan(result["pdl"].iloc[i]), f"bar {i}"

    def test_pdh_constant_throughout_day(self):
        df = self._two_day_df()
        result = _ld().detect(df)
        # All of day 2 should have the same PDH
        day2_pdh = result["pdh"].iloc[24:48].dropna().unique()
        assert len(day2_pdh) == 1

    def test_pdh_changes_across_three_days(self):
        """Day 3 should reflect day 2 H/L, not day 1."""
        idx = pd.date_range("2026-01-05 00:00", periods=72, freq="1h",
                            tz="UTC", name="timestamp")
        n = 72
        highs = [90.0] * n
        lows  = [85.0] * n
        highs[10] = 100.0       # day 1 high
        lows[15]  = 80.0
        highs[30] = 120.0       # day 2 high
        lows[35]  = 70.0
        highs[55] = 130.0       # day 3 high (irrelevant for this test)
        df = pd.DataFrame(
            {
                "open":           highs,
                "high":           highs,
                "low":            lows,
                "close":          lows,
                "volume":         [100.0] * n,
                "swing_high":     [np.nan] * n,
                "swing_low":      [np.nan] * n,
                "swing_high_idx": [-1] * n,
                "swing_low_idx":  [-1] * n,
            },
            index=idx,
        )
        result = _ld().detect(df)
        # Day 3 (bars 48-71) should have pdh = day 2 high = 120
        assert result["pdh"].iloc[48] == pytest.approx(120.0)
        assert result["pdl"].iloc[48] == pytest.approx(70.0)


# ------------------------------------------------------------------ #
# No look-ahead                                                        #
# ------------------------------------------------------------------ #

class TestNoLookahead:
    def test_eqh_only_uses_swings_at_or_before_bar(self):
        """A swing at bar 15 must not contribute to EQH at bar 10."""
        df = _make_swing_df(n=20)
        df = _set_sh(df, 4, 100.00)
        df = _set_sh(df, 15, 100.05)
        result = _ld().detect(df)
        # Bars 4..14 see only the first swing → no EQH
        for i in range(4, 15):
            assert np.isnan(result["eqh_level"].iloc[i]), f"bar {i}"
        # Bar 15+ → EQH active
        assert not np.isnan(result["eqh_level"].iloc[15])

    def test_pdh_does_not_use_future_day(self):
        idx = pd.date_range("2026-01-05 00:00", periods=48, freq="1h",
                            tz="UTC", name="timestamp")
        n = 48
        highs = [90.0] * n
        lows  = [85.0] * n
        highs[30] = 200.0       # day 2 has a much higher high
        df = pd.DataFrame(
            {
                "open": highs, "high": highs, "low": lows, "close": lows,
                "volume": [100.0] * n,
                "swing_high": [np.nan] * n, "swing_low": [np.nan] * n,
                "swing_high_idx": [-1] * n, "swing_low_idx": [-1] * n,
            },
            index=idx,
        )
        result = _ld().detect(df)
        # Day 1 should have NaN PDH — must NOT be 200
        assert np.isnan(result["pdh"].iloc[10])

    def test_eqh_carry_forward_does_not_invent_future_data(self):
        """Before any EQH forms, eqh_level must be NaN even if future swings will match."""
        df = _make_swing_df(n=20)
        df = _set_sh(df, 10, 100.00)
        df = _set_sh(df, 15, 100.05)
        result = _ld().detect(df)
        for i in range(10):                 # before first swing
            assert np.isnan(result["eqh_level"].iloc[i]), f"bar {i}"


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _make_swing_df()
        result = _ld().detect(df)
        for col in ["eqh_level", "eqh_count", "eql_level", "eql_count", "pdh", "pdl"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _make_swing_df()
        pd.testing.assert_index_equal(_ld().detect(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _make_swing_df()
        _ld().detect(df)
        assert "eqh_level" not in df.columns
        assert "pdh" not in df.columns

    def test_eqh_count_zero_default(self):
        df = _make_swing_df()
        result = _ld().detect(df)
        assert (result["eqh_count"] == 0).all()
        assert (result["eql_count"] == 0).all()

    def test_existing_columns_preserved(self):
        df = _make_swing_df()
        result = _ld().detect(df)
        for col in ["high", "low", "swing_high", "swing_low"]:
            assert col in result.columns


# ------------------------------------------------------------------ #
# Accessor methods                                                      #
# ------------------------------------------------------------------ #

class TestAccessors:
    def _build_eqh(self):
        df = _make_swing_df()
        df = _set_sh(df, 4, 100.00)
        df = _set_sh(df, 12, 100.04)
        return _ld().detect(df)

    def test_get_active_eqh_returns_dict(self):
        result = self._build_eqh()
        info = _ld().get_active_eqh(result)
        assert isinstance(info, dict)

    def test_get_active_eqh_level(self):
        result = self._build_eqh()
        info = _ld().get_active_eqh(result)
        assert info["level"] == pytest.approx(100.02)

    def test_get_active_eqh_count(self):
        result = self._build_eqh()
        info = _ld().get_active_eqh(result)
        assert info["count"] == 2

    def test_get_active_eqh_returns_none_when_empty(self):
        df = _make_swing_df()
        result = _ld().detect(df)
        assert _ld().get_active_eqh(result) is None

    def test_get_active_eql_returns_none_when_empty(self):
        df = _make_swing_df()
        result = _ld().detect(df)
        assert _ld().get_active_eql(result) is None


# ------------------------------------------------------------------ #
# Validation                                                           #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_missing_swing_columns_raises(self):
        idx = pd.date_range("2026-01-05", periods=5, freq="5min", tz="UTC")
        df = pd.DataFrame({"high": [10] * 5, "low": [8] * 5}, index=idx)
        with pytest.raises(ValueError, match="Missing columns"):
            _ld().detect(df)

    def test_missing_high_low_raises(self):
        df = _make_swing_df().drop(columns=["high"])
        with pytest.raises(ValueError, match="Missing columns"):
            _ld().detect(df)

    def test_non_datetime_index_raises(self):
        df = _make_swing_df()
        df.index = range(len(df))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _ld().detect(df)

    def test_zero_tolerance_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            LiquidityDetector(eqh_tolerance_pct=0)

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            LiquidityDetector(eqh_tolerance_pct=-0.1)

    def test_convenience_function_works(self):
        df = _make_swing_df()
        result = detect_liquidity(df)
        assert "eqh_level" in result.columns


# ------------------------------------------------------------------ #
# Integration                                                          #
# ------------------------------------------------------------------ #

class TestIntegration:
    def test_eqh_in_full_pipeline(self):
        """
        window=5, lag=2.
        Bar 4 (high=25.00) confirmed at bar 6.
        Bar 12 (high=25.02) confirmed at bar 14.
        Both within 0.1% of 25 (tolerance = 0.025).
        """
        highs = [10, 11, 12, 13, 25.00, 13, 12, 11, 10, 11, 12, 13, 25.02, 13, 12, 11, 10]
        lows  = [ 8,  9, 10, 11, 23,    11, 10,  9,  8,  9, 10, 11, 23,    11, 10,  9,  8]
        result = _full_pipeline(highs, lows)
        assert result["eqh_count"].iloc[14] == 2
        assert result["eqh_level"].iloc[14] == pytest.approx(25.01)

    def test_no_eqh_when_swings_unequal_in_pipeline(self):
        highs = [10, 11, 12, 13, 25, 13, 12, 11, 10, 11, 12, 13, 30, 13, 12, 11, 10]
        lows  = [ 8,  9, 10, 11, 23, 11, 10,  9,  8,  9, 10, 11, 28, 11, 10,  9,  8]
        result = _full_pipeline(highs, lows)
        assert (result["eqh_count"] == 0).all()

    def test_eql_in_full_pipeline(self):
        """Two swing lows ~25 confirmed at bars 6 and 14.
        At price 25, default tol = 0.025; diff 0.02 falls inside."""
        highs = [40, 38, 36, 34, 30,    34, 36, 38, 40, 38, 36, 34, 30.02, 34, 36, 38, 40]
        lows  = [38, 36, 34, 32, 25.00, 32, 34, 36, 38, 36, 34, 32, 25.02, 32, 34, 36, 38]
        result = _full_pipeline(highs, lows)
        assert result["eql_count"].iloc[14] == 2
        assert result["eql_level"].iloc[14] == pytest.approx(25.01)

    def test_pdh_in_pipeline_across_two_days(self):
        """1H bars across 48 hours → day 2 has PDH from day 1."""
        # Day 1 high = 100, day 2 high = 120
        n = 48
        highs = [90.0] * n
        lows  = [85.0] * n
        highs[10] = 100.0        # day 1 high
        highs[30] = 120.0        # day 2 high (irrelevant for day 2 PDH)
        idx = pd.date_range("2026-01-05 00:00", periods=n, freq="1h",
                            tz="UTC", name="timestamp")
        mid = [(h + l) / 2 for h, l in zip(highs, lows)]
        df = pd.DataFrame(
            {"open": mid, "high": highs, "low": lows, "close": mid, "volume": [100.] * n},
            index=idx,
        )
        with_swings = SwingDetector().detect(df, "1h")
        result = LiquidityDetector().detect(with_swings)
        # Day 2 bars (index 24+) should have PDH = day 1 high = 100
        assert result["pdh"].iloc[24] == pytest.approx(100.0)

    def test_flat_series_no_liquidity(self):
        highs = [10.0] * 20
        lows  = [8.0]  * 20
        result = _full_pipeline(highs, lows)
        # Flat data means no swings → no EQH/EQL
        assert (result["eqh_count"] == 0).all()
        assert (result["eql_count"] == 0).all()
