"""
Tests for asia_range_liquidity.py — Phase 2.9.

Critical properties:
  - Asia session default: 00:00–09:00 UTC.
  - asia_high / asia_low appear only AFTER the session ends (no look-ahead).
  - Bars inside the session have in_asia_session=True, asia_high/low=NaN.
  - asia_trade_allowed is always False (observation only).
  - Days with no Asia bars produce no range.
  - Works across multiple days.
"""

import pytest
import numpy as np
import pandas as pd

from core.smc.asia_range_liquidity import (
    AsiaRangeLiquidity,
    detect_asia_range,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_df(start, periods, freq="1h", tz="UTC", high=100.0, low=99.0):
    """Create a simple OHLC DataFrame."""
    idx = pd.date_range(start, periods=periods, freq=freq, tz=tz, name="timestamp")
    return pd.DataFrame(
        {
            "open":   [99.5] * periods,
            "high":   [high] * periods,
            "low":    [low]  * periods,
            "close":  [99.5] * periods,
            "volume": [100]  * periods,
        },
        index=idx,
    )


def _set_bar(df, pos, h, l):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("high")] = float(h)
    df.iloc[pos, df.columns.get_loc("low")]  = float(l)
    return df


def _ar(**kwargs):
    return AsiaRangeLiquidity(**kwargs)


# ------------------------------------------------------------------ #
# Basic Asia range detection                                           #
# ------------------------------------------------------------------ #

class TestBasicRange:
    def test_asia_high_low_set_after_session(self):
        """Asia bars 00:00-08:00, first post-Asia bar at 09:00 gets the range."""
        df = _make_df("2026-01-05 00:00", 24, freq="1h", tz="UTC")
        # Set specific highs/lows during Asia hours
        df = _set_bar(df, 3, h=105.0, l=99.0)   # 03:00 — high
        df = _set_bar(df, 6, h=100.0, l=95.0)   # 06:00 — low
        result = _ar().detect(df)

        # Bar at 09:00 (index 9) should have asia_high=105, asia_low=95
        assert result["asia_high"].iloc[9] == pytest.approx(105.0)
        assert result["asia_low"].iloc[9] == pytest.approx(95.0)
        assert result["asia_range_set"].iloc[9] == True

    def test_asia_bars_have_no_range(self):
        """Bars inside Asia session should NOT have the range set."""
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        result = _ar().detect(df)
        for i in range(9):  # hours 0-8
            assert np.isnan(result["asia_high"].iloc[i])
            assert np.isnan(result["asia_low"].iloc[i])
            assert result["asia_range_set"].iloc[i] == False

    def test_in_asia_session_flag(self):
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        result = _ar().detect(df)
        for i in range(9):  # 00:00-08:00 inclusive
            assert result["in_asia_session"].iloc[i] == True
        for i in range(9, 24):  # 09:00-23:00
            assert result["in_asia_session"].iloc[i] == False

    def test_range_propagates_rest_of_day(self):
        """All post-Asia bars on the same day get the range."""
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        df = _set_bar(df, 5, h=110.0, l=90.0)
        result = _ar().detect(df)
        for i in range(9, 24):
            assert result["asia_high"].iloc[i] == pytest.approx(110.0)
            assert result["asia_low"].iloc[i] == pytest.approx(90.0)
            assert result["asia_range_set"].iloc[i] == True

    def test_range_does_not_leak_to_next_day(self):
        """Day 1 range should NOT appear on day 2's Asia bars."""
        df = _make_df("2026-01-05 00:00", 48, freq="1h")
        df = _set_bar(df, 5, h=110.0, l=90.0)  # day 1 Asia
        result = _ar().detect(df)
        # Day 2 starts at index 24 (2026-01-06 00:00)
        # Day 2 Asia bars (24-32) should not have day 1's range
        assert np.isnan(result["asia_high"].iloc[24])
        assert result["in_asia_session"].iloc[24] == True


# ------------------------------------------------------------------ #
# No look-ahead                                                        #
# ------------------------------------------------------------------ #

class TestNoLookAhead:
    def test_range_not_available_during_session(self):
        """Even if a bar at 08:00 is the last Asia bar, 08:00 itself has no range."""
        df = _make_df("2026-01-05 00:00", 12, freq="1h")
        result = _ar().detect(df)
        assert np.isnan(result["asia_high"].iloc[8])   # 08:00 = last Asia bar
        assert result["asia_range_set"].iloc[8] == False
        assert not np.isnan(result["asia_high"].iloc[9])  # 09:00 = first post-Asia

    def test_highest_at_last_asia_bar(self):
        """High at 08:00 (last Asia bar). Range should include it at 09:00."""
        df = _make_df("2026-01-05 00:00", 12, freq="1h")
        df = _set_bar(df, 8, h=120.0, l=99.0)  # 08:00
        result = _ar().detect(df)
        assert result["asia_high"].iloc[9] == pytest.approx(120.0)


# ------------------------------------------------------------------ #
# trade_allowed is always False                                        #
# ------------------------------------------------------------------ #

class TestTradeAllowed:
    def test_always_false(self):
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        result = _ar().detect(df)
        assert (result["asia_trade_allowed"] == False).all()


# ------------------------------------------------------------------ #
# Multiple days                                                        #
# ------------------------------------------------------------------ #

class TestMultipleDays:
    def test_each_day_gets_own_range(self):
        df = _make_df("2026-01-05 00:00", 48, freq="1h")
        # Day 1: high at 03:00
        df = _set_bar(df, 3, h=105.0, l=95.0)
        # Day 2: high at 27:00 = 2026-01-06 03:00
        df = _set_bar(df, 27, h=115.0, l=85.0)
        result = _ar().detect(df)

        # Day 1 post-Asia (09:00 = idx 9)
        assert result["asia_high"].iloc[9] == pytest.approx(105.0)
        assert result["asia_low"].iloc[9] == pytest.approx(95.0)

        # Day 2 post-Asia (33:00 = idx 33 = 2026-01-06 09:00)
        assert result["asia_high"].iloc[33] == pytest.approx(115.0)
        assert result["asia_low"].iloc[33] == pytest.approx(85.0)

    def test_weekend_day_no_data_no_range(self):
        """Only weekday data → no range on missing days."""
        # Monday only (24 bars)
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        result = _ar().detect(df)
        ranges = _ar().get_asia_range(result)
        assert len(ranges) == 1


# ------------------------------------------------------------------ #
# Sub-hourly timeframes                                                #
# ------------------------------------------------------------------ #

class TestSubHourly:
    def test_5min_bars_asia_range(self):
        """5-minute bars: Asia = 00:00-08:55, first post-Asia at 09:00."""
        # 24h * 12 bars/h = 288 bars
        df = _make_df("2026-01-05 00:00", 288, freq="5min")
        # Set high at 04:30 (bar index = 4*12 + 6 = 54)
        df = _set_bar(df, 54, h=108.0, l=99.0)
        # Set low at 07:15 (bar index = 7*12 + 3 = 87)
        df = _set_bar(df, 87, h=100.0, l=92.0)
        result = _ar().detect(df)

        # First post-Asia bar: 09:00 = bar 108
        assert result["asia_range_set"].iloc[108] == True
        assert result["asia_high"].iloc[108] == pytest.approx(108.0)
        assert result["asia_low"].iloc[108] == pytest.approx(92.0)

        # Last Asia bar (08:55 = bar 107) should NOT have range
        assert result["asia_range_set"].iloc[107] == False


# ------------------------------------------------------------------ #
# Custom session hours                                                 #
# ------------------------------------------------------------------ #

class TestCustomHours:
    def test_custom_session_23_to_8(self):
        """Asia start=23, end=8 → wraps midnight."""
        # Need 2 days to cover the wrap
        df = _make_df("2026-01-05 00:00", 48, freq="1h")
        # Bar at 23:00 on day 1 = idx 23
        df = _set_bar(df, 23, h=112.0, l=99.0)
        result = _ar(asia_start_hour=23, asia_end_hour=8).detect(df)

        # 23:00 on day 1 should be in_asia
        assert result["in_asia_session"].iloc[23] == True
        # 00:00-07:00 on day 2 (idx 24-31) should be in_asia
        for i in range(24, 32):
            assert result["in_asia_session"].iloc[i] == True
        # 08:00 on day 2 (idx 32) should NOT be in_asia
        assert result["in_asia_session"].iloc[32] == False

    def test_custom_start_2_end_7(self):
        """Narrower Asia: 02:00–07:00 UTC."""
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        df = _set_bar(df, 4, h=107.0, l=93.0)  # 04:00 in range
        result = _ar(asia_start_hour=2, asia_end_hour=7).detect(df)

        # Bars 0,1 (00:00, 01:00) not in session
        assert result["in_asia_session"].iloc[0] == False
        assert result["in_asia_session"].iloc[1] == False
        # Bars 2-6 (02:00-06:00) in session
        for i in range(2, 7):
            assert result["in_asia_session"].iloc[i] == True
        # Bar 7 (07:00) = first post-Asia
        assert result["in_asia_session"].iloc[7] == False
        assert result["asia_range_set"].iloc[7] == True
        assert result["asia_high"].iloc[7] == pytest.approx(107.0)
        assert result["asia_low"].iloc[7] == pytest.approx(93.0)


# ------------------------------------------------------------------ #
# Edge cases                                                           #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_empty_df(self):
        df = _make_df("2026-01-05", 0)
        result = _ar().detect(df)
        assert len(result) == 0
        assert "asia_high" in result.columns

    def test_only_asia_bars_no_post_session(self):
        """Data ends before Asia session ends → no range set."""
        df = _make_df("2026-01-05 00:00", 5, freq="1h")  # 00:00-04:00
        result = _ar().detect(df)
        assert (result["asia_range_set"] == False).all()
        assert result["asia_high"].isna().all()

    def test_single_asia_bar(self):
        """Only 1 bar in Asia then data continues → range is that one bar."""
        df = _make_df("2026-01-05 08:00", 4, freq="1h")  # 08:00-11:00
        df = _set_bar(df, 0, h=105.0, l=95.0)  # 08:00 = last Asia bar
        result = _ar().detect(df)
        # Bar at 09:00 (idx 1) should get range from the single Asia bar
        assert result["asia_high"].iloc[1] == pytest.approx(105.0)
        assert result["asia_low"].iloc[1] == pytest.approx(95.0)

    def test_timezone_conversion(self):
        """Non-UTC timezone input is handled correctly."""
        # Israel time = UTC+2 (winter). 02:00 IST = 00:00 UTC
        df = _make_df("2026-01-05 02:00", 24, freq="1h", tz="Asia/Jerusalem")
        # Bar 0 = 02:00 IST = 00:00 UTC → in Asia
        # Bar 3 = 05:00 IST = 03:00 UTC → in Asia
        df = _set_bar(df, 3, h=108.0, l=92.0)
        result = _ar().detect(df)
        assert result["in_asia_session"].iloc[0] == True
        assert result["in_asia_session"].iloc[3] == True
        # Bar 9 = 11:00 IST = 09:00 UTC → NOT in Asia (first post)
        assert result["in_asia_session"].iloc[9] == False
        assert result["asia_range_set"].iloc[9] == True


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        result = _ar().detect(df)
        for col in ["asia_high", "asia_low", "asia_range_set",
                     "in_asia_session", "asia_trade_allowed"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        pd.testing.assert_index_equal(_ar().detect(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        _ar().detect(df)
        assert "asia_high" not in df.columns


# ------------------------------------------------------------------ #
# Validation                                                            #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_non_datetime_index_raises(self):
        df = _make_df("2026-01-05", 10)
        df.index = range(10)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _ar().detect(df)

    def test_missing_columns_raises(self):
        df = _make_df("2026-01-05", 10).drop(columns=["high"])
        with pytest.raises(ValueError, match="Missing"):
            _ar().detect(df)

    def test_same_start_end_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            AsiaRangeLiquidity(asia_start_hour=5, asia_end_hour=5)

    def test_invalid_start_hour_raises(self):
        with pytest.raises(ValueError):
            AsiaRangeLiquidity(asia_start_hour=25)

    def test_invalid_end_hour_raises(self):
        with pytest.raises(ValueError):
            AsiaRangeLiquidity(asia_end_hour=-1)

    def test_convenience_function_works(self):
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        result = detect_asia_range(df)
        assert "asia_high" in result.columns


# ------------------------------------------------------------------ #
# Accessor                                                             #
# ------------------------------------------------------------------ #

class TestAccessor:
    def test_get_asia_range_returns_per_day(self):
        df = _make_df("2026-01-05 00:00", 48, freq="1h")
        df = _set_bar(df, 3, h=105.0, l=95.0)
        df = _set_bar(df, 27, h=110.0, l=90.0)
        result = _ar().detect(df)
        ranges = _ar().get_asia_range(result)
        assert len(ranges) == 2
        # Newest first
        assert ranges[0]["date"] > ranges[1]["date"]

    def test_empty_when_no_range(self):
        df = _make_df("2026-01-05 00:00", 5, freq="1h")  # only Asia bars
        result = _ar().detect(df)
        assert _ar().get_asia_range(result) == []
