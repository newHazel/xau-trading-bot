"""
Tests for fix_levels_tracker.py — Phase 2.11.

Critical properties:
  - Fix times in Israel time: Shanghai PM 04:30, London AM 12:30, London PM 17:00.
  - Fix level = close of the bar at or just before the fix time.
  - Levels appear only at or after the fix time (no look-ahead).
  - Levels persist for the rest of that Israel-time calendar day.
  - fix_level_count counts how many of the 3 fixes are set at each bar.
  - DST handled via Asia/Jerusalem timezone.
"""

import pytest
import numpy as np
import pandas as pd

from core.smc.fix_levels_tracker import (
    FixLevelsTracker,
    track_fix_levels,
    DEFAULT_FIX_TIMES,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_df(start, periods, freq="1h", tz="UTC", close=2000.0):
    idx = pd.date_range(start, periods=periods, freq=freq, tz=tz, name="timestamp")
    return pd.DataFrame(
        {
            "open":   [close] * periods,
            "high":   [close + 5] * periods,
            "low":    [close - 5] * periods,
            "close":  [close] * periods,
            "volume": [100]   * periods,
        },
        index=idx,
    )


def _set_close(df, pos, close_price):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("close")] = float(close_price)
    return df


def _ft(**kwargs):
    return FixLevelsTracker(**kwargs)


# ------------------------------------------------------------------ #
# Shanghai PM Fix (04:30 Israel → ~01:30 UTC winter)                   #
# ------------------------------------------------------------------ #

class TestShanghaiPM:
    def test_fix_level_set_at_fix_time(self):
        """Bar at 01:00 UTC (03:00 IST winter) → before fix.
        Bar at 02:00 UTC (04:00 IST) → still before 04:30.
        But hourly bars: 04:00 IST is closest at-or-before 04:30."""
        # Use Israel time directly for clarity
        df = _make_df("2026-01-05 00:00", 24, freq="1h", tz="Asia/Jerusalem")
        df = _set_close(df, 4, 2050.0)  # 04:00 IST → at or before 04:30
        result = _ft().track(df)

        # Bar at 04:00 (idx 4) is before fix → no fix yet
        assert np.isnan(result["fix_shanghai_pm"].iloc[4])
        # Bar at 05:00 (idx 5) is after 04:30 → fix should be set
        assert result["fix_shanghai_pm"].iloc[5] == pytest.approx(2050.0)

    def test_fix_not_set_before_time(self):
        df = _make_df("2026-01-05 00:00", 5, freq="1h", tz="Asia/Jerusalem")
        result = _ft().track(df)
        # All bars 00:00-04:00 are before 04:30
        for i in range(5):
            assert np.isnan(result["fix_shanghai_pm"].iloc[i])


# ------------------------------------------------------------------ #
# London AM Fix (12:30 Israel)                                         #
# ------------------------------------------------------------------ #

class TestLondonAM:
    def test_london_am_fix(self):
        df = _make_df("2026-01-05 10:00", 8, freq="1h", tz="Asia/Jerusalem")
        df = _set_close(df, 2, 2100.0)  # 12:00 IST → at or before 12:30
        result = _ft().track(df)

        # 12:00 (idx 2) is before 12:30 → no fix
        assert np.isnan(result["fix_london_am"].iloc[2])
        # 13:00 (idx 3) is after 12:30 → fix set
        assert result["fix_london_am"].iloc[3] == pytest.approx(2100.0)

    def test_london_am_propagates_to_later_bars(self):
        df = _make_df("2026-01-05 10:00", 12, freq="1h", tz="Asia/Jerusalem")
        df = _set_close(df, 2, 2100.0)  # 12:00
        result = _ft().track(df)
        for i in range(3, 12):  # 13:00 - 21:00
            assert result["fix_london_am"].iloc[i] == pytest.approx(2100.0)


# ------------------------------------------------------------------ #
# London PM Fix (17:00 Israel)                                         #
# ------------------------------------------------------------------ #

class TestLondonPM:
    def test_london_pm_fix(self):
        df = _make_df("2026-01-05 14:00", 8, freq="1h", tz="Asia/Jerusalem")
        df = _set_close(df, 3, 2150.0)  # 17:00 IST → at fix time
        result = _ft().track(df)

        # 17:00 (idx 3): bar_min=1020 >= fix_min=1020 → fix is set HERE
        assert result["fix_london_pm"].iloc[3] == pytest.approx(2150.0)


# ------------------------------------------------------------------ #
# No look-ahead                                                        #
# ------------------------------------------------------------------ #

class TestNoLookAhead:
    def test_fix_level_uses_close_at_or_before(self):
        """With 30min bars: bar at 04:00 and 04:30 IST.
        04:30 bar is at fix time → its close is the fix level."""
        df = _make_df("2026-01-05 03:00", 6, freq="30min", tz="Asia/Jerusalem")
        # idx 0=03:00, 1=03:30, 2=04:00, 3=04:30, 4=05:00, 5=05:30
        df = _set_close(df, 2, 2020.0)  # 04:00
        df = _set_close(df, 3, 2030.0)  # 04:30 — this is AT fix time
        result = _ft().track(df)

        # 04:30 (idx 3) is exactly at fix time → fix level = close at 04:30
        assert result["fix_shanghai_pm"].iloc[3] == pytest.approx(2030.0)
        # Bars before 04:30 should NOT have the fix
        assert np.isnan(result["fix_shanghai_pm"].iloc[2])

    def test_bars_before_fix_have_nan(self):
        df = _make_df("2026-01-05 00:00", 4, freq="1h", tz="Asia/Jerusalem")
        result = _ft().track(df)
        for i in range(4):
            assert np.isnan(result["fix_shanghai_pm"].iloc[i])


# ------------------------------------------------------------------ #
# fix_level_count                                                      #
# ------------------------------------------------------------------ #

class TestFixLevelCount:
    def test_count_increments_through_day(self):
        """Full day: 0 fixes early, 1 after Shanghai, 2 after London AM, 3 after London PM."""
        df = _make_df("2026-01-05 00:00", 24, freq="1h", tz="Asia/Jerusalem")
        result = _ft().track(df)

        # Before Shanghai (04:30): count=0
        assert result["fix_level_count"].iloc[0] == 0  # 00:00
        assert result["fix_level_count"].iloc[4] == 0  # 04:00

        # After Shanghai (05:00): count=1
        assert result["fix_level_count"].iloc[5] == 1

        # After London AM (13:00): count=2
        assert result["fix_level_count"].iloc[13] == 2

        # After London PM (17:00): count=3
        assert result["fix_level_count"].iloc[17] == 3

    def test_count_zero_when_no_fixes(self):
        df = _make_df("2026-01-05 00:00", 4, freq="1h", tz="Asia/Jerusalem")
        result = _ft().track(df)
        assert (result["fix_level_count"] == 0).all()


# ------------------------------------------------------------------ #
# Multiple days                                                        #
# ------------------------------------------------------------------ #

class TestMultipleDays:
    def test_each_day_gets_own_fix_levels(self):
        df = _make_df("2026-01-05 00:00", 48, freq="1h", tz="Asia/Jerusalem")
        df = _set_close(df, 4, 2050.0)   # Day 1, 04:00 IST
        df = _set_close(df, 28, 2080.0)  # Day 2, 04:00 IST (24+4)
        result = _ft().track(df)

        # Day 1 Shanghai fix
        assert result["fix_shanghai_pm"].iloc[5] == pytest.approx(2050.0)
        # Day 2 Shanghai fix
        assert result["fix_shanghai_pm"].iloc[29] == pytest.approx(2080.0)

    def test_day1_fix_does_not_leak_to_day2(self):
        df = _make_df("2026-01-05 00:00", 48, freq="1h", tz="Asia/Jerusalem")
        df = _set_close(df, 4, 2050.0)   # Day 1, 04:00
        result = _ft().track(df)

        # Day 2 bars before fix time should be NaN (no day 1 leakage)
        assert np.isnan(result["fix_shanghai_pm"].iloc[24])  # Day 2, 00:00


# ------------------------------------------------------------------ #
# UTC input timezone                                                   #
# ------------------------------------------------------------------ #

class TestUTCInput:
    def test_utc_input_converts_correctly(self):
        """UTC input: 01:30 UTC = 03:30 IST (winter). Shanghai fix at 04:30 IST.
        Bar at 02:00 UTC = 04:00 IST → before fix.
        Bar at 03:00 UTC = 05:00 IST → after fix."""
        df = _make_df("2026-01-05 00:00", 24, freq="1h", tz="UTC")
        df = _set_close(df, 2, 2050.0)  # 02:00 UTC = 04:00 IST
        result = _ft().track(df)

        # 02:00 UTC (04:00 IST) → before 04:30 → no fix
        assert np.isnan(result["fix_shanghai_pm"].iloc[2])
        # 03:00 UTC (05:00 IST) → after 04:30 → fix set
        assert result["fix_shanghai_pm"].iloc[3] == pytest.approx(2050.0)

    def test_naive_index_treated_as_utc(self):
        """Naive (no tz) index → treated as UTC."""
        df = _make_df("2026-01-05 00:00", 24, freq="1h", tz=None)
        df.index = df.index.tz_localize(None)  # make naive
        df = _set_close(df, 2, 2050.0)
        result = _ft().track(df)
        # Same as UTC test above
        assert result["fix_shanghai_pm"].iloc[3] == pytest.approx(2050.0)


# ------------------------------------------------------------------ #
# Custom fix times                                                     #
# ------------------------------------------------------------------ #

class TestCustomFixTimes:
    def test_custom_single_fix(self):
        custom = {"my_fix": (10, 0)}  # 10:00 Israel
        df = _make_df("2026-01-05 08:00", 6, freq="1h", tz="Asia/Jerusalem")
        df = _set_close(df, 2, 2070.0)  # 10:00 IST
        result = _ft(fix_times=custom).track(df)
        assert "fix_my_fix" in result.columns
        assert result["fix_my_fix"].iloc[2] == pytest.approx(2070.0)
        # Default columns should NOT be present
        assert "fix_shanghai_pm" not in result.columns

    def test_invalid_fix_time_raises(self):
        with pytest.raises(ValueError, match="Invalid fix time"):
            FixLevelsTracker(fix_times={"bad": (25, 0)})

    def test_invalid_minute_raises(self):
        with pytest.raises(ValueError, match="Invalid fix time"):
            FixLevelsTracker(fix_times={"bad": (10, 61)})


# ------------------------------------------------------------------ #
# Edge cases                                                           #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_empty_df(self):
        df = _make_df("2026-01-05", 0)
        result = _ft().track(df)
        assert len(result) == 0
        assert "fix_shanghai_pm" in result.columns

    def test_single_bar_at_fix_time(self):
        """Single bar exactly at Shanghai PM fix time."""
        df = _make_df("2026-01-05 04:30", 1, freq="1h", tz="Asia/Jerusalem")
        df = _set_close(df, 0, 2055.0)
        result = _ft().track(df)
        # Bar at 04:30 is exactly at fix → fix set on this bar
        assert result["fix_shanghai_pm"].iloc[0] == pytest.approx(2055.0)

    def test_no_bar_before_fix(self):
        """Data starts after fix time → fix is set (uses last at-or-before)."""
        df = _make_df("2026-01-05 06:00", 4, freq="1h", tz="Asia/Jerusalem")
        result = _ft().track(df)
        # No bar at or before 04:30 on this day → no Shanghai fix
        # But 06:00 is after 04:30... the issue is there's no bar AT 04:30
        # Since no bar has bar_min <= fix_min (270), and 06:00=360 > 270,
        # there's no candidate. So Shanghai fix should be NaN.
        # Wait — actually data starts at 06:00, all bars have bar_min >= 360 > 270.
        # The loop sets day_fix[d] for bars where bar_min <= fix_total_min.
        # No bar qualifies → no fix.
        assert np.isnan(result["fix_shanghai_pm"].iloc[0])


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        result = _ft().track(df)
        for col in ["fix_shanghai_pm", "fix_london_am", "fix_london_pm", "fix_level_count"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        pd.testing.assert_index_equal(_ft().track(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        _ft().track(df)
        assert "fix_shanghai_pm" not in df.columns


# ------------------------------------------------------------------ #
# Validation                                                            #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_non_datetime_index_raises(self):
        df = _make_df("2026-01-05", 10)
        df.index = range(10)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _ft().track(df)

    def test_missing_close_raises(self):
        df = _make_df("2026-01-05", 10).drop(columns=["close"])
        with pytest.raises(ValueError, match="Missing"):
            _ft().track(df)

    def test_convenience_function_works(self):
        df = _make_df("2026-01-05 00:00", 24, freq="1h")
        result = track_fix_levels(df)
        assert "fix_shanghai_pm" in result.columns


# ------------------------------------------------------------------ #
# Accessor                                                             #
# ------------------------------------------------------------------ #

class TestAccessor:
    def test_get_fix_levels_at_end_of_day(self):
        df = _make_df("2026-01-05 00:00", 24, freq="1h", tz="Asia/Jerusalem")
        result = _ft().track(df)
        levels = _ft().get_fix_levels(result)
        # All 3 fixes should be set by end of day (23:00)
        assert len(levels) == 3
        names = {l["fix_name"] for l in levels}
        assert names == {"shanghai_pm", "london_am", "london_pm"}

    def test_get_fix_levels_empty_early_day(self):
        df = _make_df("2026-01-05 00:00", 3, freq="1h", tz="Asia/Jerusalem")
        result = _ft().track(df)
        levels = _ft().get_fix_levels(result)
        assert len(levels) == 0

    def test_get_fix_levels_empty_df(self):
        df = _make_df("2026-01-05", 0)
        result = _ft().track(df)
        assert _ft().get_fix_levels(result) == []
