"""
Tests for resampler.py — Phase 0.4.

Critical property verified in every test:
  No look-ahead — resampled candles only contain data from closed 1m candles.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Optional

from core.data.resampler import Resampler, resample, SUPPORTED_TIMEFRAMES


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

def _make_1m(
    start: str = "2026-01-01 10:00:00",
    periods: int = 60,
    base_price: float = 2000.0,
) -> pd.DataFrame:
    """
    Build a synthetic 1m OHLCV DataFrame.
    Each candle gets slightly different OHLC so we can verify aggregation.
    """
    idx = pd.date_range(start, periods=periods, freq="1min", tz="UTC", name="timestamp")
    opens  = [base_price + i * 0.10 for i in range(periods)]
    closes = [o + 0.05 for o in opens]
    highs  = [o + 1.00 for o in opens]
    lows   = [o - 1.00 for o in opens]
    vols   = [100.0 + i for i in range(periods)]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


# ------------------------------------------------------------------ #
# Basic resampling correctness                                         #
# ------------------------------------------------------------------ #

class TestResampleOhlcAggregation:
    """Verify OHLCV values are aggregated correctly."""

    def test_5m_open_is_first_1m_open(self):
        df = _make_1m(periods=10)
        r = Resampler()
        result = r.resample_one(df, "5m")
        # First 5m candle covers minutes 0-4
        expected_open = df["open"].iloc[0]
        assert result["open"].iloc[0] == pytest.approx(expected_open)

    def test_5m_close_is_last_1m_close(self):
        df = _make_1m(periods=10)
        result = Resampler().resample_one(df, "5m")
        expected_close = df["close"].iloc[4]   # minute index 4 = last of first 5m
        assert result["close"].iloc[0] == pytest.approx(expected_close)

    def test_5m_high_is_max_of_1m_highs(self):
        df = _make_1m(periods=10)
        result = Resampler().resample_one(df, "5m")
        expected_high = df["high"].iloc[0:5].max()
        assert result["high"].iloc[0] == pytest.approx(expected_high)

    def test_5m_low_is_min_of_1m_lows(self):
        df = _make_1m(periods=10)
        result = Resampler().resample_one(df, "5m")
        expected_low = df["low"].iloc[0:5].min()
        assert result["low"].iloc[0] == pytest.approx(expected_low)

    def test_5m_volume_is_sum_of_1m_volumes(self):
        df = _make_1m(periods=10)
        result = Resampler().resample_one(df, "5m")
        expected_vol = df["volume"].iloc[0:5].sum()
        assert result["volume"].iloc[0] == pytest.approx(expected_vol)

    def test_15m_produces_correct_candle_count(self):
        # 60 minutes of 1m data → 4 complete 15m candles (drop last in-progress)
        df = _make_1m(periods=60)
        result = Resampler().resample_one(df, "15m")
        # 60 min / 15 min = 4 candles, last dropped → 3
        assert len(result) == 3

    def test_1h_produces_correct_candle_count(self):
        # 120 minutes → 2 complete 1h candles (drop last)
        df = _make_1m(periods=120)
        result = Resampler().resample_one(df, "1h")
        assert len(result) == 1   # only first hour is fully closed

    def test_4h_candle_count(self):
        # 4*60 + 30 = 270 minutes → 1 complete 4h + 30 min in-progress
        df = _make_1m(periods=270)
        result = Resampler().resample_one(df, "4h")
        assert len(result) == 1


# ------------------------------------------------------------------ #
# No look-ahead — in-progress candle guard                            #
# ------------------------------------------------------------------ #

class TestNoLookahead:
    """
    The in-progress (not yet closed) candle must never appear in the output.
    """

    def test_last_candle_dropped_when_no_now(self):
        """Without `now`, the last resampled candle is always dropped."""
        df = _make_1m(periods=10)
        result = Resampler().resample_one(df, "5m", now=None)
        # 10 minutes → 2 candidate 5m candles; last is dropped
        assert len(result) == 1

    def test_in_progress_dropped_when_now_inside_period(self):
        """
        If `now` falls inside the second 5m period, that candle is incomplete
        and must not appear.
        """
        df = _make_1m(periods=8)   # 8 minutes: first 5m closed, second 5m open
        # now = 3 minutes into the second 5m period
        now = df.index[7]          # last 1m candle is in the in-progress period
        result = Resampler().resample_one(df, "5m", now=now)
        assert len(result) == 1    # only the first 5m is closed

    def test_candle_included_when_period_exactly_closed(self):
        """
        A 5m candle whose close time == now is fully closed and must be included.
        """
        df = _make_1m(periods=10)
        # now = exactly at the close of the second 5m candle
        now = pd.Timestamp("2026-01-01 10:10:00", tz="UTC")
        result = Resampler().resample_one(df, "5m", now=now)
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        df = _make_1m(periods=0)
        result = Resampler().resample_one(df, "5m")
        assert result.empty

    def test_output_index_is_utc(self):
        df = _make_1m(periods=15)
        result = Resampler().resample_one(df, "5m")
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"

    def test_output_index_is_sorted(self):
        df = _make_1m(periods=30)
        result = Resampler().resample_one(df, "5m")
        assert result.index.is_monotonic_increasing


# ------------------------------------------------------------------ #
# resample_all                                                          #
# ------------------------------------------------------------------ #

class TestResampleAll:
    def test_returns_all_requested_timeframes(self):
        df = _make_1m(periods=300)   # 5 hours of 1m data
        r = Resampler()
        result = r.resample_all(df, ["5m", "15m", "1h"])
        assert set(result.keys()) == {"5m", "15m", "1h"}

    def test_each_df_has_correct_columns(self):
        df = _make_1m(periods=120)
        r = Resampler()
        result = r.resample_all(df, ["5m", "1h"])
        for tf, tf_df in result.items():
            assert set(tf_df.columns) == {"open", "high", "low", "close", "volume"}, tf

    def test_same_timeframe_as_base_returns_copy(self):
        df = _make_1m(periods=10)
        result = Resampler().resample_one(df, "1m")
        pd.testing.assert_frame_equal(result, df)


# ------------------------------------------------------------------ #
# Input validation                                                      #
# ------------------------------------------------------------------ #

class TestResamplerValidation:
    def test_unsupported_timeframe_raises(self):
        df = _make_1m(periods=10)
        with pytest.raises(ValueError, match="Unsupported target timeframe"):
            Resampler().resample_one(df, "3m")

    def test_unsupported_base_timeframe_raises(self):
        with pytest.raises(ValueError, match="Unsupported base timeframe"):
            Resampler(base_timeframe="3m")

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [10.0]},
            index=[0],
        )
        with pytest.raises(TypeError, match="DatetimeIndex"):
            Resampler().resample_one(df, "5m")

    def test_tz_naive_index_raises(self):
        df = _make_1m(periods=10)
        df.index = df.index.tz_localize(None)
        with pytest.raises(ValueError, match="UTC tz-aware"):
            Resampler().resample_one(df, "5m")

    def test_missing_column_raises(self):
        df = _make_1m(periods=10).drop(columns=["volume"])
        with pytest.raises(ValueError, match="Missing columns"):
            Resampler().resample_one(df, "5m")


# ------------------------------------------------------------------ #
# Module-level convenience function                                     #
# ------------------------------------------------------------------ #

class TestResampleFunction:
    def test_convenience_wrapper_works(self):
        df = _make_1m(periods=15)
        result = resample(df, "5m")
        assert not result.empty
        assert "close" in result.columns

    def test_with_now_parameter(self):
        df = _make_1m(periods=10)
        now = pd.Timestamp("2026-01-01 10:10:00", tz="UTC")
        result = resample(df, "5m", now=now)
        assert len(result) == 2


# ------------------------------------------------------------------ #
# Data integrity — no future data leaks into past candles             #
# ------------------------------------------------------------------ #

class TestNoFutureDataLeak:
    """
    Each 5m candle must only contain price data from its own 5 minutes.
    Simulate a scenario where a spike occurs in minute 6 (second 5m period)
    and verify it does NOT appear in the first 5m candle.
    """

    def test_spike_in_period_2_not_in_period_1(self):
        df = _make_1m(periods=15)
        # Inject a massive spike at minute index 5 (belongs to second 5m candle)
        df.iloc[5, df.columns.get_loc("high")] = 99_000.0

        result = Resampler().resample_one(df, "5m", now=None)

        # First 5m candle: should not see the 99_000 spike
        first_high = result["high"].iloc[0]
        assert first_high < 99_000.0, (
            f"Look-ahead detected: first 5m high={first_high} contains data from period 2"
        )
