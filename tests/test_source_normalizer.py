"""
Tests for source_normalizer.py — Phase 0.3.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import timezone

from core.data.source_normalizer import (
    normalize_symbol,
    normalize_timezone,
    normalize_ohlcv_schema,
    check_decimal_precision,
    validate_time_continuity,
    normalize_and_validate,
    ContinuityReport,
    PrecisionReport,
)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

def _make_df(
    n: int = 5,
    start: str = "2026-01-01 10:00:00",
    freq: str = "5min",
    tz: str = "UTC",
    col_names=None,
) -> pd.DataFrame:
    """Build a minimal valid OHLCV DataFrame."""
    idx = pd.date_range(start, periods=n, freq=freq, tz=tz, name="timestamp")
    cols = col_names or ["open", "high", "low", "close", "volume"]
    data = {
        "open":   [2000.0] * n,
        "high":   [2010.0] * n,
        "low":    [1990.0] * n,
        "close":  [2005.0] * n,
        "volume": [100.0]  * n,
    }
    return pd.DataFrame({c: data.get(c, [0.0] * n) for c in cols}, index=idx)


# ------------------------------------------------------------------ #
# normalize_symbol                                                     #
# ------------------------------------------------------------------ #

class TestNormalizeSymbol:
    @pytest.mark.parametrize("raw,expected", [
        ("XAUUSD",   "XAUUSD"),
        ("XAUUSDT",  "XAUUSD"),
        ("GC=F",     "XAUUSD"),
        ("gold",     "XAUUSD"),   # case-insensitive
        ("DXY",      "DXY"),
        ("DX-Y.NYB", "DXY"),
        ("US10Y",    "US10Y"),
        ("^TNX",     "US10Y"),
    ])
    def test_known_aliases(self, raw, expected):
        assert normalize_symbol(raw) == expected

    def test_unknown_symbol_raises(self):
        with pytest.raises(ValueError, match="Unknown symbol"):
            normalize_symbol("BTCUSD")

    def test_strips_whitespace(self):
        assert normalize_symbol("  xauusd  ") == "XAUUSD"


# ------------------------------------------------------------------ #
# normalize_timezone                                                   #
# ------------------------------------------------------------------ #

class TestNormalizeTimezone:
    def test_tz_naive_becomes_utc(self):
        df = _make_df(tz=None)
        df.index = pd.DatetimeIndex(df.index)   # strip tz
        result = normalize_timezone(df)
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"

    def test_tz_aware_non_utc_converted(self):
        df = _make_df(tz="Asia/Jerusalem")
        result = normalize_timezone(df)
        assert str(result.index.tz) == "UTC"

    def test_utc_stays_utc(self):
        df = _make_df(tz="UTC")
        result = normalize_timezone(df)
        assert str(result.index.tz) == "UTC"

    def test_raises_on_non_datetime_index(self):
        df = pd.DataFrame({"open": [1, 2]}, index=[0, 1])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            normalize_timezone(df)

    def test_index_name_set_to_timestamp(self):
        df = _make_df()
        df.index.name = "time"
        result = normalize_timezone(df)
        assert result.index.name == "timestamp"


# ------------------------------------------------------------------ #
# normalize_ohlcv_schema                                               #
# ------------------------------------------------------------------ #

class TestNormalizeOhlcvSchema:
    def test_valid_df_passes_through(self):
        df = _make_df()
        result = normalize_ohlcv_schema(df)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        assert result.index.tz is not None

    def test_uppercase_columns_normalised(self):
        df = _make_df()
        df.columns = [c.upper() for c in df.columns]
        result = normalize_ohlcv_schema(df)
        assert "open" in result.columns

    def test_extra_columns_dropped(self):
        df = _make_df()
        df["extra"] = 999
        result = normalize_ohlcv_schema(df)
        assert "extra" not in result.columns
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_missing_column_raises(self):
        df = _make_df()
        df = df.drop(columns=["close"])
        with pytest.raises(ValueError, match="Missing required columns"):
            normalize_ohlcv_schema(df)

    def test_columns_cast_to_float64(self):
        df = _make_df()
        df["open"] = df["open"].astype(str)   # intentionally wrong type
        result = normalize_ohlcv_schema(df)
        assert result["open"].dtype == np.float64

    def test_timestamp_as_column_becomes_index(self):
        df = _make_df()
        df = df.reset_index()   # 'timestamp' is now a column
        result = normalize_ohlcv_schema(df)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "timestamp"

    def test_sorted_chronologically(self):
        df = _make_df(n=4)
        df = df.iloc[::-1]   # reverse order
        result = normalize_ohlcv_schema(df)
        assert result.index.is_monotonic_increasing

    def test_output_is_utc(self):
        df = _make_df(tz="Asia/Jerusalem")
        result = normalize_ohlcv_schema(df)
        assert str(result.index.tz) == "UTC"


# ------------------------------------------------------------------ #
# check_decimal_precision                                              #
# ------------------------------------------------------------------ #

class TestCheckDecimalPrecision:
    def test_normal_xau_prices_pass(self):
        df = _make_df()
        report = check_decimal_precision(df)
        assert report.is_acceptable is True
        assert report.rows_outside_price_range == 0

    def test_price_below_range_flagged(self):
        df = _make_df()
        df["close"] = 100.0   # way below 500
        report = check_decimal_precision(df)
        assert report.rows_outside_price_range > 0
        assert report.is_acceptable is False

    def test_price_above_range_flagged(self):
        df = _make_df()
        df["high"] = 99_000.0
        report = check_decimal_precision(df)
        assert report.rows_outside_price_range > 0
        assert report.is_acceptable is False

    def test_too_many_decimals_flagged(self):
        df = _make_df()
        df["close"] = 2000.1234567   # 7 decimal places
        report = check_decimal_precision(df)
        assert report.max_decimals_found > 5
        assert report.is_acceptable is False

    def test_two_decimal_places_ok(self):
        df = _make_df()
        df["close"] = 2345.67
        report = check_decimal_precision(df)
        assert report.max_decimals_found <= 5
        assert report.is_acceptable is True


# ------------------------------------------------------------------ #
# validate_time_continuity                                             #
# ------------------------------------------------------------------ #

class TestValidateTimeContinuity:
    def test_complete_series_is_acceptable(self):
        df = _make_df(n=10, freq="5min")
        report = validate_time_continuity(df, "5m")
        assert report.missing_count == 0
        assert report.is_acceptable is True

    def test_one_gap_within_threshold(self):
        df = _make_df(n=10, freq="5min")
        df = df.drop(df.index[4])   # remove one candle
        report = validate_time_continuity(df, "5m", max_allowed_missing=2)
        assert report.missing_count == 1
        assert report.is_acceptable is True

    def test_too_many_gaps_not_acceptable(self):
        df = _make_df(n=20, freq="5min")
        df = df.drop(df.index[2:8])   # remove 6 candles
        report = validate_time_continuity(df, "5m", max_allowed_missing=2)
        assert report.missing_count == 6
        assert report.is_acceptable is False

    def test_empty_df_is_acceptable(self):
        df = _make_df(n=0)
        report = validate_time_continuity(df, "5m")
        assert report.total_candles == 0
        assert report.is_acceptable is True

    def test_unknown_timeframe_raises(self):
        df = _make_df(n=5)
        with pytest.raises(ValueError, match="Unknown timeframe"):
            validate_time_continuity(df, "3m")

    def test_missing_timestamps_recorded(self):
        df = _make_df(n=10, freq="5min")
        dropped_ts = df.index[3]
        df = df.drop(dropped_ts)
        report = validate_time_continuity(df, "5m")
        assert dropped_ts in report.missing_timestamps

    def test_1h_continuity(self):
        df = _make_df(n=24, freq="1h")
        report = validate_time_continuity(df, "1h")
        assert report.missing_count == 0
        assert report.is_acceptable is True


# ------------------------------------------------------------------ #
# normalize_and_validate (pipeline)                                    #
# ------------------------------------------------------------------ #

class TestNormalizeAndValidate:
    def test_full_pipeline_valid_input(self):
        df = _make_df(n=10, freq="5min")
        result_df, symbol, precision, continuity = normalize_and_validate(
            df, "XAUUSDT", "5m"
        )
        assert symbol == "XAUUSD"
        assert str(result_df.index.tz) == "UTC"
        assert precision.is_acceptable is True
        assert continuity.is_acceptable is True

    def test_pipeline_raises_on_unknown_symbol(self):
        df = _make_df()
        with pytest.raises(ValueError, match="Unknown symbol"):
            normalize_and_validate(df, "ETHUSD", "5m")

    def test_pipeline_raises_on_missing_column(self):
        df = _make_df().drop(columns=["volume"])
        with pytest.raises(ValueError, match="Missing required columns"):
            normalize_and_validate(df, "XAUUSD", "5m")
