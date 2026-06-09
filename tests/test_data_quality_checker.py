"""
Tests for data_quality_checker.py — Phase 0.6.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from core.data.data_quality_checker import (
    DataQualityChecker,
    DataQualityReport,
    OhlcViolation,
    build_quality_checker,
)
from core.data.gap_detector import GapDetector


# ------------------------------------------------------------------ #
# Shared fixtures                                                       #
# ------------------------------------------------------------------ #

GAP_CFG = {
    "enabled": True,
    "max_missing_candles_allowed": 2,
    "price_gap_atr_multiplier_warning": 1.0,
    "price_gap_atr_multiplier_block": 1.5,
    "cooldown_after_gap_minutes": 60,
    "weekend_gap_cooldown_minutes": 120,
    "atr_period": 14,
}

CAL_CFG = {"gap_detection": GAP_CFG}


def _checker(with_gap: bool = True) -> DataQualityChecker:
    det = GapDetector(config=GAP_CFG, timeframe="5m") if with_gap else None
    return DataQualityChecker(gap_detector=det, max_allowed_missing=2, fill_nan=True)


def _make_df(
    n: int = 20,
    start: str = "2026-01-05 10:00",   # Monday
    freq: str = "5min",
    base: float = 2000.0,
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open":   [base] * n,
            "high":   [base + 5] * n,
            "low":    [base - 5] * n,
            "close":  [base] * n,
            "volume": [100.0] * n,
        },
        index=idx,
    )


# ------------------------------------------------------------------ #
# Clean data                                                           #
# ------------------------------------------------------------------ #

class TestCleanData:
    def test_clean_df_is_usable(self):
        df = _make_df()
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert report.is_usable is True

    def test_clean_df_has_no_errors(self):
        df = _make_df()
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert report.errors == []

    def test_clean_df_has_no_warnings(self):
        df = _make_df()
        _, report = _checker(with_gap=False).check(df, "XAUUSD", "5m")
        assert report.warnings == []

    def test_total_candles_correct(self):
        df = _make_df(n=30)
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert report.total_candles == 30

    def test_summary_line_contains_ok(self):
        df = _make_df()
        _, report = _checker(with_gap=False).check(df, "XAUUSD", "5m")
        assert "OK" in report.summary_line


# ------------------------------------------------------------------ #
# Duplicate timestamps                                                 #
# ------------------------------------------------------------------ #

class TestDuplicateTimestamps:
    def test_duplicates_detected_and_deduplicated(self):
        df = _make_df(n=10)
        # Duplicate the third row
        df = pd.concat([df, df.iloc[[2]]])
        df = df.sort_index()
        cleaned, report = _checker().check(df, "XAUUSD", "5m")
        assert report.duplicate_timestamps == 1
        assert cleaned.index.duplicated().sum() == 0

    def test_duplicates_produce_warning_not_error(self):
        df = _make_df(n=10)
        df = pd.concat([df, df.iloc[[2]]]).sort_index()
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert any("duplicate" in w.lower() for w in report.warnings)
        assert report.is_usable is True   # duplicates alone don't block


# ------------------------------------------------------------------ #
# NaN handling                                                         #
# ------------------------------------------------------------------ #

class TestNanHandling:
    def test_nan_rows_detected_and_filled(self):
        df = _make_df(n=10)
        df.iloc[5, df.columns.get_loc("close")] = float("nan")
        cleaned, report = _checker().check(df, "XAUUSD", "5m")
        assert report.nan_rows_found == 1
        assert report.nan_rows_filled == 1
        assert cleaned["close"].isna().sum() == 0

    def test_nan_without_fill_produces_error(self):
        df = _make_df(n=10)
        df.iloc[3, df.columns.get_loc("open")] = float("nan")
        det = GapDetector(config=GAP_CFG, timeframe="5m")
        checker = DataQualityChecker(gap_detector=det, fill_nan=False)
        _, report = checker.check(df, "XAUUSD", "5m")
        assert report.is_usable is False
        assert any("NaN" in e for e in report.errors)

    def test_no_nan_reports_zero_filled(self):
        df = _make_df(n=10)
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert report.nan_rows_found == 0
        assert report.nan_rows_filled == 0


# ------------------------------------------------------------------ #
# Zero / negative prices                                               #
# ------------------------------------------------------------------ #

class TestZeroNegativePrices:
    def test_zero_price_makes_unusable(self):
        df = _make_df(n=10)
        df.iloc[4, df.columns.get_loc("low")] = 0.0
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert report.zero_or_negative_prices == 1
        assert report.is_usable is False

    def test_negative_price_makes_unusable(self):
        df = _make_df(n=10)
        df.iloc[4, df.columns.get_loc("low")] = -5.0
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert report.is_usable is False

    def test_all_positive_prices_pass(self):
        df = _make_df(n=10)
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert report.zero_or_negative_prices == 0


# ------------------------------------------------------------------ #
# OHLC integrity                                                       #
# ------------------------------------------------------------------ #

class TestOhlcIntegrity:
    def test_high_below_close_is_violation(self):
        df = _make_df(n=10)
        df.iloc[3, df.columns.get_loc("high")] = 1998.0   # below open/close=2000
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert len(report.ohlc_violations) >= 1
        assert report.is_usable is False

    def test_low_above_open_is_violation(self):
        df = _make_df(n=10)
        df.iloc[5, df.columns.get_loc("low")] = 2002.0    # above open/close=2000
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert len(report.ohlc_violations) >= 1

    def test_high_below_low_is_violation(self):
        df = _make_df(n=10)
        row = 6
        df.iloc[row, df.columns.get_loc("high")] = 1990.0
        df.iloc[row, df.columns.get_loc("low")]  = 2010.0
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert any(v.timestamp == df.index[row] for v in report.ohlc_violations)

    def test_valid_candles_have_no_violations(self):
        df = _make_df(n=20)
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert report.ohlc_violations == []


# ------------------------------------------------------------------ #
# Continuity                                                           #
# ------------------------------------------------------------------ #

class TestContinuityCheck:
    def test_missing_candles_within_threshold_is_warning(self):
        df = _make_df(n=20)
        df = df.drop(df.index[5])   # 1 missing
        _, report = _checker(with_gap=False).check(df, "XAUUSD", "5m")
        assert report.missing_candles == 1
        assert report.is_usable is True
        assert any("missing" in w.lower() for w in report.warnings)

    def test_too_many_missing_makes_unusable(self):
        df = _make_df(n=20)
        df = df.drop(df.index[5:10])   # 5 missing
        _, report = _checker(with_gap=False).check(df, "XAUUSD", "5m")
        assert report.is_usable is False

    def test_no_missing_candles_no_continuity_error(self):
        df = _make_df(n=20)
        _, report = _checker(with_gap=False).check(df, "XAUUSD", "5m")
        assert report.missing_candles == 0


# ------------------------------------------------------------------ #
# Gap integration                                                      #
# ------------------------------------------------------------------ #

class TestGapIntegration:
    def test_blocking_price_gap_makes_unusable(self):
        df = _make_df(n=30)
        df.iloc[15, df.columns.get_loc("open")] = 2030.0   # ATR≈10, gap≈30 → block
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert report.has_blocking_gap is True
        assert report.is_usable is False

    def test_no_gap_detector_skips_gap_check(self):
        df = _make_df(n=10)
        checker = DataQualityChecker(gap_detector=None)
        _, report = checker.check(df, "XAUUSD", "5m")
        assert report.gap_report is None

    def test_warning_gap_does_not_block(self):
        df = _make_df(n=30)
        # Set open=2012, high=2017, low=2007, close=2012 — valid OHLC, gap≈12 → warn only
        df.iloc[15, df.columns.get_loc("open")]  = 2012.0
        df.iloc[15, df.columns.get_loc("high")]  = 2017.0
        df.iloc[15, df.columns.get_loc("low")]   = 2007.0
        df.iloc[15, df.columns.get_loc("close")] = 2012.0
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert report.has_blocking_gap is False
        assert report.is_usable is True


# ------------------------------------------------------------------ #
# Report properties                                                    #
# ------------------------------------------------------------------ #

class TestReportProperties:
    def test_summary_line_not_usable(self):
        df = _make_df(n=10)
        df.iloc[3, df.columns.get_loc("high")] = 1990.0   # OHLC violation
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert "NOT_USABLE" in report.summary_line

    def test_checked_at_is_set(self):
        df = _make_df(n=5)
        _, report = _checker().check(df, "XAUUSD", "5m")
        assert isinstance(report.checked_at, datetime)


# ------------------------------------------------------------------ #
# Factory                                                              #
# ------------------------------------------------------------------ #

class TestBuildQualityChecker:
    def test_factory_returns_checker(self):
        checker = build_quality_checker(CAL_CFG, "5m")
        assert isinstance(checker, DataQualityChecker)

    def test_factory_checker_runs_without_error(self):
        df = _make_df(n=10)
        checker = build_quality_checker(CAL_CFG, "5m")
        cleaned, report = checker.check(df, "XAUUSD", "5m")
        assert isinstance(report, DataQualityReport)
