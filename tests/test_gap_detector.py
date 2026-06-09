"""
Tests for gap_detector.py — Phase 0.5.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import timedelta

from core.data.gap_detector import (
    GapDetector,
    GapEvent,
    GapReport,
    GapSeverity,
    GapType,
    build_gap_detector,
)


# ------------------------------------------------------------------ #
# Config fixture                                                        #
# ------------------------------------------------------------------ #

BASE_CFG = {
    "enabled": True,
    "max_missing_candles_allowed": 2,
    "price_gap_atr_multiplier_warning": 1.0,
    "price_gap_atr_multiplier_block": 1.5,
    "cooldown_after_gap_minutes": 60,
    "weekend_gap_cooldown_minutes": 120,
    "atr_period": 14,
}


def _detector(cfg: dict = None, tf: str = "5m") -> GapDetector:
    return GapDetector(config=cfg or BASE_CFG, timeframe=tf)


# ------------------------------------------------------------------ #
# OHLCV DataFrame builders                                             #
# ------------------------------------------------------------------ #

def _make_df(
    start: str,
    periods: int,
    freq: str = "5min",
    base_price: float = 2000.0,
    high_offset: float = 5.0,
    low_offset: float = 5.0,
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open":   [base_price] * periods,
            "high":   [base_price + high_offset] * periods,
            "low":    [base_price - low_offset] * periods,
            "close":  [base_price] * periods,
            "volume": [100.0] * periods,
        },
        index=idx,
    )


def _insert_time_gap(df: pd.DataFrame, drop_indices: list) -> pd.DataFrame:
    """Remove rows at given integer positions to create a time gap."""
    return df.drop(df.index[drop_indices])


def _insert_price_gap(df: pd.DataFrame, row: int, new_open: float) -> pd.DataFrame:
    df = df.copy()
    df.iloc[row, df.columns.get_loc("open")] = new_open
    return df


# ------------------------------------------------------------------ #
# Time gap tests                                                       #
# ------------------------------------------------------------------ #

class TestTimeGaps:
    def test_no_gaps_clean_series(self):
        df = _make_df("2026-01-05 10:00", 20)   # Monday, clean 5m series
        report = _detector().scan(df)
        assert len(report.time_gaps) == 0

    def test_single_missing_candle_is_warning(self):
        df = _make_df("2026-01-05 10:00", 20)
        df = _insert_time_gap(df, [5])   # remove 1 candle
        report = _detector().scan(df)
        assert len(report.time_gaps) == 1
        assert report.time_gaps[0].severity == GapSeverity.WARNING
        assert report.time_gaps[0].missing_candles == 1

    def test_two_missing_candles_is_warning(self):
        df = _make_df("2026-01-05 10:00", 20)
        df = _insert_time_gap(df, [5, 6])
        report = _detector().scan(df)
        assert len(report.time_gaps) == 1
        assert report.time_gaps[0].severity == GapSeverity.WARNING
        assert report.time_gaps[0].missing_candles == 2

    def test_three_missing_candles_is_block(self):
        df = _make_df("2026-01-05 10:00", 20)
        df = _insert_time_gap(df, [5, 6, 7])
        report = _detector().scan(df)
        assert len(report.time_gaps) == 1
        assert report.time_gaps[0].severity == GapSeverity.BLOCK
        assert report.time_gaps[0].cooldown_minutes == BASE_CFG["cooldown_after_gap_minutes"]

    def test_time_gap_has_correct_type(self):
        df = _make_df("2026-01-05 10:00", 20)
        df = _insert_time_gap(df, [5])
        report = _detector().scan(df)
        assert report.time_gaps[0].gap_type == GapType.TIME

    def test_empty_df_returns_empty_report(self):
        df = _make_df("2026-01-05 10:00", 0)
        report = _detector().scan(df)
        assert report.total_gap_count == 0

    def test_single_row_returns_empty_report(self):
        df = _make_df("2026-01-05 10:00", 1)
        report = _detector().scan(df)
        assert report.total_gap_count == 0

    def test_1h_timeframe_correct_expected_spacing(self):
        df = _make_df("2026-01-05 10:00", 10, freq="1h")
        df = _insert_time_gap(df, [3])
        report = _detector(tf="1h").scan(df)
        assert len(report.time_gaps) == 1
        assert report.time_gaps[0].missing_candles == 1


# ------------------------------------------------------------------ #
# Price gap tests                                                      #
# ------------------------------------------------------------------ #

class TestPriceGaps:
    def _detector_with_known_atr(self) -> GapDetector:
        """ATR for flat 2000.0 series (high=2005, low=1995) ≈ 10.0."""
        return _detector()

    def test_no_price_gap_within_threshold(self):
        # Gap of 2 USD on ATR≈10 → ratio=0.2 < warning threshold 1.0
        df = _make_df("2026-01-05 10:00", 30)
        df = _insert_price_gap(df, 15, new_open=2002.0)
        report = self._detector_with_known_atr().scan(df)
        assert len(report.price_gaps) == 0

    def test_price_gap_warning(self):
        # ATR≈10, gap=12 → ratio≈1.2, between warning(1.0) and block(1.5)
        df = _make_df("2026-01-05 10:00", 30)
        df = _insert_price_gap(df, 15, new_open=2012.0)
        report = self._detector_with_known_atr().scan(df)
        assert len(report.price_gaps) == 1
        assert report.price_gaps[0].severity == GapSeverity.WARNING

    def test_price_gap_block(self):
        # ATR≈10, gap=20 → ratio≈2.0 > block threshold 1.5
        df = _make_df("2026-01-05 10:00", 30)
        df = _insert_price_gap(df, 15, new_open=2020.0)
        report = self._detector_with_known_atr().scan(df)
        assert len(report.price_gaps) == 1
        assert report.price_gaps[0].severity == GapSeverity.BLOCK

    def test_price_gap_has_correct_type(self):
        df = _make_df("2026-01-05 10:00", 30)
        df = _insert_price_gap(df, 15, new_open=2020.0)
        report = _detector().scan(df)
        assert report.price_gaps[0].gap_type == GapType.PRICE

    def test_price_gap_stores_prev_close_and_curr_open(self):
        df = _make_df("2026-01-05 10:00", 30)
        df = _insert_price_gap(df, 15, new_open=2020.0)
        report = _detector().scan(df)
        gap = report.price_gaps[0]
        assert gap.previous_close == pytest.approx(2000.0)
        assert gap.current_open   == pytest.approx(2020.0)

    def test_block_gap_has_cooldown(self):
        df = _make_df("2026-01-05 10:00", 30)
        df = _insert_price_gap(df, 15, new_open=2020.0)
        report = _detector().scan(df)
        assert report.price_gaps[0].cooldown_minutes == BASE_CFG["cooldown_after_gap_minutes"]

    def test_warning_gap_has_no_cooldown(self):
        df = _make_df("2026-01-05 10:00", 30)
        df = _insert_price_gap(df, 15, new_open=2012.0)
        report = _detector().scan(df)
        assert report.price_gaps[0].cooldown_minutes == 0


# ------------------------------------------------------------------ #
# Weekend gap tests                                                    #
# ------------------------------------------------------------------ #

class TestWeekendGaps:
    def _make_friday_monday(self, friday_close: float, monday_open: float) -> pd.DataFrame:
        """Two candles: Friday 22:00 and Monday 00:00."""
        idx = pd.DatetimeIndex([
            pd.Timestamp("2026-01-02 22:00:00", tz="UTC"),  # Friday
            pd.Timestamp("2026-01-05 00:00:00", tz="UTC"),  # Monday
        ], name="timestamp")
        return pd.DataFrame(
            {
                "open":   [friday_close, monday_open],
                "high":   [friday_close + 5, monday_open + 5],
                "low":    [friday_close - 5, monday_open - 5],
                "close":  [friday_close, monday_open],
                "volume": [100.0, 100.0],
            },
            index=idx,
        )

    def test_weekend_gap_detected(self):
        df = self._make_friday_monday(2000.0, 2010.0)
        report = _detector().scan(df)
        assert len(report.weekend_gaps) == 1

    def test_weekend_gap_has_correct_type(self):
        df = self._make_friday_monday(2000.0, 2010.0)
        report = _detector().scan(df)
        assert report.weekend_gaps[0].gap_type == GapType.WEEKEND

    def test_weekend_gap_uses_longer_cooldown(self):
        df = self._make_friday_monday(2000.0, 2010.0)
        report = _detector().scan(df)
        assert report.weekend_gaps[0].cooldown_minutes == BASE_CFG["weekend_gap_cooldown_minutes"]

    def test_weekend_gap_not_also_flagged_as_price_gap(self):
        """A weekend transition must not appear in both weekend_gaps and price_gaps."""
        df = self._make_friday_monday(2000.0, 2030.0)
        report = _detector().scan(df)
        assert len(report.price_gaps) == 0
        assert len(report.weekend_gaps) == 1

    def test_small_weekend_gap_still_logged(self):
        """Even a tiny weekend gap is always recorded (forced_weekend=True)."""
        df = self._make_friday_monday(2000.0, 2000.5)
        report = _detector().scan(df)
        assert len(report.weekend_gaps) == 1
        assert report.weekend_gaps[0].severity == GapSeverity.WARNING


# ------------------------------------------------------------------ #
# GapReport helpers                                                    #
# ------------------------------------------------------------------ #

class TestGapReport:
    def test_has_blocking_gap_true(self):
        df = _make_df("2026-01-05 10:00", 30)
        df = _insert_price_gap(df, 15, new_open=2030.0)
        report = _detector().scan(df)
        assert report.has_blocking_gap is True

    def test_has_blocking_gap_false_on_clean_series(self):
        df = _make_df("2026-01-05 10:00", 20)
        report = _detector().scan(df)
        assert report.has_blocking_gap is False

    def test_all_gaps_combines_all_types(self):
        # price gap on Monday series + manual time gap
        df = _make_df("2026-01-05 10:00", 30)
        df = _insert_time_gap(df, [5])
        df = _insert_price_gap(df, 10, new_open=2030.0)
        report = _detector().scan(df)
        assert len(report.all_gaps) == len(report.time_gaps) + len(report.price_gaps)

    def test_total_candles_correct(self):
        df = _make_df("2026-01-05 10:00", 20)
        report = _detector().scan(df)
        assert report.total_candles == 20


# ------------------------------------------------------------------ #
# Disabled detector                                                    #
# ------------------------------------------------------------------ #

class TestDisabledDetector:
    def test_disabled_returns_empty_report(self):
        cfg = {**BASE_CFG, "enabled": False}
        df = _make_df("2026-01-05 10:00", 30)
        df = _insert_price_gap(df, 15, new_open=2030.0)
        report = GapDetector(config=cfg, timeframe="5m").scan(df)
        assert report.total_gap_count == 0


# ------------------------------------------------------------------ #
# Single-transition live check                                         #
# ------------------------------------------------------------------ #

class TestCheckSingleTransition:
    def test_no_gap_returns_none(self):
        det = _detector()
        prev_ts = pd.Timestamp("2026-01-05 10:00", tz="UTC")
        curr_ts = pd.Timestamp("2026-01-05 10:05", tz="UTC")
        result = det.check_single_transition(2000.0, 2000.5, prev_ts, curr_ts, atr=10.0)
        assert result is None

    def test_price_gap_detected_live(self):
        det = _detector()
        prev_ts = pd.Timestamp("2026-01-05 10:00", tz="UTC")
        curr_ts = pd.Timestamp("2026-01-05 10:05", tz="UTC")
        result = det.check_single_transition(2000.0, 2020.0, prev_ts, curr_ts, atr=10.0)
        assert result is not None
        assert result.gap_type == GapType.PRICE

    def test_weekend_boundary_detected_live(self):
        det = _detector()
        prev_ts = pd.Timestamp("2026-01-02 22:00", tz="UTC")  # Friday
        curr_ts = pd.Timestamp("2026-01-05 00:00", tz="UTC")  # Monday
        result = det.check_single_transition(2000.0, 2003.0, prev_ts, curr_ts, atr=10.0)
        assert result is not None
        assert result.gap_type == GapType.WEEKEND


# ------------------------------------------------------------------ #
# Factory                                                              #
# ------------------------------------------------------------------ #

class TestBuildGapDetector:
    def test_factory_extracts_gap_detection_section(self):
        full_cfg = {
            "enabled": True,
            "gap_detection": BASE_CFG,
        }
        det = build_gap_detector(full_cfg, "5m")
        assert isinstance(det, GapDetector)

    def test_factory_uses_defaults_when_section_missing(self):
        det = build_gap_detector({}, "1h")
        assert isinstance(det, GapDetector)

    def test_unsupported_timeframe_raises(self):
        with pytest.raises(ValueError, match="Unknown timeframe"):
            GapDetector(config=BASE_CFG, timeframe="3m")
