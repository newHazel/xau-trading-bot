"""
Data Quality Checker — Phase 0.6.

Runs after data is fetched and normalised. Produces a DataQualityReport
that answers:

  - How many candles were loaded?
  - How many are missing (time continuity)?
  - How many had NaN values (and were forward-filled)?
  - Are OHLC relationships valid (high >= open/close >= low)?
  - Are there duplicate timestamps?
  - Are there zero/negative prices?
  - What gaps were detected (time / price / weekend)?
  - Is the dataset safe to use for analysis?

Results are written to the data_quality_log SQLite table (Phase 0.7).
This module only builds the report — the DB write happens in the logger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd

from core.data.gap_detector import GapDetector, GapEvent, GapReport, GapSeverity
from core.data.source_normalizer import ContinuityReport, validate_time_continuity

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Result types                                                         #
# ------------------------------------------------------------------ #

@dataclass
class OhlcViolation:
    timestamp: pd.Timestamp
    reason: str   # e.g. "high < low", "low > open"


@dataclass
class DataQualityReport:
    """
    Full quality snapshot for one (symbol, timeframe, fetch) cycle.
    Stored in data_quality_log per run.
    """
    symbol: str
    timeframe: str
    checked_at: datetime

    # Counts
    total_candles: int = 0
    nan_rows_found: int = 0
    nan_rows_filled: int = 0
    duplicate_timestamps: int = 0
    zero_or_negative_prices: int = 0
    ohlc_violations: List[OhlcViolation] = field(default_factory=list)

    # Continuity (from source_normalizer)
    continuity: Optional[ContinuityReport] = None

    # Gaps (from gap_detector)
    gap_report: Optional[GapReport] = None

    # Overall verdict
    is_usable: bool = True          # False → do not trade this data
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------- #
    # Derived properties                                                 #
    # ---------------------------------------------------------------- #

    @property
    def missing_candles(self) -> int:
        return self.continuity.missing_count if self.continuity else 0

    @property
    def has_blocking_gap(self) -> bool:
        return self.gap_report.has_blocking_gap if self.gap_report else False

    @property
    def summary_line(self) -> str:
        status = "OK" if self.is_usable else "NOT_USABLE"
        return (
            f"[QC {status}] {self.symbol} {self.timeframe} | "
            f"candles={self.total_candles} missing={self.missing_candles} "
            f"nan_filled={self.nan_rows_filled} dups={self.duplicate_timestamps} "
            f"ohlc_violations={len(self.ohlc_violations)} "
            f"blocking_gap={self.has_blocking_gap} "
            f"warnings={len(self.warnings)} errors={len(self.errors)}"
        )


# ------------------------------------------------------------------ #
# Checker                                                              #
# ------------------------------------------------------------------ #

class DataQualityChecker:
    """
    Runs all quality checks on a normalised OHLCV DataFrame.

    Usage
    -----
    checker = DataQualityChecker(gap_detector=det, max_allowed_missing=2)
    report  = checker.check(df, symbol="XAUUSD", timeframe="5m")
    """

    def __init__(
        self,
        gap_detector: Optional[GapDetector] = None,
        max_allowed_missing: int = 2,
        fill_nan: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        gap_detector : GapDetector, optional
            If provided, gap analysis is included in the report.
        max_allowed_missing : int
            Passed to validate_time_continuity.
        fill_nan : bool
            If True, forward-fill NaN values in-place before analysis
            (safe — XAU rarely has true NaN in good data sources).
        """
        self._gap_detector    = gap_detector
        self._max_missing     = max_allowed_missing
        self._fill_nan        = fill_nan

    def check(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        """
        Run all quality checks.

        Returns
        -------
        (cleaned_df, report)
            cleaned_df : DataFrame with NaNs filled (if fill_nan=True).
            report     : DataQualityReport — log this to SQLite.
        """
        report = DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            checked_at=datetime.utcnow(),
            total_candles=len(df),
        )

        df = df.copy()

        # Run checks in order — each may add warnings/errors
        df = self._check_duplicates(df, report)
        df = self._check_nan(df, report)
        self._check_zero_negative(df, report)
        self._check_ohlc_integrity(df, report)
        self._check_continuity(df, timeframe, report)
        self._check_gaps(df, report)

        # Final verdict
        report.is_usable = len(report.errors) == 0

        logger.info(report.summary_line)
        for w in report.warnings:
            logger.warning("[QC] %s %s | %s", symbol, timeframe, w)
        for e in report.errors:
            logger.error("[QC] %s %s | %s", symbol, timeframe, e)

        return df, report

    # ---------------------------------------------------------------- #
    # Individual checks                                                  #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _check_duplicates(
        df: pd.DataFrame, report: DataQualityReport
    ) -> pd.DataFrame:
        dupes = df.index.duplicated().sum()
        report.duplicate_timestamps = int(dupes)
        if dupes > 0:
            report.warnings.append(f"{dupes} duplicate timestamps — keeping first.")
            df = df[~df.index.duplicated(keep="first")]
        return df

    def _check_nan(
        self, df: pd.DataFrame, report: DataQualityReport
    ) -> pd.DataFrame:
        nan_rows = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
        report.nan_rows_found = nan_rows
        if nan_rows > 0:
            if self._fill_nan:
                df = df.ffill()
                report.nan_rows_filled = nan_rows
                report.warnings.append(
                    f"{nan_rows} rows with NaN prices — forward-filled."
                )
            else:
                report.errors.append(
                    f"{nan_rows} rows with NaN prices — fill_nan=False, data not usable."
                )
        return df

    @staticmethod
    def _check_zero_negative(
        df: pd.DataFrame, report: DataQualityReport
    ) -> None:
        price_cols = ["open", "high", "low", "close"]
        bad = int((df[price_cols] <= 0).any(axis=1).sum())
        report.zero_or_negative_prices = bad
        if bad > 0:
            report.errors.append(
                f"{bad} rows with zero or negative prices — data corrupted."
            )

    @staticmethod
    def _check_ohlc_integrity(
        df: pd.DataFrame, report: DataQualityReport
    ) -> None:
        """
        Each candle must satisfy:
          high >= max(open, close)
          low  <= min(open, close)
          high >= low
        """
        violations: List[OhlcViolation] = []
        for ts, row in df.iterrows():
            reasons = []
            if row["high"] < row["open"] or row["high"] < row["close"]:
                reasons.append("high < open or close")
            if row["low"] > row["open"] or row["low"] > row["close"]:
                reasons.append("low > open or close")
            if row["high"] < row["low"]:
                reasons.append("high < low")
            if reasons:
                violations.append(OhlcViolation(timestamp=ts, reason="; ".join(reasons)))

        report.ohlc_violations = violations
        if violations:
            report.errors.append(
                f"{len(violations)} OHLC integrity violations detected."
            )

    def _check_continuity(
        self,
        df: pd.DataFrame,
        timeframe: str,
        report: DataQualityReport,
    ) -> None:
        try:
            continuity = validate_time_continuity(df, timeframe, self._max_missing)
            report.continuity = continuity
            if not continuity.is_acceptable:
                report.errors.append(
                    f"Time continuity: {continuity.missing_count} missing candles "
                    f"(max allowed: {self._max_missing})."
                )
            elif continuity.missing_count > 0:
                report.warnings.append(
                    f"Time continuity: {continuity.missing_count} missing candles "
                    f"(within threshold of {self._max_missing})."
                )
        except ValueError as exc:
            report.warnings.append(f"Continuity check skipped: {exc}")

    def _check_gaps(
        self, df: pd.DataFrame, report: DataQualityReport
    ) -> None:
        if self._gap_detector is None:
            return
        gap_report = self._gap_detector.scan(df)
        report.gap_report = gap_report
        if gap_report.has_blocking_gap:
            report.errors.append(
                f"Blocking gap detected — "
                f"{gap_report.total_gap_count} gap(s) total."
            )
        elif gap_report.total_gap_count > 0:
            report.warnings.append(
                f"{gap_report.total_gap_count} non-blocking gap(s) detected."
            )


# ------------------------------------------------------------------ #
# Factory                                                              #
# ------------------------------------------------------------------ #

def build_quality_checker(
    market_calendar_config: dict,
    timeframe: str,
    max_allowed_missing: int = 2,
    fill_nan: bool = True,
) -> DataQualityChecker:
    """
    Convenience factory that wires up the GapDetector automatically.

    Parameters
    ----------
    market_calendar_config : dict
        Full market_calendar.yaml dict (contains gap_detection sub-section).
    timeframe : str
        e.g. '5m'. Passed to GapDetector for spacing calculation.
    """
    from core.data.gap_detector import build_gap_detector
    gap_det = build_gap_detector(market_calendar_config, timeframe)
    return DataQualityChecker(
        gap_detector=gap_det,
        max_allowed_missing=max_allowed_missing,
        fill_nan=fill_nan,
    )
