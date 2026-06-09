"""
Source Normalizer — Phase 0.3.

Ensures every DataFrame that enters the system has:
  - canonical symbol name
  - UTC tz-aware DatetimeIndex
  - correct OHLCV column names and float64 types
  - acceptable decimal precision for XAU prices
  - no unexpected time gaps (continuity check)

All functions are pure — they return a new object or raise on violation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Symbol aliases → canonical name                                      #
# ------------------------------------------------------------------ #

_SYMBOL_ALIASES: Dict[str, str] = {
    # XAU variants
    "XAUUSD":   "XAUUSD",
    "XAUUSDT":  "XAUUSD",
    "GC=F":     "XAUUSD",
    "GOLD":     "XAUUSD",
    "XAU":      "XAUUSD",
    # DXY variants
    "DXY":      "DXY",
    "DX-Y.NYB": "DXY",
    "USDX":     "DXY",
    # US10Y variants
    "US10Y":    "US10Y",
    "^TNX":     "US10Y",
}

# Expected timeframe → pandas offset alias for resampling / gap detection
_TF_TO_OFFSET: Dict[str, str] = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
}

# XAU price precision: 2 decimal places standard, warn if > 5
_XAU_MAX_DECIMALS = 5
_XAU_MIN_PRICE    = 500.0    # sanity lower bound
_XAU_MAX_PRICE    = 10_000.0 # sanity upper bound

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


# ------------------------------------------------------------------ #
# Result types                                                         #
# ------------------------------------------------------------------ #

@dataclass
class ContinuityReport:
    """Summary of time-continuity validation."""
    timeframe: str
    total_candles: int
    expected_candles: int
    missing_count: int
    missing_timestamps: List[pd.Timestamp] = field(default_factory=list)
    is_acceptable: bool = True           # False if gaps exceed threshold
    max_allowed_missing: int = 2


@dataclass
class PrecisionReport:
    """Summary of decimal precision check."""
    symbol: str
    max_decimals_found: int
    rows_outside_price_range: int
    is_acceptable: bool = True


# ------------------------------------------------------------------ #
# 1. normalize_symbol                                                  #
# ------------------------------------------------------------------ #

def normalize_symbol(raw_symbol: str) -> str:
    """
    Map any known symbol alias to the canonical form.

    Raises ValueError for completely unknown symbols so the caller knows
    immediately rather than silently processing wrong data.
    """
    key = raw_symbol.strip().upper()
    canonical = _SYMBOL_ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            f"Unknown symbol '{raw_symbol}'. "
            f"Known aliases: {sorted(_SYMBOL_ALIASES.keys())}"
        )
    return canonical


# ------------------------------------------------------------------ #
# 2. normalize_timezone                                                #
# ------------------------------------------------------------------ #

def normalize_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame index is a UTC tz-aware DatetimeIndex.

    Accepts:
      - tz-naive index  → assumed UTC, localizes to UTC
      - tz-aware index  → converts to UTC
      - non-DatetimeIndex → raises TypeError
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"Expected DatetimeIndex, got {type(df.index).__name__}. "
            "Call normalize_ohlcv_schema() first if the timestamp is a column."
        )

    idx = df.index
    if idx.tz is None:
        logger.debug("normalize_timezone: tz-naive index — assuming UTC.")
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    df = df.copy()
    df.index = idx
    df.index.name = "timestamp"
    return df


# ------------------------------------------------------------------ #
# 3. normalize_ohlcv_schema                                            #
# ------------------------------------------------------------------ #

def normalize_ohlcv_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce the canonical OHLCV schema:
      - Index named 'timestamp', UTC-aware DatetimeIndex.
      - Columns: open, high, low, close, volume — all float64.
      - Drops any extra columns silently.
      - Raises ValueError if required columns are missing after name normalisation.

    Column name normalisation (case-insensitive):
      Open/HIGH/OPEN etc. → open
    """
    df = df.copy()

    # If 'timestamp' is a column rather than the index, set it as index
    lower_cols = {c.lower(): c for c in df.columns}
    if "timestamp" in lower_cols and not isinstance(df.index, pd.DatetimeIndex):
        df = df.rename(columns={lower_cols["timestamp"]: "timestamp"})
        df = df.set_index("timestamp")

    # Normalise column names to lower-case
    df.columns = [c.lower() for c in df.columns]

    # Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalisation: {missing}")

    # Keep only the required columns (drop extras)
    df = df[REQUIRED_COLUMNS]

    # Cast to float64
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.DatetimeIndex(df.index)
        except Exception as exc:
            raise TypeError(f"Cannot convert index to DatetimeIndex: {exc}") from exc

    df.index.name = "timestamp"

    # Sort chronologically
    df = df.sort_index()

    # Apply timezone normalisation
    df = normalize_timezone(df)

    return df


# ------------------------------------------------------------------ #
# 4. check_decimal_precision                                           #
# ------------------------------------------------------------------ #

def check_decimal_precision(
    df: pd.DataFrame,
    symbol: str = "XAUUSD",
) -> PrecisionReport:
    """
    Verify price values have reasonable decimal precision and range.

    For XAU/USD:
      - Prices typically 4-5 decimal places maximum
      - Sane range: 500 – 10 000

    Returns a PrecisionReport. Does NOT raise — caller decides what to do.
    """
    price_cols = ["open", "high", "low", "close"]
    max_dec = 0

    for col in price_cols:
        if col not in df.columns:
            continue
        for val in df[col].dropna():
            s = f"{val:.10f}".rstrip("0")
            if "." in s:
                decimals = len(s.split(".")[1])
                max_dec = max(max_dec, decimals)

    # Count rows where any price is outside the expected range
    price_df = df[price_cols].dropna()
    out_of_range = int(
        ((price_df < _XAU_MIN_PRICE) | (price_df > _XAU_MAX_PRICE)).any(axis=1).sum()
    )

    is_acceptable = (max_dec <= _XAU_MAX_DECIMALS) and (out_of_range == 0)

    report = PrecisionReport(
        symbol=symbol,
        max_decimals_found=max_dec,
        rows_outside_price_range=out_of_range,
        is_acceptable=is_acceptable,
    )

    if not is_acceptable:
        logger.warning(
            "[Normalizer] Precision issues for %s: max_decimals=%d, out_of_range_rows=%d",
            symbol, max_dec, out_of_range,
        )

    return report


# ------------------------------------------------------------------ #
# 5. validate_time_continuity                                          #
# ------------------------------------------------------------------ #

def validate_time_continuity(
    df: pd.DataFrame,
    timeframe: str,
    max_allowed_missing: int = 2,
) -> ContinuityReport:
    """
    Check that the DatetimeIndex has no unexpected gaps.

    Builds the full expected index between first and last timestamp at the
    given timeframe frequency, then compares against the actual index.

    Does NOT raise — returns a ContinuityReport so the caller can decide
    whether to block, warn, or continue.
    """
    offset = _TF_TO_OFFSET.get(timeframe)
    if offset is None:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. "
            f"Supported: {list(_TF_TO_OFFSET.keys())}"
        )

    if df.empty:
        return ContinuityReport(
            timeframe=timeframe,
            total_candles=0,
            expected_candles=0,
            missing_count=0,
            is_acceptable=True,
            max_allowed_missing=max_allowed_missing,
        )

    start = df.index[0]
    end   = df.index[-1]

    expected_idx = pd.date_range(start=start, end=end, freq=offset, tz="UTC")
    actual_set   = set(df.index)
    missing      = [ts for ts in expected_idx if ts not in actual_set]

    is_acceptable = len(missing) <= max_allowed_missing

    report = ContinuityReport(
        timeframe=timeframe,
        total_candles=len(df),
        expected_candles=len(expected_idx),
        missing_count=len(missing),
        missing_timestamps=missing[:50],  # cap to avoid huge objects
        is_acceptable=is_acceptable,
        max_allowed_missing=max_allowed_missing,
    )

    if missing:
        level = logging.WARNING if is_acceptable else logging.ERROR
        logger.log(
            level,
            "[Normalizer] Time continuity: %d missing candles out of %d expected (%s). "
            "Acceptable threshold: %d.",
            len(missing), len(expected_idx), timeframe, max_allowed_missing,
        )

    return report


# ------------------------------------------------------------------ #
# Convenience: run all checks in one call                              #
# ------------------------------------------------------------------ #

def normalize_and_validate(
    df: pd.DataFrame,
    raw_symbol: str,
    timeframe: str,
    max_allowed_missing: int = 2,
) -> Tuple[pd.DataFrame, str, PrecisionReport, ContinuityReport]:
    """
    Full normalization pipeline:
      1. normalize_symbol
      2. normalize_ohlcv_schema  (includes timezone)
      3. check_decimal_precision
      4. validate_time_continuity

    Returns (normalized_df, canonical_symbol, precision_report, continuity_report).
    Raises on hard errors (unknown symbol, missing columns, bad index).
    """
    canonical = normalize_symbol(raw_symbol)
    df_norm   = normalize_ohlcv_schema(df)
    precision = check_decimal_precision(df_norm, symbol=canonical)
    continuity = validate_time_continuity(df_norm, timeframe, max_allowed_missing)
    return df_norm, canonical, precision, continuity
