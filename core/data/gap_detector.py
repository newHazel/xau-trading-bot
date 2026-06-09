"""
Gap Detector — Phase 0.5.

Detects three types of gaps in OHLCV data:

  1. TIME GAP   — consecutive timestamps are further apart than expected
                  (missing candles in the series).
  2. PRICE GAP  — current open is far from previous close (ATR-based).
                  Severity: WARNING (>= 1.0× ATR) or BLOCK (>= 1.5× ATR).
  3. WEEKEND GAP — price gap that occurs specifically across the weekend
                  (Friday close → Sunday/Monday open). Longer cooldown.

Each detected gap becomes a GapEvent that can be stored in the
gap_events SQLite table and acted on by the caller.

All thresholds come from config — nothing is hardcoded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Weekday integers (Monday=0 … Sunday=6)
_FRIDAY   = 4
_SATURDAY = 5
_SUNDAY   = 6


# ------------------------------------------------------------------ #
# Data types                                                           #
# ------------------------------------------------------------------ #

class GapType(str, Enum):
    TIME    = "time"
    PRICE   = "price"
    WEEKEND = "weekend"


class GapSeverity(str, Enum):
    INFO    = "info"     # logged only
    WARNING = "warning"  # log + alert, but don't block
    BLOCK   = "block"    # block trading until cooldown expires


@dataclass
class GapEvent:
    """Mirrors the gap_events SQLite table row."""
    timestamp: pd.Timestamp          # timestamp of the candle AFTER the gap
    timeframe: str
    gap_type: GapType
    severity: GapSeverity
    previous_close: Optional[float]  # None for time-only gaps
    current_open: Optional[float]
    gap_size: Optional[float]        # price units
    gap_atr_ratio: Optional[float]   # gap_size / ATR
    missing_candles: int             # 0 for price/weekend gaps
    cooldown_minutes: int            # how long to pause trading after this gap
    action_taken: str                # "warn", "block", "cooldown"


@dataclass
class GapReport:
    """Aggregated result of scanning a full DataFrame."""
    timeframe: str
    total_candles: int
    time_gaps: List[GapEvent] = field(default_factory=list)
    price_gaps: List[GapEvent] = field(default_factory=list)
    weekend_gaps: List[GapEvent] = field(default_factory=list)

    @property
    def all_gaps(self) -> List[GapEvent]:
        return self.time_gaps + self.price_gaps + self.weekend_gaps

    @property
    def has_blocking_gap(self) -> bool:
        return any(g.severity == GapSeverity.BLOCK for g in self.all_gaps)

    @property
    def total_gap_count(self) -> int:
        return len(self.all_gaps)


# ------------------------------------------------------------------ #
# ATR helper (simple, no external dependency)                          #
# ------------------------------------------------------------------ #

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    True Range and ATR — computed without look-ahead.
    Returns a Series aligned to df.index.
    """
    high  = df["high"]
    low   = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


# ------------------------------------------------------------------ #
# Main detector class                                                  #
# ------------------------------------------------------------------ #

class GapDetector:
    """
    Scans a normalised OHLCV DataFrame for gaps.

    Parameters come from config/market_calendar.yaml → gap_detection section.
    """

    def __init__(self, config: dict, timeframe: str) -> None:
        """
        Parameters
        ----------
        config : dict
            The 'gap_detection' sub-dict from market_calendar.yaml.
        timeframe : str
            e.g. '5m', '1h'. Used to compute expected candle spacing.
        """
        self._tf          = timeframe
        self._enabled     = config.get("enabled", True)
        self._max_missing = config.get("max_missing_candles_allowed", 2)
        self._warn_mult   = config.get("price_gap_atr_multiplier_warning", 1.0)
        self._block_mult  = config.get("price_gap_atr_multiplier_block", 1.5)
        self._cooldown    = config.get("cooldown_after_gap_minutes", 60)
        self._wk_cooldown = config.get("weekend_gap_cooldown_minutes", 120)
        self._atr_period  = config.get("atr_period", 14)
        self._offset      = self._tf_to_timedelta(timeframe)

    # ---------------------------------------------------------------- #
    # Public                                                             #
    # ---------------------------------------------------------------- #

    def scan(self, df: pd.DataFrame) -> GapReport:
        """
        Full gap scan over the entire DataFrame.
        Returns a GapReport with all detected gaps.
        """
        report = GapReport(timeframe=self._tf, total_candles=len(df))

        if not self._enabled or df.empty or len(df) < 2:
            return report

        atr = _compute_atr(df, period=self._atr_period)

        for i in range(1, len(df)):
            prev_ts    = df.index[i - 1]
            curr_ts    = df.index[i]
            prev_close = df["close"].iloc[i - 1]
            curr_open  = df["open"].iloc[i]
            curr_atr   = atr.iloc[i]

            # 1. Time gap
            tg = self._check_time_gap(prev_ts, curr_ts, curr_ts)
            if tg:
                report.time_gaps.append(tg)

            # 2. Weekend gap (supersedes regular price gap if on a weekend boundary)
            if self._is_weekend_boundary(prev_ts, curr_ts):
                wg = self._check_price_gap(
                    prev_close, curr_open, curr_atr, curr_ts,
                    force_weekend=True,
                )
                if wg:
                    report.weekend_gaps.append(wg)
                continue   # don't also flag as regular price gap

            # 3. Regular price gap
            pg = self._check_price_gap(prev_close, curr_open, curr_atr, curr_ts)
            if pg:
                report.price_gaps.append(pg)

        self._log_summary(report)
        return report

    def check_single_transition(
        self,
        prev_close: float,
        curr_open: float,
        prev_ts: pd.Timestamp,
        curr_ts: pd.Timestamp,
        atr: float,
    ) -> Optional[GapEvent]:
        """
        Check one candle transition (live/replay use).
        Returns a GapEvent if a gap is detected, else None.
        """
        if self._is_weekend_boundary(prev_ts, curr_ts):
            return self._check_price_gap(
                prev_close, curr_open, atr, curr_ts, force_weekend=True
            )

        # Time gap check
        tg = self._check_time_gap(prev_ts, curr_ts, curr_ts)
        if tg:
            return tg

        return self._check_price_gap(prev_close, curr_open, atr, curr_ts)

    # ---------------------------------------------------------------- #
    # Internal checks                                                    #
    # ---------------------------------------------------------------- #

    def _check_time_gap(
        self,
        prev_ts: pd.Timestamp,
        curr_ts: pd.Timestamp,
        event_ts: pd.Timestamp,
    ) -> Optional[GapEvent]:
        expected_next = prev_ts + self._offset
        if curr_ts <= expected_next:
            return None   # no time gap

        missing = int(round((curr_ts - prev_ts) / self._offset)) - 1
        if missing <= self._max_missing:
            severity = GapSeverity.WARNING
            action   = "warn"
        else:
            severity = GapSeverity.BLOCK
            action   = "block"

        return GapEvent(
            timestamp=event_ts,
            timeframe=self._tf,
            gap_type=GapType.TIME,
            severity=severity,
            previous_close=None,
            current_open=None,
            gap_size=None,
            gap_atr_ratio=None,
            missing_candles=missing,
            cooldown_minutes=self._cooldown if severity == GapSeverity.BLOCK else 0,
            action_taken=action,
        )

    def _check_price_gap(
        self,
        prev_close: float,
        curr_open: float,
        atr: float,
        event_ts: pd.Timestamp,
        force_weekend: bool = False,
    ) -> Optional[GapEvent]:
        if atr <= 0:
            return None

        gap_size  = abs(curr_open - prev_close)
        atr_ratio = gap_size / atr

        if not force_weekend and atr_ratio < self._warn_mult:
            return None   # gap too small to care about

        # Determine severity
        if atr_ratio >= self._block_mult:
            severity = GapSeverity.BLOCK
            action   = "block"
        elif atr_ratio >= self._warn_mult or force_weekend:
            severity = GapSeverity.WARNING
            action   = "warn"
        else:
            return None

        gap_type    = GapType.WEEKEND if force_weekend else GapType.PRICE
        cooldown    = self._wk_cooldown if force_weekend else (
            self._cooldown if severity == GapSeverity.BLOCK else 0
        )

        return GapEvent(
            timestamp=event_ts,
            timeframe=self._tf,
            gap_type=gap_type,
            severity=severity,
            previous_close=prev_close,
            current_open=curr_open,
            gap_size=round(gap_size, 5),
            gap_atr_ratio=round(atr_ratio, 4),
            missing_candles=0,
            cooldown_minutes=cooldown,
            action_taken=action,
        )

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _is_weekend_boundary(prev_ts: pd.Timestamp, curr_ts: pd.Timestamp) -> bool:
        """
        True when prev_ts is on Friday and curr_ts is on Sunday or Monday,
        indicating a weekend gap regardless of gap size.
        """
        prev_day = prev_ts.weekday()
        curr_day = curr_ts.weekday()
        return prev_day == _FRIDAY and curr_day in (_SUNDAY, 0)   # 0 = Monday

    @staticmethod
    def _tf_to_timedelta(timeframe: str) -> timedelta:
        _MAP = {
            "1m":  timedelta(minutes=1),
            "5m":  timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h":  timedelta(hours=1),
            "4h":  timedelta(hours=4),
            "1d":  timedelta(days=1),
        }
        td = _MAP.get(timeframe)
        if td is None:
            raise ValueError(
                f"Unknown timeframe '{timeframe}' for gap detection. "
                f"Supported: {list(_MAP.keys())}"
            )
        return td

    @staticmethod
    def _log_summary(report: GapReport) -> None:
        if report.total_gap_count == 0:
            return
        logger.info(
            "[GapDetector] %s | time_gaps=%d price_gaps=%d weekend_gaps=%d | blocking=%s",
            report.timeframe,
            len(report.time_gaps),
            len(report.price_gaps),
            len(report.weekend_gaps),
            report.has_blocking_gap,
        )


# ------------------------------------------------------------------ #
# Convenience factory                                                   #
# ------------------------------------------------------------------ #

def build_gap_detector(market_calendar_config: dict, timeframe: str) -> GapDetector:
    """
    Build a GapDetector from the full market_calendar.yaml config dict.
    Extracts the 'gap_detection' sub-section automatically.
    """
    gap_cfg = market_calendar_config.get("gap_detection", {})
    return GapDetector(config=gap_cfg, timeframe=timeframe)
