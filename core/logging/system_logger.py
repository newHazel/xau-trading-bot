"""
System Logger — writes gap events, data quality results, and health checks.
Also handles candle storage and news event persistence.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.logging.db import Database
from core.data.gap_detector import GapEvent, GapReport
from core.data.data_quality_checker import DataQualityReport

logger = logging.getLogger(__name__)


class SystemLogger:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ---------------------------------------------------------------- #
    # Gap events                                                         #
    # ---------------------------------------------------------------- #

    def log_gap_event(self, gap: GapEvent) -> None:
        try:
            self._db.execute(
                """
                INSERT INTO gap_events
                    (timestamp, timeframe, gap_type, severity, previous_close,
                     current_open, gap_size, gap_atr_ratio, missing_candles,
                     cooldown_minutes, action_taken)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    gap.timestamp.isoformat(),
                    gap.timeframe,
                    gap.gap_type.value,
                    gap.severity.value,
                    gap.previous_close,
                    gap.current_open,
                    gap.gap_size,
                    gap.gap_atr_ratio,
                    gap.missing_candles,
                    gap.cooldown_minutes,
                    gap.action_taken,
                ),
            )
            logger.debug("[SystemLogger] Gap event logged: %s %s", gap.gap_type, gap.severity)
        except Exception as exc:
            logger.error("[SystemLogger] Failed to log gap event: %s", exc)

    def log_gap_report(self, report: GapReport) -> None:
        """Persist all gap events from a full scan report."""
        for gap in report.all_gaps:
            self.log_gap_event(gap)

    # ---------------------------------------------------------------- #
    # Data quality log                                                   #
    # ---------------------------------------------------------------- #

    def log_quality_report(self, report: DataQualityReport) -> None:
        """Write one row per quality dimension from a DataQualityReport."""
        ts = report.checked_at.strftime("%Y-%m-%dT%H:%M:%SZ")

        checks = [
            ("ohlc_integrity",    "ok" if not report.ohlc_violations else "error",
             {"violations": len(report.ohlc_violations)}),
            ("nan_check",         "ok" if report.nan_rows_found == 0 else "warning",
             {"found": report.nan_rows_found, "filled": report.nan_rows_filled}),
            ("zero_negative",     "ok" if report.zero_or_negative_prices == 0 else "error",
             {"count": report.zero_or_negative_prices}),
            ("duplicates",        "ok" if report.duplicate_timestamps == 0 else "warning",
             {"count": report.duplicate_timestamps}),
            ("continuity",        ("ok" if report.missing_candles == 0
                                   else ("warning" if report.continuity and report.continuity.is_acceptable
                                         else "error")),
             {"missing": report.missing_candles}),
            ("gap_detection",     ("ok" if not report.gap_report or report.gap_report.total_gap_count == 0
                                   else ("error" if report.has_blocking_gap else "warning")),
             {"total_gaps": report.gap_report.total_gap_count if report.gap_report else 0}),
            ("overall",           "ok" if report.is_usable else "error",
             {"warnings": len(report.warnings), "errors": len(report.errors)}),
        ]

        rows = [
            (ts, report.symbol, report.timeframe, name, status, json.dumps(details))
            for name, status, details in checks
        ]
        try:
            self._db.executemany(
                "INSERT INTO data_quality_log (timestamp, symbol, timeframe, check_name, status, details_json) "
                "VALUES (?,?,?,?,?,?)",
                rows,
            )
            logger.debug(
                "[SystemLogger] Quality report logged for %s %s",
                report.symbol, report.timeframe,
            )
        except Exception as exc:
            logger.error("[SystemLogger] Failed to log quality report: %s", exc)

    # ---------------------------------------------------------------- #
    # Health checks                                                      #
    # ---------------------------------------------------------------- #

    def log_health_check(
        self,
        check_name: str,
        status: str,
        message: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            self._db.execute(
                "INSERT INTO health_checks (timestamp, check_name, status, message, duration_ms) "
                "VALUES (?,?,?,?,?)",
                (ts, check_name, status, message, duration_ms),
            )
        except Exception as exc:
            logger.error("[SystemLogger] Failed to log health check %s: %s", check_name, exc)

    # ---------------------------------------------------------------- #
    # Candles                                                            #
    # ---------------------------------------------------------------- #

    def store_candles(
        self,
        df,   # pd.DataFrame with UTC DatetimeIndex
        symbol: str,
        timeframe: str,
        source: str,
    ) -> int:
        """
        Bulk-insert candles. Skips rows that already exist (UNIQUE constraint).
        Returns the number of new rows inserted.
        """
        rows = [
            (
                symbol, timeframe,
                ts.isoformat(),
                float(row["open"]), float(row["high"]),
                float(row["low"]),  float(row["close"]),
                float(row["volume"]), source,
            )
            for ts, row in df.iterrows()
        ]
        try:
            self._db.executemany(
                "INSERT OR IGNORE INTO candles "
                "(symbol, timeframe, timestamp, open, high, low, close, volume, source) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
            logger.debug(
                "[SystemLogger] Stored up to %d candles for %s %s from %s",
                len(rows), symbol, timeframe, source,
            )
            return len(rows)
        except Exception as exc:
            logger.error("[SystemLogger] Failed to store candles: %s", exc)
            return 0

    # ---------------------------------------------------------------- #
    # News events                                                        #
    # ---------------------------------------------------------------- #

    def log_news_event(self, event: Dict[str, Any]) -> None:
        try:
            self._db.execute(
                "INSERT OR IGNORE INTO news_events "
                "(event_time, currency, impact, tier, title, actual, forecast, previous, source) "
                "VALUES (:event_time,:currency,:impact,:tier,:title,:actual,:forecast,:previous,:source)",
                event,
            )
        except Exception as exc:
            logger.error("[SystemLogger] Failed to log news event: %s", exc)
