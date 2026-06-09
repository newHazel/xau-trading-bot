"""
Health Check System — Phase 9.1.

Runs configurable health checks:
  - data_freshness: last candle within threshold
  - db_writable: can write to database
  - telegram_responsive: telegram bot reachable
  - news_calendar_loaded: news events loaded
  - memory_usage_ok: memory below threshold
  - no_repeated_errors: no repeated errors in last hour
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass(frozen=True)
class HealthCheckResult:
    name: str
    status: CheckStatus
    detail: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class HealthReport:
    checks: List[HealthCheckResult]
    overall_healthy: bool
    failed_count: int
    warned_count: int
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_healthy": self.overall_healthy,
            "failed_count": self.failed_count,
            "warned_count": self.warned_count,
            "timestamp": self.timestamp.isoformat(),
            "checks": [c.to_dict() for c in self.checks],
        }


CheckFn = Callable[[], HealthCheckResult]


class HealthChecker:
    """Runs health checks and reports system status."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._enabled_checks = set(config.get("checks", [
            "data_freshness", "db_writable", "telegram_responsive",
            "news_calendar_loaded", "memory_usage_ok", "no_repeated_errors_in_last_hour",
        ]))
        self._data_freshness_minutes = config.get("data_freshness_max_minutes", 5)
        self._memory_max_mb = config.get("memory_max_mb", 500)
        self._error_threshold = config.get("error_threshold_per_hour", 10)

        self._custom_checks: Dict[str, CheckFn] = {}
        self._last_candle_time: Optional[datetime] = None
        self._db_writable: Optional[bool] = None
        self._telegram_ok: Optional[bool] = None
        self._news_loaded: Optional[bool] = None
        self._memory_mb: float = 0
        self._errors_last_hour: int = 0

    def register_check(self, name: str, fn: CheckFn) -> None:
        self._custom_checks[name] = fn

    def update_state(
        self,
        last_candle_time: Optional[datetime] = None,
        db_writable: Optional[bool] = None,
        telegram_ok: Optional[bool] = None,
        news_loaded: Optional[bool] = None,
        memory_mb: Optional[float] = None,
        errors_last_hour: Optional[int] = None,
    ) -> None:
        if last_candle_time is not None:
            self._last_candle_time = last_candle_time
        if db_writable is not None:
            self._db_writable = db_writable
        if telegram_ok is not None:
            self._telegram_ok = telegram_ok
        if news_loaded is not None:
            self._news_loaded = news_loaded
        if memory_mb is not None:
            self._memory_mb = memory_mb
        if errors_last_hour is not None:
            self._errors_last_hour = errors_last_hour

    def run_checks(self, now: Optional[datetime] = None) -> HealthReport:
        now = now or datetime.utcnow()
        results: List[HealthCheckResult] = []

        if "data_freshness" in self._enabled_checks:
            results.append(self._check_data_freshness(now))
        if "db_writable" in self._enabled_checks:
            results.append(self._check_db_writable(now))
        if "telegram_responsive" in self._enabled_checks:
            results.append(self._check_telegram(now))
        if "news_calendar_loaded" in self._enabled_checks:
            results.append(self._check_news(now))
        if "memory_usage_ok" in self._enabled_checks:
            results.append(self._check_memory(now))
        if "no_repeated_errors_in_last_hour" in self._enabled_checks:
            results.append(self._check_errors(now))

        for name, fn in self._custom_checks.items():
            if name in self._enabled_checks:
                results.append(fn())

        failed = sum(1 for r in results if r.status == CheckStatus.FAIL)
        warned = sum(1 for r in results if r.status == CheckStatus.WARN)

        return HealthReport(
            checks=results,
            overall_healthy=failed == 0,
            failed_count=failed,
            warned_count=warned,
            timestamp=now,
        )

    def _check_data_freshness(self, now: datetime) -> HealthCheckResult:
        if self._last_candle_time is None:
            return HealthCheckResult("data_freshness", CheckStatus.FAIL, "no candle data received", now)
        age = (now - self._last_candle_time).total_seconds() / 60
        if age > self._data_freshness_minutes:
            return HealthCheckResult("data_freshness", CheckStatus.FAIL,
                                     f"last candle {age:.1f}min ago (max {self._data_freshness_minutes})", now)
        return HealthCheckResult("data_freshness", CheckStatus.PASS, f"last candle {age:.1f}min ago", now)

    def _check_db_writable(self, now: datetime) -> HealthCheckResult:
        if self._db_writable is None:
            return HealthCheckResult("db_writable", CheckStatus.WARN, "not checked yet", now)
        if not self._db_writable:
            return HealthCheckResult("db_writable", CheckStatus.FAIL, "database not writable", now)
        return HealthCheckResult("db_writable", CheckStatus.PASS, "database writable", now)

    def _check_telegram(self, now: datetime) -> HealthCheckResult:
        if self._telegram_ok is None:
            return HealthCheckResult("telegram_responsive", CheckStatus.WARN, "not checked yet", now)
        if not self._telegram_ok:
            return HealthCheckResult("telegram_responsive", CheckStatus.FAIL, "telegram not responsive", now)
        return HealthCheckResult("telegram_responsive", CheckStatus.PASS, "telegram ok", now)

    def _check_news(self, now: datetime) -> HealthCheckResult:
        if self._news_loaded is None:
            return HealthCheckResult("news_calendar_loaded", CheckStatus.WARN, "not checked yet", now)
        if not self._news_loaded:
            return HealthCheckResult("news_calendar_loaded", CheckStatus.FAIL, "news calendar not loaded", now)
        return HealthCheckResult("news_calendar_loaded", CheckStatus.PASS, "news calendar loaded", now)

    def _check_memory(self, now: datetime) -> HealthCheckResult:
        if self._memory_mb > self._memory_max_mb:
            return HealthCheckResult("memory_usage_ok", CheckStatus.FAIL,
                                     f"memory {self._memory_mb:.0f}MB > max {self._memory_max_mb}MB", now)
        if self._memory_mb > self._memory_max_mb * 0.8:
            return HealthCheckResult("memory_usage_ok", CheckStatus.WARN,
                                     f"memory {self._memory_mb:.0f}MB approaching limit", now)
        return HealthCheckResult("memory_usage_ok", CheckStatus.PASS, f"memory {self._memory_mb:.0f}MB", now)

    def _check_errors(self, now: datetime) -> HealthCheckResult:
        if self._errors_last_hour >= self._error_threshold:
            return HealthCheckResult("no_repeated_errors_in_last_hour", CheckStatus.FAIL,
                                     f"{self._errors_last_hour} errors (threshold {self._error_threshold})", now)
        if self._errors_last_hour > 0:
            return HealthCheckResult("no_repeated_errors_in_last_hour", CheckStatus.WARN,
                                     f"{self._errors_last_hour} errors", now)
        return HealthCheckResult("no_repeated_errors_in_last_hour", CheckStatus.PASS, "no errors", now)
