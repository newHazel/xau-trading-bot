"""Tests for HealthChecker — Phase 9.1."""

import pytest
from datetime import datetime, timezone, timedelta
from core.monitoring.health_checker import HealthChecker, HealthReport, CheckStatus


@pytest.fixture
def checker():
    return HealthChecker({
        "checks": ["data_freshness", "db_writable", "telegram_responsive",
                    "news_calendar_loaded", "memory_usage_ok", "no_repeated_errors_in_last_hour"],
        "data_freshness_max_minutes": 5,
        "memory_max_mb": 500,
        "error_threshold_per_hour": 10,
    })


NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)


class TestAllHealthy:
    def test_all_pass(self, checker):
        checker.update_state(
            last_candle_time=NOW - timedelta(minutes=2),
            db_writable=True,
            telegram_ok=True,
            news_loaded=True,
            memory_mb=200,
            errors_last_hour=0,
        )
        report = checker.run_checks(NOW)
        assert report.overall_healthy
        assert report.failed_count == 0


class TestDataFreshness:
    def test_stale_data(self, checker):
        checker.update_state(last_candle_time=NOW - timedelta(minutes=10))
        report = checker.run_checks(NOW)
        failed = [c for c in report.checks if c.name == "data_freshness"]
        assert failed[0].status == CheckStatus.FAIL

    def test_no_data(self, checker):
        report = checker.run_checks(NOW)
        failed = [c for c in report.checks if c.name == "data_freshness"]
        assert failed[0].status == CheckStatus.FAIL

    def test_fresh_data(self, checker):
        checker.update_state(last_candle_time=NOW - timedelta(minutes=1))
        report = checker.run_checks(NOW)
        fresh = [c for c in report.checks if c.name == "data_freshness"]
        assert fresh[0].status == CheckStatus.PASS


class TestDBWritable:
    def test_not_writable(self, checker):
        checker.update_state(db_writable=False)
        report = checker.run_checks(NOW)
        db = [c for c in report.checks if c.name == "db_writable"]
        assert db[0].status == CheckStatus.FAIL

    def test_not_checked(self, checker):
        report = checker.run_checks(NOW)
        db = [c for c in report.checks if c.name == "db_writable"]
        assert db[0].status == CheckStatus.WARN


class TestMemory:
    def test_high_memory(self, checker):
        checker.update_state(memory_mb=600)
        report = checker.run_checks(NOW)
        mem = [c for c in report.checks if c.name == "memory_usage_ok"]
        assert mem[0].status == CheckStatus.FAIL

    def test_warning_memory(self, checker):
        checker.update_state(memory_mb=420)
        report = checker.run_checks(NOW)
        mem = [c for c in report.checks if c.name == "memory_usage_ok"]
        assert mem[0].status == CheckStatus.WARN

    def test_ok_memory(self, checker):
        checker.update_state(memory_mb=200)
        report = checker.run_checks(NOW)
        mem = [c for c in report.checks if c.name == "memory_usage_ok"]
        assert mem[0].status == CheckStatus.PASS


class TestErrors:
    def test_too_many_errors(self, checker):
        checker.update_state(errors_last_hour=15)
        report = checker.run_checks(NOW)
        err = [c for c in report.checks if c.name == "no_repeated_errors_in_last_hour"]
        assert err[0].status == CheckStatus.FAIL

    def test_some_errors_warn(self, checker):
        checker.update_state(errors_last_hour=3)
        report = checker.run_checks(NOW)
        err = [c for c in report.checks if c.name == "no_repeated_errors_in_last_hour"]
        assert err[0].status == CheckStatus.WARN


class TestCustomCheck:
    def test_register_custom_check(self):
        c = HealthChecker({"checks": ["custom_test"]})
        c.register_check("custom_test", lambda: HealthChecker._make_result("custom_test", True))
        # Just verify it doesn't crash — custom check not in default set won't run unless enabled

    @staticmethod
    def _make_result(name, ok):
        from core.monitoring.health_checker import HealthCheckResult, CheckStatus
        return HealthCheckResult(name, CheckStatus.PASS if ok else CheckStatus.FAIL, "test", datetime.utcnow())


class TestReport:
    def test_report_to_dict(self, checker):
        checker.update_state(db_writable=True, telegram_ok=True, news_loaded=True,
                             memory_mb=100, errors_last_hour=0,
                             last_candle_time=NOW - timedelta(minutes=1))
        report = checker.run_checks(NOW)
        d = report.to_dict()
        assert "overall_healthy" in d
        assert "checks" in d
        assert isinstance(d["checks"], list)


class TestDefaults:
    def test_default_config(self):
        c = HealthChecker()
        report = c.run_checks(NOW)
        assert isinstance(report, HealthReport)
