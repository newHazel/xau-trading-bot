"""Tests for Setup Expiry Manager — Phase 4.6."""

import pytest
from datetime import datetime, timedelta
from core.engine.setup_expiry_manager import SetupExpiryManager

CONFIG = {
    "setup_expiry_minutes": 90,
    "setup_expiry_starts_from": "fvg_creation",
    "max_active_setups_per_symbol": 1,
}


@pytest.fixture
def sem():
    return SetupExpiryManager(CONFIG)


class TestRegister:
    def test_register_success(self, sem):
        now = datetime(2026, 3, 18, 12, 0)
        assert sem.register_setup("S1", "long", now) is True
        assert sem.active_count == 1

    def test_register_duplicate_fails(self, sem):
        now = datetime(2026, 3, 18, 12, 0)
        sem.register_setup("S1", "long", now)
        assert sem.register_setup("S1", "long", now) is False

    def test_max_active_limit(self, sem):
        now = datetime(2026, 3, 18, 12, 0)
        sem.register_setup("S1", "long", now)
        assert sem.register_setup("S2", "short", now) is False


class TestExpiry:
    def test_not_expired_within_window(self, sem):
        now = datetime(2026, 3, 18, 12, 0)
        sem.register_setup("S1", "long", now)
        assert sem.is_expired("S1", now + timedelta(minutes=60)) is False

    def test_expired_after_window(self, sem):
        now = datetime(2026, 3, 18, 12, 0)
        sem.register_setup("S1", "long", now)
        assert sem.is_expired("S1", now + timedelta(minutes=91)) is True

    def test_exactly_at_expiry(self, sem):
        now = datetime(2026, 3, 18, 12, 0)
        sem.register_setup("S1", "long", now)
        assert sem.is_expired("S1", now + timedelta(minutes=90)) is True

    def test_unknown_setup_is_expired(self, sem):
        assert sem.is_expired("UNKNOWN", datetime.now()) is True


class TestCheckAndExpire:
    def test_removes_expired(self, sem):
        now = datetime(2026, 3, 18, 12, 0)
        sem.register_setup("S1", "long", now)
        expired = sem.check_and_expire(now + timedelta(minutes=100))
        assert "S1" in expired
        assert sem.active_count == 0

    def test_keeps_active(self, sem):
        now = datetime(2026, 3, 18, 12, 0)
        sem.register_setup("S1", "long", now)
        expired = sem.check_and_expire(now + timedelta(minutes=30))
        assert expired == []
        assert sem.active_count == 1


class TestRemove:
    def test_remove_existing(self, sem):
        now = datetime(2026, 3, 18, 12, 0)
        sem.register_setup("S1", "long", now)
        assert sem.remove_setup("S1") is True
        assert sem.active_count == 0

    def test_remove_missing(self, sem):
        assert sem.remove_setup("NOPE") is False


class TestGetSetup:
    def test_get_existing(self, sem):
        now = datetime(2026, 3, 18, 12, 0)
        sem.register_setup("S1", "long", now)
        s = sem.get_setup("S1")
        assert s is not None
        assert s.direction == "long"
        assert s.origin == "fvg_creation"

    def test_get_missing(self, sem):
        assert sem.get_setup("NOPE") is None
