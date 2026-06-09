"""Tests for Setup ID Manager — Phase 4.5."""

import pytest
from datetime import datetime
from core.engine.setup_id_manager import SetupIDManager


@pytest.fixture
def mgr():
    return SetupIDManager()


class TestGenerate:
    def test_format(self, mgr):
        dt = datetime(2026, 3, 15, 14, 32)
        sid = mgr.generate(dt, "long", "FVG7842")
        assert sid == "XAU-20260315-1432-LONG-FVG7842"

    def test_short_direction(self, mgr):
        dt = datetime(2026, 3, 15, 9, 5)
        sid = mgr.generate(dt, "short", "OB123")
        assert sid == "XAU-20260315-0905-SHORT-OB123"

    def test_custom_symbol(self, mgr):
        dt = datetime(2026, 1, 1, 0, 0)
        sid = mgr.generate(dt, "long", "Z1", symbol="GOLD")
        assert sid.startswith("GOLD-")


class TestRegister:
    def test_register_new(self, mgr):
        assert mgr.register("XAU-20260315-1432-LONG-FVG1") is True

    def test_register_duplicate(self, mgr):
        mgr.register("XAU-20260315-1432-LONG-FVG1")
        assert mgr.register("XAU-20260315-1432-LONG-FVG1") is False

    def test_is_duplicate(self, mgr):
        mgr.register("ID1")
        assert mgr.is_duplicate("ID1") is True
        assert mgr.is_duplicate("ID2") is False


class TestGenerateAndRegister:
    def test_first_call_returns_id(self, mgr):
        dt = datetime(2026, 3, 15, 14, 32)
        sid = mgr.generate_and_register(dt, "long", "FVG1")
        assert sid is not None

    def test_duplicate_returns_none(self, mgr):
        dt = datetime(2026, 3, 15, 14, 32)
        mgr.generate_and_register(dt, "long", "FVG1")
        sid = mgr.generate_and_register(dt, "long", "FVG1")
        assert sid is None


class TestCount:
    def test_count(self, mgr):
        assert mgr.issued_count == 0
        mgr.register("A")
        mgr.register("B")
        assert mgr.issued_count == 2

    def test_reset(self, mgr):
        mgr.register("A")
        mgr.reset()
        assert mgr.issued_count == 0
        assert mgr.is_duplicate("A") is False
