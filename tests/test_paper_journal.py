"""Tests for PaperJournal — Phase 7.3."""

import pytest
from datetime import datetime, timezone
from paper_trading.paper_journal import PaperJournal, JournalEntry


@pytest.fixture
def journal():
    return PaperJournal()


TS1 = datetime(2026, 1, 21, 10, 30, tzinfo=timezone.utc)
TS2 = datetime(2026, 1, 21, 11, 30, tzinfo=timezone.utc)


class TestAddEntry:
    def test_add_creates_entry(self, journal):
        e = journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1)
        assert e.entry_id == 1
        assert e.setup_id == "XAU-001"
        assert e.result == "pending"
        assert journal.total_entries == 1

    def test_ids_increment(self, journal):
        e1 = journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1)
        e2 = journal.add_entry("XAU-002", "short", "A", 2000.0, 2005.0, 1990.0, 1982.5, TS1)
        assert e2.entry_id == 2

    def test_conditions_stored(self, journal):
        conds = {"htf_bias": True, "sweep": True, "fvg": True}
        e = journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1,
                              mandatory_conditions=conds)
        assert e.mandatory_conditions["htf_bias"]


class TestCloseEntry:
    def test_close_as_win(self, journal):
        e = journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1)
        closed = journal.close_entry(e.entry_id, exit_price=2017.5, exit_time=TS2,
                                     exit_type="tp2_hit", net_r=3.2, gross_r=3.5)
        assert closed.result == "win"
        assert closed.exit_price == 2017.5

    def test_close_as_loss(self, journal):
        e = journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1)
        closed = journal.close_entry(e.entry_id, exit_price=1995.0, exit_time=TS2,
                                     exit_type="sl_hit", net_r=-1.2, gross_r=-1.0)
        assert closed.result == "loss"

    def test_close_as_breakeven(self, journal):
        e = journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1)
        closed = journal.close_entry(e.entry_id, exit_price=2000.35, exit_time=TS2,
                                     exit_type="trailing_sl", net_r=0.05, gross_r=0.07)
        assert closed.result == "breakeven"

    def test_close_nonexistent_returns_none(self, journal):
        result = journal.close_entry(999, exit_price=2000.0, exit_time=TS2,
                                     exit_type="sl_hit", net_r=-1.0, gross_r=-1.0)
        assert result is None

    def test_notes_appended(self, journal):
        e = journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1, notes="initial")
        journal.close_entry(e.entry_id, 2010.0, TS2, "tp1_hit", 2.0, 2.0, notes="closed at TP1")
        assert "initial" in e.notes
        assert "closed at TP1" in e.notes


class TestViolations:
    def test_add_violation(self, journal):
        e = journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1)
        journal.add_violation(e.entry_id, "entered during news block")
        assert len(e.violations) == 1
        assert "news" in e.violations[0]

    def test_get_violations(self, journal):
        e1 = journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1)
        journal.add_entry("XAU-002", "short", "A", 2000.0, 2005.0, 1990.0, 1982.5, TS1)
        journal.add_violation(e1.entry_id, "violation 1")
        violations = journal.get_violations()
        assert len(violations) == 1


class TestQueries:
    def test_get_open(self, journal):
        journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1)
        e2 = journal.add_entry("XAU-002", "short", "A", 2000.0, 2005.0, 1990.0, 1982.5, TS1)
        journal.close_entry(e2.entry_id, 2005.0, TS2, "sl_hit", -1.0, -1.0)
        assert len(journal.get_open()) == 1
        assert len(journal.get_closed()) == 1

    def test_get_by_grade(self, journal):
        journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1)
        journal.add_entry("XAU-002", "short", "A", 2000.0, 2005.0, 1990.0, 1982.5, TS1)
        assert len(journal.get_by_grade("A+")) == 1
        assert len(journal.get_by_grade("A")) == 1

    def test_get_by_direction(self, journal):
        journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1)
        journal.add_entry("XAU-002", "short", "A", 2000.0, 2005.0, 1990.0, 1982.5, TS1)
        assert len(journal.get_by_direction("long")) == 1
        assert len(journal.get_by_direction("short")) == 1


class TestClear:
    def test_clear(self, journal):
        journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1)
        journal.clear()
        assert journal.total_entries == 0


class TestToDict:
    def test_entry_to_dict(self, journal):
        e = journal.add_entry("XAU-001", "long", "A+", 2000.0, 1995.0, 2010.0, 2017.5, TS1)
        d = e.to_dict()
        assert d["setup_id"] == "XAU-001"
        assert d["result"] == "pending"
        assert "entry_price" in d
