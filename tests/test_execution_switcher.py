"""Tests for ExecutionSwitcher — Phase 11.5."""

import pytest
from datetime import datetime, timezone
from core.indicators.execution_switcher import ExecutionSwitcher, ExecutionTF


@pytest.fixture
def switcher():
    return ExecutionSwitcher()


# Israel time 16:30 = UTC 14:30 (winter, UTC+2)
OVERLAP_START_UTC = datetime(2026, 1, 21, 14, 30, tzinfo=timezone.utc)
# Israel time 19:00 = UTC 17:00
OVERLAP_END_UTC = datetime(2026, 1, 21, 17, 0, tzinfo=timezone.utc)
# Israel time 12:00 = UTC 10:00 (outside overlap)
OUTSIDE_OVERLAP_UTC = datetime(2026, 1, 21, 10, 0, tzinfo=timezone.utc)


class TestOutsideOverlap:
    def test_returns_5m_outside_overlap(self, switcher):
        d = switcher.decide(OUTSIDE_OVERLAP_UTC, volatility_regime="high")
        assert d.chosen_tf == ExecutionTF.M5
        assert not d.in_overlap

    def test_reason_says_outside(self, switcher):
        d = switcher.decide(OUTSIDE_OVERLAP_UTC)
        assert "outside" in d.reason.lower()


class TestInOverlap:
    def test_high_vol_picks_1m(self, switcher):
        d = switcher.decide(OVERLAP_START_UTC, volatility_regime="high")
        assert d.chosen_tf == ExecutionTF.M1
        assert d.in_overlap

    def test_extreme_vol_picks_1m(self, switcher):
        d = switcher.decide(OVERLAP_START_UTC, volatility_regime="extreme")
        assert d.chosen_tf == ExecutionTF.M1

    def test_normal_vol_picks_5m(self, switcher):
        d = switcher.decide(OVERLAP_START_UTC, volatility_regime="normal")
        assert d.chosen_tf == ExecutionTF.M5
        assert d.in_overlap

    def test_low_vol_picks_5m(self, switcher):
        d = switcher.decide(OVERLAP_START_UTC, volatility_regime="low")
        assert d.chosen_tf == ExecutionTF.M5


class TestConfig:
    def test_disable_m1(self):
        s = ExecutionSwitcher({"allow_m1_in_overlap": False})
        d = s.decide(OVERLAP_START_UTC, volatility_regime="high")
        assert d.chosen_tf == ExecutionTF.M5
        assert "disabled" in d.reason.lower()

    def test_disable_vol_requirement(self):
        s = ExecutionSwitcher({"m1_requires_high_vol": False})
        d = s.decide(OVERLAP_START_UTC, volatility_regime="normal")
        assert d.chosen_tf == ExecutionTF.M1


class TestToDict:
    def test_decision_to_dict(self, switcher):
        d = switcher.decide(OVERLAP_START_UTC, volatility_regime="high").to_dict()
        assert d["chosen_tf"] == "1m"
        assert d["in_overlap"] is True
        assert d["volatility_regime"] == "high"
