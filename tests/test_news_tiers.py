"""Tests for NewsTiers classification — Phase 3.3."""

import pytest
from core.filters.news_tiers import NewsTiers

CONFIG = {
    "tiers": {
        "tier_1": {
            "events": ["FOMC", "Interest Rate Decision", "NFP", "Powell Speech"],
            "block_before_minutes": 60,
            "block_after_minutes": 60,
        },
        "tier_2": {
            "events": ["CPI", "PCE", "Core CPI", "Core PCE"],
            "block_before_minutes": 45,
            "block_after_minutes": 45,
        },
        "tier_3": {
            "events": ["Unemployment Rate", "Retail Sales", "GDP", "ISM Manufacturing"],
            "block_before_minutes": 30,
            "block_after_minutes": 30,
        },
        "tier_4": {
            "events": ["PPI", "Building Permits", "Consumer Confidence"],
            "block_before_minutes": 0,
            "block_after_minutes": 0,
            "degrade_grade": True,
        },
    },
}


@pytest.fixture
def tiers():
    return NewsTiers(CONFIG)


class TestTierClassification:
    @pytest.mark.parametrize("title,expected_tier", [
        ("FOMC Meeting Minutes", 1),
        ("Interest Rate Decision", 1),
        ("NFP", 1),
        ("Powell Speech at Jackson Hole", 1),
        ("CPI m/m", 2),
        ("Core PCE Price Index", 2),
        ("Unemployment Rate", 3),
        ("Retail Sales m/m", 3),
        ("ISM Manufacturing PMI", 3),
        ("PPI m/m", 4),
        ("Building Permits", 4),
        ("Consumer Confidence", 4),
    ])
    def test_classify_event(self, tiers, title, expected_tier):
        tc = tiers.classify(title)
        assert tc is not None
        assert tc.tier == expected_tier

    def test_unknown_event(self, tiers):
        assert tiers.classify("Random Press Conference") is None


class TestTierProperties:
    def test_tier_1_blocking(self, tiers):
        tc = tiers.get_tier(1)
        assert tc.block_before_minutes == 60
        assert tc.block_after_minutes == 60
        assert tc.degrade_grade is False

    def test_tier_4_degrade(self, tiers):
        tc = tiers.get_tier(4)
        assert tc.block_before_minutes == 0
        assert tc.degrade_grade is True

    def test_nonexistent_tier(self, tiers):
        assert tiers.get_tier(99) is None
