"""Tests for Correlation Spike Filter — Phase 3.5."""

import pytest
import numpy as np
from core.filters.correlation_spike_filter import (
    CorrelationSpikeFilter,
    CorrelationState,
)

CONFIG = {
    "correlation_spike": {
        "enabled": True,
        "pair": "DXY",
        "expected_correlation": -0.7,
        "lookback_minutes": 30,
        "break_threshold": 0.3,
        "action_on_break": "degraded_mode",
    },
}


@pytest.fixture
def csf():
    return CorrelationSpikeFilter(CONFIG)


def _negatively_correlated(n=30):
    """XAU up while DXY down — correlation near -0.7."""
    np.random.seed(42)
    shared = np.random.randn(n) * 0.5
    np.random.seed(7)
    xau_own = np.random.randn(n) * 0.3
    np.random.seed(13)
    dxy_own = np.random.randn(n) * 0.3
    xau = np.cumsum(shared + xau_own) + 2300
    dxy = np.cumsum(-shared * 0.6 + dxy_own) + 104
    return xau.tolist(), dxy.tolist()


def _positively_correlated(n=30):
    """Both XAU and DXY rise together — abnormal panic/decoupling."""
    np.random.seed(42)
    noise = np.random.randn(n) * 0.5 + 0.2
    xau = np.cumsum(noise) + 2300
    dxy = np.cumsum(noise * 0.6) + 104
    return xau.tolist(), dxy.tolist()


class TestNormalCorrelation:
    def test_negative_correlation_is_normal(self, csf):
        xau, dxy = _negatively_correlated()
        result = csf.check(xau, dxy)
        assert result.state == CorrelationState.NORMAL
        assert result.current_correlation is not None
        assert result.current_correlation < 0

    def test_is_spike_false(self, csf):
        xau, dxy = _negatively_correlated()
        assert csf.is_spike(xau, dxy) is False


class TestSpikeDetection:
    def test_positive_correlation_is_spike(self, csf):
        xau, dxy = _positively_correlated()
        result = csf.check(xau, dxy)
        assert result.state == CorrelationState.SPIKE
        assert result.deviation >= 0.3

    def test_is_spike_true(self, csf):
        xau, dxy = _positively_correlated()
        assert csf.is_spike(xau, dxy) is True


class TestNoData:
    def test_none_xau(self, csf):
        result = csf.check(None, [104.0] * 10)
        assert result.state == CorrelationState.NO_DATA

    def test_none_dxy(self, csf):
        result = csf.check([2300.0] * 10, None)
        assert result.state == CorrelationState.NO_DATA

    def test_too_short(self, csf):
        result = csf.check([2300, 2301], [104, 103])
        assert result.state == CorrelationState.NO_DATA

    def test_zero_variance(self, csf):
        result = csf.check([2300.0] * 10, [104.0] * 10)
        assert result.state == CorrelationState.NO_DATA


class TestDisabled:
    def test_disabled_returns_normal(self):
        config = {"correlation_spike": {"enabled": False}}
        csf = CorrelationSpikeFilter(config)
        xau, dxy = _positively_correlated()
        result = csf.check(xau, dxy)
        assert result.state == CorrelationState.NORMAL


class TestLookback:
    def test_uses_lookback_window(self):
        config = {
            "correlation_spike": {
                "enabled": True,
                "expected_correlation": -0.7,
                "lookback_minutes": 10,
                "break_threshold": 0.3,
            },
        }
        csf = CorrelationSpikeFilter(config)
        # First 20 points negatively correlated, last 10 positively
        xau_neg, dxy_neg = _negatively_correlated(20)
        xau_pos, dxy_pos = _positively_correlated(10)
        xau = xau_neg + xau_pos
        dxy = dxy_neg + dxy_pos
        result = csf.check(xau, dxy)
        # Only last 10 used → should detect spike
        assert result.state == CorrelationState.SPIKE


class TestResultDict:
    def test_spike_to_dict(self, csf):
        xau, dxy = _positively_correlated()
        d = csf.check(xau, dxy).to_dict()
        assert d["state"] == "spike"
        assert isinstance(d["current_correlation"], float)
        assert isinstance(d["deviation"], float)

    def test_normal_to_dict(self, csf):
        xau, dxy = _negatively_correlated()
        d = csf.check(xau, dxy).to_dict()
        assert d["state"] == "normal"


class TestActionOnBreak:
    def test_default_action(self, csf):
        assert csf.action_on_break == "degraded_mode"
