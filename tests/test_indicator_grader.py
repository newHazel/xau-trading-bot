"""Tests for the indicator → grader bridge — Phase 11 integration."""

import pytest
from datetime import datetime, timezone
from core.indicators.indicator_grader import build_indicator_results
from core.indicators.vwap import VWAPReading, VWAPBias
from core.indicators.ema import EMAReading
from core.indicators.rsi_divergence import Divergence, DivergenceType
from core.indicators.volume_profile import ProfileReading, PriceLevel

NOW = datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc)


def _vwap(bias):
    return VWAPReading(timestamp=NOW, vwap=2650, session="london", bias=bias, distance_atr=1.0, price=2655)


def _ema(bias):
    return EMAReading(timestamp=NOW, ema_fast=2650, ema_slow=2640, price=2655, bias=bias)


def _div(dtype):
    return Divergence(type=dtype, start_ts=NOW, end_ts=NOW, price_start=2640,
                      price_end=2635, rsi_start=30, rsi_end=40, strength=0.5)


def _vp(level):
    return ProfileReading(timestamp=NOW, poc=2650, current_price=2650, current_level=level)


class TestEmptyReadings:
    def test_all_none_all_false(self):
        r = build_indicator_results("long")
        assert r == {
            "vwap_aligned": False,
            "rsi_divergence_confirms": False,
            "ema_trend_aligned": False,
            "volume_profile_favorable": False,
        }


class TestVWAP:
    def test_long_above_aligned(self):
        r = build_indicator_results("long", vwap=_vwap(VWAPBias.ABOVE))
        assert r["vwap_aligned"]

    def test_long_below_not_aligned(self):
        r = build_indicator_results("long", vwap=_vwap(VWAPBias.BELOW))
        assert not r["vwap_aligned"]

    def test_short_below_aligned(self):
        r = build_indicator_results("short", vwap=_vwap(VWAPBias.BELOW))
        assert r["vwap_aligned"]

    def test_short_above_not_aligned(self):
        r = build_indicator_results("short", vwap=_vwap(VWAPBias.ABOVE))
        assert not r["vwap_aligned"]


class TestEMA:
    def test_long_aligned(self):
        r = build_indicator_results("long", ema=_ema("long"))
        assert r["ema_trend_aligned"]

    def test_long_short_bias_not_aligned(self):
        r = build_indicator_results("long", ema=_ema("short"))
        assert not r["ema_trend_aligned"]

    def test_neutral_not_aligned(self):
        r = build_indicator_results("long", ema=_ema("neutral"))
        assert not r["ema_trend_aligned"]


class TestRSIDivergence:
    def test_long_bullish_confirms(self):
        r = build_indicator_results("long", divergence=_div(DivergenceType.BULLISH_REGULAR))
        assert r["rsi_divergence_confirms"]

    def test_long_bearish_no_confirm(self):
        r = build_indicator_results("long", divergence=_div(DivergenceType.BEARISH_REGULAR))
        assert not r["rsi_divergence_confirms"]

    def test_short_bearish_confirms(self):
        r = build_indicator_results("short", divergence=_div(DivergenceType.BEARISH_REGULAR))
        assert r["rsi_divergence_confirms"]


class TestVolumeProfile:
    def test_poc_favorable(self):
        r = build_indicator_results("long", volume_profile=_vp(PriceLevel.POC))
        assert r["volume_profile_favorable"]

    def test_lvn_favorable(self):
        r = build_indicator_results("long", volume_profile=_vp(PriceLevel.LVN))
        assert r["volume_profile_favorable"]

    def test_hvn_not_favorable(self):
        r = build_indicator_results("long", volume_profile=_vp(PriceLevel.HVN))
        assert not r["volume_profile_favorable"]

    def test_normal_not_favorable(self):
        r = build_indicator_results("long", volume_profile=_vp(PriceLevel.NORMAL))
        assert not r["volume_profile_favorable"]


class TestFullStack:
    def test_all_aligned_long(self):
        r = build_indicator_results(
            "long",
            vwap=_vwap(VWAPBias.ABOVE),
            ema=_ema("long"),
            divergence=_div(DivergenceType.BULLISH_REGULAR),
            volume_profile=_vp(PriceLevel.POC),
        )
        assert all(r.values())

    def test_buy_alias(self):
        r = build_indicator_results("buy", vwap=_vwap(VWAPBias.ABOVE))
        assert r["vwap_aligned"]
