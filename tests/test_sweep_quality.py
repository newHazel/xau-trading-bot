"""Sweep quality levers (Pine-v3.2 parity, gold focus):
  1. min-penetration gate — a 1-tick poke beyond a level must NOT register as a
     sweep when sweep_min_penetration_enabled is on (mult=0 stays byte-identical).
  2. get_last_sweep carries wick_extreme (the sweep leg's true tip).
  3. sweep_extreme_broken — the continuous mid-sequence kill predicate.
  4. sweep_src telemetry rides PipelineSignal end-to-end.
"""

import numpy as np
import pandas as pd
import pytest

from core.smc.sweep_detector import SweepDetector
from core.engine.sequence_runner import sweep_extreme_broken
from core.engine.signal_pipeline import PipelineSignal
from core.alerts.telegram_sender import TelegramSender

SWEEP_COLS = ["sweep_bull_level", "sweep_bull_type", "sweep_bull_wick_bar",
              "sweep_bear_level", "sweep_bear_type", "sweep_bear_wick_bar"]


def _df(rows):
    """rows: list of (high, low, close, swing_low). Non-sweep cols filled NaN.
    Bars have ~1.0 range so rolling ATR ~= 1.0 → penetration thresholds are legible."""
    n = len(rows)
    idx = pd.date_range("2026-01-05", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "high":  [r[0] for r in rows],
        "low":   [r[1] for r in rows],
        "close": [r[2] for r in rows],
        "swing_high": np.nan,
        "swing_low":  [r[3] for r in rows],
        "eqh_level": np.nan, "eql_level": np.nan,
        "pdh": np.nan, "pdl": np.nan,
    }, index=idx)


def _bull_sweep_frame(pen: float):
    """Support at 100 (swing_low confirmed on bar 2); bar 5 wicks `pen` below it
    and closes back above → a bullish sweep candidate on bar 5."""
    base = [
        (101.0, 100.0, 100.6, np.nan),   # 0
        (101.2, 100.2, 100.8, np.nan),   # 1
        (101.0, 100.0, 100.5, 100.0),    # 2: swing_low=100 confirmed here
        (101.0, 100.1, 100.6, np.nan),   # 3
        (101.1, 100.2, 100.7, np.nan),   # 4
        (100.9, 100.0 - pen, 100.4, np.nan),  # 5: wick below the level, close back above
        (101.0, 100.3, 100.8, np.nan),   # 6
    ]
    return _df(base)


class TestMinPenetration:
    def test_shallow_poke_filtered_when_enabled(self):
        df = _bull_sweep_frame(pen=0.05)          # ~0.05 pen vs ATR≈1.0
        default = SweepDetector().detect(df)
        gated = SweepDetector(min_penetration_atr_mult=0.5).detect(df)
        assert default["sweep_bull_level"].notna().any()      # baseline sees the sweep
        assert not gated["sweep_bull_level"].notna().any()    # gated: poke too shallow

    def test_deep_wick_passes_the_gate(self):
        df = _bull_sweep_frame(pen=0.9)
        gated = SweepDetector(min_penetration_atr_mult=0.5).detect(df)
        assert gated["sweep_bull_level"].notna().any()

    def test_mult_zero_is_byte_identical(self):
        df = _bull_sweep_frame(pen=0.05)
        a = SweepDetector().detect(df)[SWEEP_COLS]
        b = SweepDetector(min_penetration_atr_mult=0.0).detect(df)[SWEEP_COLS]
        pd.testing.assert_frame_equal(a, b)

    def test_negative_mult_rejected(self):
        with pytest.raises(ValueError):
            SweepDetector(min_penetration_atr_mult=-0.1)

    def test_bear_side_symmetric(self):
        n = 7
        idx = pd.date_range("2026-01-05", periods=n, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "high":  [101.0, 101.2, 101.0, 101.0, 101.1, 102.05, 101.0],
            "low":   [100.0, 100.2, 100.0, 100.1, 100.2, 100.9, 100.3],
            "close": [100.6, 100.8, 100.5, 100.6, 100.7, 101.4, 100.8],
            "swing_high": [np.nan, np.nan, 102.0, np.nan, np.nan, np.nan, np.nan],
            "swing_low": np.nan, "eqh_level": np.nan, "eql_level": np.nan,
            "pdh": np.nan, "pdl": np.nan,
        }, index=idx)
        # bar 5 pokes 0.05 above swing_high 102 and closes back below
        assert SweepDetector().detect(df)["sweep_bear_level"].notna().any()
        gated = SweepDetector(min_penetration_atr_mult=0.5).detect(df)
        assert not gated["sweep_bear_level"].notna().any()


class TestWickExtreme:
    def test_bull_wick_extreme_is_the_wick_low(self):
        df = _bull_sweep_frame(pen=0.9)
        det = SweepDetector()
        out = det.detect(df)
        sw = det.get_last_sweep(out, direction="bull")
        assert sw is not None
        assert sw["type"] == "swing_low"
        assert sw["wick_extreme"] == pytest.approx(100.0 - 0.9)


class TestSweepExtremeBroken:
    SWEEP = {"wick_extreme": 100.0, "type": "swing_low", "direction": "bull"}

    def test_long_dies_on_close_below_extreme(self):
        assert sweep_extreme_broken("long", 99.9, self.SWEEP) is True

    def test_long_survives_above_extreme(self):
        assert sweep_extreme_broken("long", 100.1, self.SWEEP) is False

    def test_short_dies_on_close_above_extreme(self):
        s = {"wick_extreme": 105.0}
        assert sweep_extreme_broken("short", 105.2, s) is True
        assert sweep_extreme_broken("short", 104.8, s) is False

    def test_missing_extreme_or_sweep_never_kills(self):
        assert sweep_extreme_broken("long", 1.0, None) is False
        assert sweep_extreme_broken("long", 1.0, {"type": "pdl"}) is False
        assert sweep_extreme_broken(None, 1.0, self.SWEEP) is False


class TestPDHCompleteDayOnly:
    """PDH/PDL from COMPLETE previous days only — a rolling window that starts
    mid-day must NOT emit a partial-day extreme as the 'previous day high/low'."""

    def _mk(self, start, hours):
        n = hours * 4                      # 15m bars
        idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
        rng = np.random.default_rng(7)
        base = 100 + rng.normal(0, 0.2, n).cumsum()
        return pd.DataFrame({
            "high": base + 0.5, "low": base - 0.5, "close": base,
            "open": base, "volume": 1.0,
            "swing_high": np.nan, "swing_low": np.nan,
        }, index=idx)

    def test_partial_first_day_yields_nan(self):
        from core.smc.liquidity_detector import LiquidityDetector
        # window starts 18:00 D0 → D0 is partial → D1 bars must have NaN pdh/pdl
        df = self._mk("2026-01-05 18:00", hours=18)          # D0 18:00 → D1 12:00
        out = LiquidityDetector().detect(df)
        d1 = out[out.index.normalize() == pd.Timestamp("2026-01-06", tz="UTC")]
        assert d1["pdh"].isna().all() and d1["pdl"].isna().all()

    def test_complete_prev_day_yields_levels(self):
        from core.smc.liquidity_detector import LiquidityDetector
        # window starts 18:00 D0 and spans through D2 → D1 is COMPLETE → D2 has levels
        df = self._mk("2026-01-05 18:00", hours=42)          # → D2 12:00
        out = LiquidityDetector().detect(df)
        d1 = df[df.index.normalize() == pd.Timestamp("2026-01-06", tz="UTC")]
        d2 = out[out.index.normalize() == pd.Timestamp("2026-01-07", tz="UTC")]
        assert d2["pdh"].notna().all()
        assert d2["pdh"].iloc[0] == pytest.approx(float(d1["high"].max()))
        assert d2["pdl"].iloc[0] == pytest.approx(float(d1["low"].min()))

    def test_midnight_start_day_counts_as_complete(self):
        from core.smc.liquidity_detector import LiquidityDetector
        df = self._mk("2026-01-05 00:00", hours=30)          # D0 whole + D1 morning
        out = LiquidityDetector().detect(df)
        d1 = out[out.index.normalize() == pd.Timestamp("2026-01-06", tz="UTC")]
        d0 = df[df.index.normalize() == pd.Timestamp("2026-01-05", tz="UTC")]
        assert d1["pdh"].notna().all()
        assert d1["pdh"].iloc[0] == pytest.approx(float(d0["high"].max()))


class TestSweepSrcTelemetry:
    def _sig(self, **kw):
        from datetime import datetime, timezone
        base = dict(setup_id="S1", direction="long", entry=100.0, sl=99.0,
                    tp1=102.0, tp2=103.5, lot_size=0.1, grade="A", score=30,
                    timestamp=datetime(2026, 1, 5, tzinfo=timezone.utc),
                    bar_index=10, approved=True)
        base.update(kw)
        return PipelineSignal(**base)

    def test_default_none_and_dict_passthrough(self):
        s = self._sig()
        assert s.sweep_src is None
        assert self._sig(sweep_src="pdl").to_signal_dict()["sweep_src"] == "pdl"

    def test_telegram_line_only_when_present(self):
        alert = {"grade": "A", "direction": "long", "setup_id": "S1", "entry": 100.0,
                 "sl": 99.0, "tp1": 102.0, "tp2": 103.5, "rr": 2.0, "symbol": "XAUUSD"}
        assert "Sweep:" not in TelegramSender.format_signal(alert)
        alert["sweep_src"] = "pdl"
        assert "Sweep: PDL" in TelegramSender.format_signal(alert)
