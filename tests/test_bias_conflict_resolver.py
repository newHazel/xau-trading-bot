"""
Tests for bias_conflict_resolver.py — Phase 1.6.

Critical properties:
  - "bullish" only when 4H=bullish AND 1H=bullish.
  - "bearish" only when 4H=bearish AND 1H=bearish.
  - Any neutral or conflicting combination → "neutral".
  - Temporal alignment: 4H bias used at each 1H bar is the MOST RECENT
    4H bar whose timestamp ≤ the 1H bar (strict backward, no look-ahead).
  - If no 4H bar exists yet at a given 1H bar → bias_4h treated as "neutral".
"""

import pytest
import numpy as np
import pandas as pd

from core.structure.swing_detector import SwingDetector
from core.structure.market_structure import MarketStructure
from core.structure.bias_conflict_resolver import HTFConflictResolver, resolve_htf_bias


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _bias_df(biases: list, freq: str, start: str = "2026-01-05 00:00") -> pd.DataFrame:
    """Minimal DataFrame with structure_bias column only."""
    idx = pd.date_range(start, periods=len(biases), freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame({"structure_bias": biases}, index=idx)


def _make_raw_df(highs, lows, freq="1h", start="2026-01-05 00:00"):
    n = len(highs)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    mid = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame(
        {"open": mid, "high": highs, "low": lows, "close": mid, "volume": [100.] * n},
        index=idx,
    )


def _swing_and_structure(highs, lows, freq="1h", tf="1h"):
    df = _make_raw_df(highs, lows, freq=freq)
    with_swings = SwingDetector().detect(df, tf)
    return MarketStructure().classify(with_swings)


def _res():
    return HTFConflictResolver()


# ------------------------------------------------------------------ #
# Core combination logic (pure function)                               #
# ------------------------------------------------------------------ #

class TestCombineLogic:
    def test_both_bullish_returns_bullish(self):
        assert HTFConflictResolver.combine("bullish", "bullish") == "bullish"

    def test_both_bearish_returns_bearish(self):
        assert HTFConflictResolver.combine("bearish", "bearish") == "bearish"

    def test_4h_bullish_1h_bearish_returns_neutral(self):
        assert HTFConflictResolver.combine("bullish", "bearish") == "neutral"

    def test_4h_bearish_1h_bullish_returns_neutral(self):
        assert HTFConflictResolver.combine("bearish", "bullish") == "neutral"

    def test_4h_neutral_returns_neutral(self):
        assert HTFConflictResolver.combine("neutral", "bullish") == "neutral"
        assert HTFConflictResolver.combine("neutral", "bearish") == "neutral"

    def test_1h_neutral_returns_neutral(self):
        assert HTFConflictResolver.combine("bullish", "neutral") == "neutral"
        assert HTFConflictResolver.combine("bearish", "neutral") == "neutral"

    def test_both_neutral_returns_neutral(self):
        assert HTFConflictResolver.combine("neutral", "neutral") == "neutral"


# ------------------------------------------------------------------ #
# DataFrame resolution                                                 #
# ------------------------------------------------------------------ #

class TestResolve:
    def test_aligned_bullish_throughout(self):
        df_4h = _bias_df(["bullish"] * 5, freq="4h")
        df_1h = _bias_df(["bullish"] * 20, freq="1h")
        result = _res().resolve(df_4h, df_1h)
        assert (result["resolved_bias"] == "bullish").all()

    def test_aligned_bearish_throughout(self):
        df_4h = _bias_df(["bearish"] * 5, freq="4h")
        df_1h = _bias_df(["bearish"] * 20, freq="1h")
        result = _res().resolve(df_4h, df_1h)
        assert (result["resolved_bias"] == "bearish").all()

    def test_conflict_produces_neutral(self):
        df_4h = _bias_df(["bullish"] * 5, freq="4h")
        df_1h = _bias_df(["bearish"] * 20, freq="1h")
        result = _res().resolve(df_4h, df_1h)
        assert (result["resolved_bias"] == "neutral").all()

    def test_4h_neutral_produces_neutral(self):
        df_4h = _bias_df(["neutral"] * 5, freq="4h")
        df_1h = _bias_df(["bullish"] * 20, freq="1h")
        result = _res().resolve(df_4h, df_1h)
        assert (result["resolved_bias"] == "neutral").all()

    def test_bias_changes_mid_series(self):
        """4H switches from bearish to bullish mid-way; 1H is bullish throughout.
        Before the switch → neutral (conflict). After → bullish (aligned)."""
        df_4h = _bias_df(["bearish", "bullish"], freq="4h",
                         start="2026-01-05 00:00")
        # 1H bars span across both 4H bars
        df_1h = _bias_df(["bullish"] * 8, freq="1h",
                         start="2026-01-05 00:00")
        result = _res().resolve(df_4h, df_1h)
        # First 4 1H bars (00:00–03:00) use 4H bar at 00:00 (bearish) → neutral
        for i in range(4):
            assert result["resolved_bias"].iloc[i] == "neutral", f"bar {i}"
        # Last 4 1H bars (04:00–07:00) use 4H bar at 04:00 (bullish) → bullish
        for i in range(4, 8):
            assert result["resolved_bias"].iloc[i] == "bullish", f"bar {i}"

    def test_bias_4h_column_present(self):
        df_4h = _bias_df(["bullish"] * 3, freq="4h")
        df_1h = _bias_df(["bullish"] * 12, freq="1h")
        result = _res().resolve(df_4h, df_1h)
        assert "bias_4h" in result.columns

    def test_bias_1h_column_present(self):
        df_4h = _bias_df(["bullish"] * 3, freq="4h")
        df_1h = _bias_df(["bullish"] * 12, freq="1h")
        result = _res().resolve(df_4h, df_1h)
        assert "bias_1h" in result.columns

    def test_resolved_bias_column_present(self):
        df_4h = _bias_df(["bullish"] * 3, freq="4h")
        df_1h = _bias_df(["bullish"] * 12, freq="1h")
        result = _res().resolve(df_4h, df_1h)
        assert "resolved_bias" in result.columns

    def test_resolved_bias_values_valid(self):
        df_4h = _bias_df(["bullish", "bearish", "neutral"], freq="4h")
        df_1h = _bias_df(["bullish"] * 12, freq="1h")
        result = _res().resolve(df_4h, df_1h)
        assert result["resolved_bias"].isin(["bullish", "bearish", "neutral"]).all()


# ------------------------------------------------------------------ #
# Temporal alignment (no look-ahead)                                   #
# ------------------------------------------------------------------ #

class TestTemporalAlignment:
    def test_1h_bars_before_any_4h_bar_are_neutral(self):
        """If 1H bars start BEFORE the first 4H bar, those 1H bars see
        no 4H data → bias_4h = neutral → resolved = neutral."""
        df_4h = _bias_df(["bullish"] * 3, freq="4h",
                         start="2026-01-05 04:00")   # 4H starts at 04:00
        df_1h = _bias_df(["bullish"] * 8, freq="1h",
                         start="2026-01-05 00:00")   # 1H starts at 00:00
        result = _res().resolve(df_4h, df_1h)
        # 1H bars 00:00–03:00 have no 4H bar yet → neutral
        for i in range(4):
            assert result["resolved_bias"].iloc[i] == "neutral", f"bar {i}"
        # 1H bars 04:00–07:00 see 4H bar at 04:00 (bullish) → bullish
        for i in range(4, 8):
            assert result["resolved_bias"].iloc[i] == "bullish", f"bar {i}"

    def test_future_4h_bar_does_not_affect_past_1h_bars(self):
        """A 4H bar that starts AFTER a 1H bar must not influence that bar."""
        df_4h = _bias_df(["neutral", "bullish"], freq="4h",
                         start="2026-01-05 00:00")
        df_1h = _bias_df(["bullish"] * 8, freq="1h",
                         start="2026-01-05 00:00")
        result = _res().resolve(df_4h, df_1h)
        # 1H bars 00:00–03:00 use 4H bar at 00:00 (neutral) → neutral
        for i in range(4):
            assert result["resolved_bias"].iloc[i] == "neutral", f"bar {i}"
        # 1H bars 04:00–07:00 use 4H bar at 04:00 (bullish) → bullish
        for i in range(4, 8):
            assert result["resolved_bias"].iloc[i] == "bullish", f"bar {i}"

    def test_4h_bias_aligned_correctly_at_boundary(self):
        """A 1H bar exactly at the 4H bar boundary should use that 4H bar."""
        df_4h = _bias_df(["bearish", "bullish"], freq="4h",
                         start="2026-01-05 00:00")
        # 1H bar exactly at 04:00 should use the 4H bar at 04:00 (bullish)
        df_1h = _bias_df(["bullish"], freq="1h",
                         start="2026-01-05 04:00")
        result = _res().resolve(df_4h, df_1h)
        assert result["resolved_bias"].iloc[0] == "bullish"


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_indexed_on_1h_timeline(self):
        df_4h = _bias_df(["bullish"] * 5, freq="4h")
        df_1h = _bias_df(["bullish"] * 20, freq="1h")
        result = _res().resolve(df_4h, df_1h)
        pd.testing.assert_index_equal(result.index, df_1h.index)

    def test_output_is_copy_not_inplace(self):
        df_4h = _bias_df(["bullish"] * 5, freq="4h")
        df_1h = _bias_df(["bullish"] * 20, freq="1h")
        _res().resolve(df_4h, df_1h)
        assert "resolved_bias" not in df_1h.columns

    def test_existing_1h_columns_preserved(self):
        df_4h = _bias_df(["bullish"] * 5, freq="4h")
        df_1h = _bias_df(["bullish"] * 20, freq="1h")
        result = _res().resolve(df_4h, df_1h)
        assert "structure_bias" in result.columns

    def test_get_current_bias(self):
        df_4h = _bias_df(["bullish"] * 5, freq="4h")
        df_1h = _bias_df(["bullish"] * 20, freq="1h")
        result = _res().resolve(df_4h, df_1h)
        assert _res().get_current_bias(result) == "bullish"

    def test_get_current_bias_neutral(self):
        df_4h = _bias_df(["bullish"] * 5, freq="4h")
        df_1h = _bias_df(["bearish"] * 20, freq="1h")
        result = _res().resolve(df_4h, df_1h)
        assert _res().get_current_bias(result) == "neutral"


# ------------------------------------------------------------------ #
# Validation / Error handling                                          #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_missing_structure_bias_in_4h_raises(self):
        idx = pd.date_range("2026-01-05", periods=5, freq="4h", tz="UTC")
        df_4h = pd.DataFrame({"other": [1] * 5}, index=idx)
        df_1h = _bias_df(["bullish"] * 20, freq="1h")
        with pytest.raises(ValueError, match="structure_bias"):
            _res().resolve(df_4h, df_1h)

    def test_missing_structure_bias_in_1h_raises(self):
        df_4h = _bias_df(["bullish"] * 5, freq="4h")
        idx = pd.date_range("2026-01-05", periods=20, freq="1h", tz="UTC")
        df_1h = pd.DataFrame({"other": [1] * 20}, index=idx)
        with pytest.raises(ValueError, match="structure_bias"):
            _res().resolve(df_4h, df_1h)

    def test_non_datetime_index_4h_raises(self):
        df_4h = pd.DataFrame({"structure_bias": ["bullish"] * 5}, index=range(5))
        df_1h = _bias_df(["bullish"] * 20, freq="1h")
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _res().resolve(df_4h, df_1h)

    def test_non_datetime_index_1h_raises(self):
        df_4h = _bias_df(["bullish"] * 5, freq="4h")
        df_1h = pd.DataFrame({"structure_bias": ["bullish"] * 20}, index=range(20))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _res().resolve(df_4h, df_1h)

    def test_convenience_function_works(self):
        df_4h = _bias_df(["bullish"] * 5, freq="4h")
        df_1h = _bias_df(["bullish"] * 20, freq="1h")
        result = resolve_htf_bias(df_4h, df_1h)
        assert "resolved_bias" in result.columns


# ------------------------------------------------------------------ #
# Integration with SwingDetector + MarketStructure                     #
# ------------------------------------------------------------------ #

class TestIntegration:
    def _make_bullish_structure(self, freq, tf):
        """
        Build a bullish market: two ascending peaks + two ascending troughs.
        Works for both 1H and 4H (same OHLCV pattern, different freq/tf).
        """
        # HH at bar 12, HL at bar 16 confirmed (window=3, lag=1)
        highs = [10,11,12,13,20,13,12,11,10,11,12,13,25,13,12,11,10,11,12,13,
                 10,11,12,13, 5, 9,10,11,12,13]
        lows  = [ 8, 9,10,11,18,11,10, 9, 5, 9,10,11,23,11,10, 9, 8, 9,10,11,
                   8, 9,10,11, 3, 7, 8, 9,10,11]
        return _swing_and_structure(highs[:20], lows[:20], freq=freq, tf=tf)

    def test_bullish_alignment_in_full_pipeline(self):
        """4H pre-set to bullish; 1H built from real swing structure.
        When 1H bias reaches 'bullish' → resolved must be 'bullish'."""
        # df_4h: always bullish (covers the full 1H time window)
        df_4h = _bias_df(["bullish"] * 10, freq="4h", start="2026-01-05 00:00")

        # df_1h: real swing detection that turns bullish by bar 17
        df_1h = self._make_bullish_structure("1h", "1h")
        df_1h.index = pd.date_range(
            "2026-01-05 00:00", periods=len(df_1h), freq="1h", tz="UTC", name="timestamp"
        )

        result = _res().resolve(df_4h, df_1h)
        bullish_1h_mask = result["bias_1h"] == "bullish"
        assert bullish_1h_mask.any(), "Expected 1H to reach bullish bias"
        assert (result.loc[bullish_1h_mask, "resolved_bias"] == "bullish").all()

    def test_conflict_resolution_in_full_pipeline(self):
        """4H bullish, 1H bearish → resolved = neutral for those bars."""
        df_4h = self._make_bullish_structure("4h", "4h")
        # 1H shows bearish structure
        df_1h_raw = _make_raw_df(
            highs=[10,11,12,13,25,13,12,11,10,11,12,13,20,13,12,11,10,11,12,13],
            lows =[ 8, 9,10,11,23,11,10, 9, 8, 9,10,11,18,11,10, 9, 5, 9,10,11],
            freq="1h",
        )
        df_1h_sw = SwingDetector().detect(df_1h_raw, "1h")
        df_1h    = MarketStructure().classify(df_1h_sw)

        df_4h.index = pd.date_range(
            "2026-01-05 00:00", periods=len(df_4h), freq="4h", tz="UTC", name="timestamp"
        )
        df_1h.index = pd.date_range(
            "2026-01-05 00:00", periods=len(df_1h), freq="1h", tz="UTC", name="timestamp"
        )

        result = _res().resolve(df_4h, df_1h)
        # Where 4H=bullish and 1H=bearish → must be neutral
        conflict_mask = (result["bias_4h"] == "bullish") & (result["bias_1h"] == "bearish")
        if conflict_mask.any():
            assert (result.loc[conflict_mask, "resolved_bias"] == "neutral").all()

    def test_resolved_bias_column_has_only_valid_values(self):
        df_4h = self._make_bullish_structure("4h", "4h")
        df_1h = self._make_bullish_structure("1h", "1h")
        df_4h.index = pd.date_range(
            "2026-01-05", periods=len(df_4h), freq="4h", tz="UTC", name="timestamp"
        )
        df_1h.index = pd.date_range(
            "2026-01-05", periods=len(df_1h), freq="1h", tz="UTC", name="timestamp"
        )
        result = _res().resolve(df_4h, df_1h)
        assert result["resolved_bias"].isin(["bullish", "bearish", "neutral"]).all()
