"""
Tests for market_structure.py — Phase 1.2.

Critical properties:
  - First swing of each side has no label (no previous to compare).
  - Labels reflect only information confirmed up to that bar (no look-ahead).
  - Bias = bullish iff last_high==HH and last_low==HL.
  - Bias = bearish iff last_high==LH and last_low==LL.
  - Bias is carried forward on every bar (not just swing bars).
  - Equal highs/lows produce EH/EL labels and neutral bias.
"""

import pytest
import numpy as np
import pandas as pd

from core.structure.swing_detector import SwingDetector
from core.structure.market_structure import MarketStructure, classify_structure


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_raw_df(highs, lows, start="2026-01-05 10:00", freq="5min"):
    n = len(highs)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    mid = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame(
        {"open": mid, "high": highs, "low": lows, "close": mid, "volume": [100.0] * n},
        index=idx,
    )


def _make_swing_df(n=20, start="2026-01-05 10:00", freq="5min"):
    """Minimal DataFrame with swing columns (all NaN / −1) for unit injection."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "high":            [10.0] * n,
            "low":             [8.0]  * n,
            "open":            [9.0]  * n,
            "close":           [9.0]  * n,
            "volume":          [100.0] * n,
            "swing_high":      [np.nan] * n,
            "swing_low":       [np.nan] * n,
            "swing_high_idx":  [-1] * n,
            "swing_low_idx":   [-1] * n,
        },
        index=idx,
    )


def _set_sh(df, pos, price):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("swing_high")]     = float(price)
    df.iloc[pos, df.columns.get_loc("swing_high_idx")] = max(0, pos - 2)
    return df


def _set_sl(df, pos, price):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("swing_low")]     = float(price)
    df.iloc[pos, df.columns.get_loc("swing_low_idx")] = max(0, pos - 2)
    return df


def _det_classify(highs, lows, tf="5m", freq="5min"):
    """Run SwingDetector then MarketStructure on synthetic OHLCV data."""
    df = _make_raw_df(highs, lows, freq=freq)
    with_swings = SwingDetector().detect(df, tf)
    return MarketStructure().classify(with_swings)


def _ms():
    return MarketStructure()


# ------------------------------------------------------------------ #
# Basic label classification                                           #
# ------------------------------------------------------------------ #

class TestBasicClassification:
    def test_ascending_highs_labeled_HH(self):
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)
        df = _set_sh(df, 8, 25.0)   # 25 > 20 → HH
        result = _ms().classify(df)
        assert result["swing_label_high"].iloc[8] == "HH"

    def test_descending_highs_labeled_LH(self):
        df = _make_swing_df()
        df = _set_sh(df, 3, 25.0)
        df = _set_sh(df, 8, 20.0)   # 20 < 25 → LH
        result = _ms().classify(df)
        assert result["swing_label_high"].iloc[8] == "LH"

    def test_ascending_lows_labeled_HL(self):
        df = _make_swing_df()
        df = _set_sl(df, 3, 5.0)
        df = _set_sl(df, 8, 8.0)    # 8 > 5 → HL
        result = _ms().classify(df)
        assert result["swing_label_low"].iloc[8] == "HL"

    def test_descending_lows_labeled_LL(self):
        df = _make_swing_df()
        df = _set_sl(df, 3, 8.0)
        df = _set_sl(df, 8, 5.0)    # 5 < 8 → LL
        result = _ms().classify(df)
        assert result["swing_label_low"].iloc[8] == "LL"

    def test_first_swing_high_has_no_label(self):
        df = _make_swing_df()
        df = _set_sh(df, 5, 20.0)
        df = _set_sh(df, 12, 25.0)
        result = _ms().classify(df)
        # First confirmed high at bar 5 → no previous → None
        assert result["swing_label_high"].iloc[5] is None

    def test_first_swing_low_has_no_label(self):
        df = _make_swing_df()
        df = _set_sl(df, 5, 8.0)
        df = _set_sl(df, 12, 5.0)
        result = _ms().classify(df)
        assert result["swing_label_low"].iloc[5] is None

    def test_equal_highs_labeled_EH(self):
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)
        df = _set_sh(df, 8, 20.0)   # equal → EH
        result = _ms().classify(df)
        assert result["swing_label_high"].iloc[8] == "EH"

    def test_equal_lows_labeled_EL(self):
        df = _make_swing_df()
        df = _set_sl(df, 3, 5.0)
        df = _set_sl(df, 8, 5.0)    # equal → EL
        result = _ms().classify(df)
        assert result["swing_label_low"].iloc[8] == "EL"

    def test_three_ascending_highs_all_HH(self):
        df = _make_swing_df()
        df = _set_sh(df, 2, 10.0)
        df = _set_sh(df, 6, 15.0)
        df = _set_sh(df, 10, 20.0)
        result = _ms().classify(df)
        assert result["swing_label_high"].iloc[2]  is None   # first
        assert result["swing_label_high"].iloc[6]  == "HH"
        assert result["swing_label_high"].iloc[10] == "HH"

    def test_three_descending_lows_all_LL(self):
        df = _make_swing_df()
        df = _set_sl(df, 2, 15.0)
        df = _set_sl(df, 6, 10.0)
        df = _set_sl(df, 10, 5.0)
        result = _ms().classify(df)
        assert result["swing_label_low"].iloc[2]  is None
        assert result["swing_label_low"].iloc[6]  == "LL"
        assert result["swing_label_low"].iloc[10] == "LL"


# ------------------------------------------------------------------ #
# Bias classification                                                  #
# ------------------------------------------------------------------ #

class TestBiasClassification:
    def test_bullish_bias_requires_HH_and_HL(self):
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)
        df = _set_sl(df, 4, 5.0)
        df = _set_sh(df, 8, 25.0)   # HH
        df = _set_sl(df, 12, 8.0)   # HL
        result = _ms().classify(df)
        assert result["structure_bias"].iloc[12] == "bullish"

    def test_bearish_bias_requires_LH_and_LL(self):
        df = _make_swing_df()
        df = _set_sh(df, 2, 25.0)
        df = _set_sl(df, 4, 8.0)
        df = _set_sh(df, 8, 20.0)   # LH
        df = _set_sl(df, 12, 5.0)   # LL
        result = _ms().classify(df)
        assert result["structure_bias"].iloc[12] == "bearish"

    def test_neutral_bias_when_mixed(self):
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)
        df = _set_sl(df, 4, 5.0)
        df = _set_sh(df, 8, 25.0)   # HH
        df = _set_sl(df, 12, 3.0)   # LL  ← mixed
        result = _ms().classify(df)
        assert result["structure_bias"].iloc[12] == "neutral"

    def test_neutral_bias_when_no_swings(self):
        df = _make_swing_df()
        result = _ms().classify(df)
        assert (result["structure_bias"] == "neutral").all()

    def test_neutral_bias_when_only_high_labeled(self):
        # HH confirmed but no low label yet → neutral
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)
        df = _set_sh(df, 8, 25.0)
        result = _ms().classify(df)
        assert result["structure_bias"].iloc[8] == "neutral"

    def test_neutral_bias_when_only_low_labeled(self):
        df = _make_swing_df()
        df = _set_sl(df, 2, 8.0)
        df = _set_sl(df, 8, 5.0)
        result = _ms().classify(df)
        assert result["structure_bias"].iloc[8] == "neutral"

    def test_equal_labels_produce_neutral_bias(self):
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)
        df = _set_sl(df, 4, 5.0)
        df = _set_sh(df, 8, 20.0)   # EH
        df = _set_sl(df, 12, 5.0)   # EL
        result = _ms().classify(df)
        assert result["structure_bias"].iloc[12] == "neutral"

    def test_bias_carried_forward_between_swings(self):
        # Bullish established at bar 12; bars 13-17 must remain bullish.
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)
        df = _set_sl(df, 4, 5.0)
        df = _set_sh(df, 8, 25.0)   # HH
        df = _set_sl(df, 12, 8.0)   # HL → bullish
        result = _ms().classify(df)
        for i in range(13, 18):
            assert result["structure_bias"].iloc[i] == "bullish", f"bar {i}"

    def test_bias_updates_when_new_swing_arrives(self):
        # Starts bullish after HH+HL, becomes neutral after LL flips it.
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)
        df = _set_sl(df, 4, 5.0)
        df = _set_sh(df, 8, 25.0)   # HH → neutral (no low label yet)
        df = _set_sl(df, 12, 8.0)   # HL → bullish
        df = _set_sl(df, 16, 3.0)   # LL → mixed → neutral
        result = _ms().classify(df)
        assert result["structure_bias"].iloc[12] == "bullish"
        assert result["structure_bias"].iloc[16] == "neutral"


# ------------------------------------------------------------------ #
# No look-ahead                                                        #
# ------------------------------------------------------------------ #

class TestNoLookahead:
    def test_bars_before_first_swing_have_no_labels(self):
        df = _make_swing_df()
        df = _set_sh(df, 10, 20.0)
        result = _ms().classify(df)
        for i in range(10):
            assert result["swing_label_high"].iloc[i] is None
            assert result["swing_label_low"].iloc[i]  is None

    def test_bars_before_first_swing_are_neutral(self):
        df = _make_swing_df()
        df = _set_sh(df, 10, 20.0)
        result = _ms().classify(df)
        for i in range(10):
            assert result["structure_bias"].iloc[i] == "neutral"

    def test_future_spike_does_not_relabel_past_bars(self):
        # Inject a massive spike at bar 18. Bars 0-12 must not be affected.
        df = _make_swing_df()
        df = _set_sh(df, 5, 20.0)
        df = _set_sh(df, 18, 99_000.0)
        result = _ms().classify(df)
        # All bars before the spike confirm should not show HH
        for i in range(14):
            label = result["swing_label_high"].iloc[i]
            assert label is None or label != "HH", f"bar {i} incorrectly labelled HH"

    def test_label_placed_exactly_at_confirmation_bar(self):
        # Swing is injected at bar 8 — confirm bar IS bar 8 (SwingDetector places it).
        df = _make_swing_df()
        df = _set_sh(df, 3, 20.0)
        df = _set_sh(df, 8, 25.0)
        result = _ms().classify(df)
        assert result["swing_label_high"].iloc[8] == "HH"
        # Bars 4-7 must not have the HH label
        for i in range(4, 8):
            assert result["swing_label_high"].iloc[i] is None


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _make_swing_df()
        result = _ms().classify(df)
        for col in ["swing_label_high", "swing_label_low", "structure_bias"]:
            assert col in result.columns

    def test_existing_columns_preserved(self):
        df = _make_swing_df()
        result = _ms().classify(df)
        for col in ["swing_high", "swing_low", "swing_high_idx", "swing_low_idx",
                    "high", "low"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _make_swing_df()
        result = _ms().classify(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _make_swing_df()
        _ms().classify(df)
        assert "swing_label_high" not in df.columns

    def test_structure_bias_on_every_row(self):
        df = _make_swing_df()
        result = _ms().classify(df)
        assert result["structure_bias"].notna().all()
        assert result["structure_bias"].isin(["bullish", "bearish", "neutral"]).all()

    def test_no_label_rows_have_none_not_nan(self):
        df = _make_swing_df()
        result = _ms().classify(df)
        # No swings injected → all label cells must be None
        assert result["swing_label_high"].iloc[0] is None
        assert result["swing_label_low"].iloc[0]  is None


# ------------------------------------------------------------------ #
# Accessor methods                                                      #
# ------------------------------------------------------------------ #

class TestAccessors:
    def _build_bullish(self):
        df = _make_swing_df()
        df = _set_sh(df, 2, 20.0)
        df = _set_sl(df, 4, 5.0)
        df = _set_sh(df, 8, 25.0)
        df = _set_sl(df, 12, 8.0)
        return _ms().classify(df)

    def test_get_current_bias_bullish(self):
        result = self._build_bullish()
        assert _ms().get_current_bias(result) == "bullish"

    def test_get_current_bias_bearish(self):
        df = _make_swing_df()
        df = _set_sh(df, 2, 25.0)
        df = _set_sl(df, 4, 8.0)
        df = _set_sh(df, 8, 20.0)
        df = _set_sl(df, 12, 5.0)
        result = _ms().classify(df)
        assert _ms().get_current_bias(result) == "bearish"

    def test_get_current_bias_neutral_when_no_swings(self):
        df = _make_swing_df()
        result = _ms().classify(df)
        assert _ms().get_current_bias(result) == "neutral"

    def test_get_structure_sequence_returns_list(self):
        result = self._build_bullish()
        seq = _ms().get_structure_sequence(result)
        assert isinstance(seq, list)

    def test_get_structure_sequence_newest_first(self):
        result = self._build_bullish()
        seq = _ms().get_structure_sequence(result, n=10)
        if len(seq) > 1:
            assert seq[0]["confirm_ts"] >= seq[1]["confirm_ts"]

    def test_get_structure_sequence_respects_n(self):
        result = self._build_bullish()
        seq = _ms().get_structure_sequence(result, n=1)
        assert len(seq) <= 1

    def test_get_structure_sequence_has_correct_keys(self):
        result = self._build_bullish()
        seq = _ms().get_structure_sequence(result, n=10)
        for entry in seq:
            for key in ("confirm_ts", "label", "price", "bar_idx", "side"):
                assert key in entry

    def test_get_structure_sequence_labels_are_valid(self):
        result = self._build_bullish()
        seq = _ms().get_structure_sequence(result, n=10)
        valid = {"HH", "LH", "EH", "HL", "LL", "EL"}
        for entry in seq:
            assert entry["label"] in valid

    def test_get_structure_sequence_sides_correct(self):
        result = self._build_bullish()
        seq = _ms().get_structure_sequence(result, n=10)
        for entry in seq:
            assert entry["side"] in ("high", "low")
            if entry["side"] == "high":
                assert entry["label"] in ("HH", "LH", "EH")
            else:
                assert entry["label"] in ("HL", "LL", "EL")

    def test_get_structure_sequence_empty_when_no_labeled_swings(self):
        # Only one swing → no label
        df = _make_swing_df()
        df = _set_sh(df, 5, 20.0)
        result = _ms().classify(df)
        seq = _ms().get_structure_sequence(result)
        assert seq == []


# ------------------------------------------------------------------ #
# Validation / Error handling                                          #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_missing_swing_columns_raises(self):
        idx = pd.date_range("2026-01-05", periods=5, freq="5min", tz="UTC")
        df = pd.DataFrame({"high": [10] * 5, "low": [8] * 5}, index=idx)
        with pytest.raises(ValueError, match="Missing swing columns"):
            _ms().classify(df)

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame(
            {
                "swing_high": [np.nan] * 5,
                "swing_low":  [np.nan] * 5,
                "swing_high_idx": [-1] * 5,
                "swing_low_idx":  [-1] * 5,
            },
            index=range(5),
        )
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _ms().classify(df)

    def test_convenience_function_works(self):
        df = _make_swing_df()
        result = classify_structure(df)
        assert "structure_bias" in result.columns


# ------------------------------------------------------------------ #
# Integration with SwingDetector                                       #
# ------------------------------------------------------------------ #

class TestIntegration:
    def test_bullish_market_detected(self):
        """
        Two ascending peaks and two ascending troughs → bullish.

        window=5, lag=2:
          Peak 1  at bar  4 (high=20) confirmed at bar  6
          Trough 1 at bar  8 (low=5)  confirmed at bar 10
          Peak 2  at bar 12 (high=25) confirmed at bar 14
          Trough 2 at bar 16 (low=8)  confirmed at bar 18

        After bar 18: HH (25>20) + HL (8>5) → bullish.
        """
        highs = [10, 11, 12, 13, 20, 13, 12, 11, 10, 11, 12, 13, 25, 13, 12, 11, 10, 11, 12, 13]
        lows  = [ 8,  9, 10, 11, 18, 11, 10,  9,  5,  9, 10, 11, 23, 11, 10,  9,  8,  9, 10, 11]
        result = _det_classify(highs, lows)
        assert result["structure_bias"].iloc[-1] == "bullish"

    def test_bearish_market_detected(self):
        """
        Descending peaks and descending troughs → bearish.

        Peak 1  at bar  4 (high=25) confirmed at bar  6
        Trough 1 at bar  8 (low=8)  confirmed at bar 10
        Peak 2  at bar 12 (high=20) confirmed at bar 14  → LH
        Trough 2 at bar 16 (low=5)  confirmed at bar 18  → LL  → bearish
        """
        highs = [10, 11, 12, 13, 25, 13, 12, 11, 10, 11, 12, 13, 20, 13, 12, 11, 10, 11, 12, 13]
        lows  = [ 8,  9, 10, 11, 23, 11, 10,  9,  8,  9, 10, 11, 18, 11, 10,  9,  5,  9, 10, 11]
        result = _det_classify(highs, lows)
        assert result["structure_bias"].iloc[-1] == "bearish"

    def test_HH_label_on_correct_bar(self):
        highs = [10, 11, 12, 13, 20, 13, 12, 11, 10, 11, 12, 13, 25, 13, 12, 11, 10, 11, 12, 13]
        lows  = [ 8,  9, 10, 11, 18, 11, 10,  9,  5,  9, 10, 11, 23, 11, 10,  9,  8,  9, 10, 11]
        result = _det_classify(highs, lows)
        # Peak 2 (high=25) confirmed at bar 14
        assert result["swing_label_high"].iloc[14] == "HH"

    def test_HL_label_on_correct_bar(self):
        highs = [10, 11, 12, 13, 20, 13, 12, 11, 10, 11, 12, 13, 25, 13, 12, 11, 10, 11, 12, 13]
        lows  = [ 8,  9, 10, 11, 18, 11, 10,  9,  5,  9, 10, 11, 23, 11, 10,  9,  8,  9, 10, 11]
        result = _det_classify(highs, lows)
        # Trough 2 (low=8) confirmed at bar 18
        assert result["swing_label_low"].iloc[18] == "HL"

    def test_flat_series_produces_no_labels(self):
        highs = [10.0] * 20
        lows  = [8.0]  * 20
        result = _det_classify(highs, lows)
        assert result["swing_label_high"].isna().all() or (result["swing_label_high"] == None).all()
        assert (result["structure_bias"] == "neutral").all()

    def test_1m_timeframe_integration(self):
        """window=3, lag=1. Integration test on 1m data."""
        highs = [10, 11, 20, 11, 10,  9, 11, 12, 25, 12, 11, 10, 9, 8]
        lows  = [ 8,  9, 18,  9,  5,  7,  9, 10, 23, 10,  9,  8, 7, 6]
        result = _det_classify(highs, lows, tf="1m", freq="1min")
        # At least one HH should exist if peak 2 > peak 1
        high_labels = result["swing_label_high"].dropna()
        assert "HH" in high_labels.values
