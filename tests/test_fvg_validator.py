"""
Tests for fvg_validator.py — Phase 2.5.

Critical properties:
  - Validation only runs on bars where an FVG is present.
  - Bars without an FVG must produce fvg_valid=None, fvg_invalid_reason=None.
  - bull FVG  ⇒ requires structure_bias == "bullish"  AND  displacement at c2 == "bull"
  - bear FVG  ⇒ requires structure_bias == "bearish"  AND  displacement at c2 == "bear"
  - Size check uses STRICT > (size > min_size_atr_pct × ATR).
  - Reason returned is the FIRST failing check, in order: bias → size → displacement.
"""

import pytest
import numpy as np
import pandas as pd

from core.smc.fvg_validator import FVGValidator, validate_fvgs


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _blank_df(n=20, start="2026-01-05 10:00", freq="5min",
              high=10.0, low=9.0, close=9.5, opn=9.5):
    """A DataFrame with all the columns the validator expects."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open":              [opn]    * n,
            "high":              [high]   * n,
            "low":               [low]    * n,
            "close":             [close]  * n,
            "volume":            [100.0]  * n,
            "fvg_type":          pd.Series([None] * n, dtype=object, index=idx),
            "fvg_top":           [np.nan] * n,
            "fvg_bottom":        [np.nan] * n,
            "fvg_size":          [np.nan] * n,
            "fvg_c1_idx":        [-1]     * n,
            "displacement_type": pd.Series([None] * n, dtype=object, index=idx),
            "structure_bias":    ["neutral"] * n,
        },
        index=idx,
    )


def _set_fvg(df, pos, fvg_type, top, bottom, size, c1_idx):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("fvg_type")]   = fvg_type
    df.iloc[pos, df.columns.get_loc("fvg_top")]    = float(top)
    df.iloc[pos, df.columns.get_loc("fvg_bottom")] = float(bottom)
    df.iloc[pos, df.columns.get_loc("fvg_size")]   = float(size)
    df.iloc[pos, df.columns.get_loc("fvg_c1_idx")] = int(c1_idx)
    return df


def _set_disp(df, pos, disp_type):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("displacement_type")] = disp_type
    return df


def _set_bias(df, from_pos, bias):
    df = df.copy()
    df.iloc[from_pos:, df.columns.get_loc("structure_bias")] = bias
    return df


def _vd(**kwargs):
    return FVGValidator(**kwargs)


# ------------------------------------------------------------------ #
# Bias alignment                                                       #
# ------------------------------------------------------------------ #

class TestBiasAlignment:
    def test_bull_fvg_with_bullish_bias_passes_bias_check(self):
        # Set up FVG, displacement, and bias so only bias is at issue
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bull", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bull")  # c2 = c1+1 = 9
        df = _set_bias(df, from_pos=0, bias="bullish")
        result = _vd().validate(df)
        # Should not fail on bias (might still fail on size/disp — but here it's all valid)
        assert result["fvg_valid"].iloc[10] is True

    def test_bull_fvg_with_bearish_bias_fails_bias(self):
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bull", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bull")
        df = _set_bias(df, from_pos=0, bias="bearish")
        result = _vd().validate(df)
        assert result["fvg_valid"].iloc[10] is False
        assert result["fvg_invalid_reason"].iloc[10] == "bias"

    def test_bull_fvg_with_neutral_bias_fails_bias(self):
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bull", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bull")
        # bias stays "neutral" by default
        result = _vd().validate(df)
        assert result["fvg_valid"].iloc[10] is False
        assert result["fvg_invalid_reason"].iloc[10] == "bias"

    def test_bear_fvg_with_bearish_bias_passes_bias_check(self):
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bear", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bear")
        df = _set_bias(df, from_pos=0, bias="bearish")
        result = _vd().validate(df)
        assert result["fvg_valid"].iloc[10] is True

    def test_bear_fvg_with_bullish_bias_fails_bias(self):
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bear", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bear")
        df = _set_bias(df, from_pos=0, bias="bullish")
        result = _vd().validate(df)
        assert result["fvg_valid"].iloc[10] is False
        assert result["fvg_invalid_reason"].iloc[10] == "bias"


# ------------------------------------------------------------------ #
# Size threshold                                                      #
# ------------------------------------------------------------------ #

class TestSizeThreshold:
    def test_size_well_above_threshold_passes(self):
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bull", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bull")
        df = _set_bias(df, from_pos=0, bias="bullish")
        result = _vd().validate(df)   # default 0.5 × ATR ≈ 0.5
        assert result["fvg_valid"].iloc[10] is True

    def test_size_below_threshold_fails(self):
        df = _blank_df()
        # Tiny gap: size=0.1, ATR≈1, threshold=0.5 → 0.1 ≤ 0.5 → fail
        df = _set_fvg(df, pos=10, fvg_type="bull", top=10.1, bottom=10.0, size=0.1, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bull")
        df = _set_bias(df, from_pos=0, bias="bullish")
        result = _vd().validate(df)
        assert result["fvg_valid"].iloc[10] is False
        assert result["fvg_invalid_reason"].iloc[10] == "size"

    def test_size_threshold_strict_inequality(self):
        """size == threshold is NOT enough (rule is size > threshold)."""
        df = _blank_df()
        # Setup: ATR=1.0, threshold=0.5×1=0.5; gap size=0.5 → must fail
        df = _set_fvg(df, pos=10, fvg_type="bull", top=10.5, bottom=10.0, size=0.5, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bull")
        df = _set_bias(df, from_pos=0, bias="bullish")
        result = _vd().validate(df)
        # The rolling-mean ATR may not be exactly 1.0 (last bar's TR can drift it),
        # so this verifies "size at-or-below threshold fails", regardless of ATR's exact value
        # If ATR ≥ 1 → threshold ≥ 0.5 → size 0.5 NOT > threshold → fail
        assert result["fvg_valid"].iloc[10] is False
        assert result["fvg_invalid_reason"].iloc[10] == "size"

    def test_lower_threshold_admits_smaller_gaps(self):
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bull", top=10.2, bottom=10.0, size=0.2, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bull")
        df = _set_bias(df, from_pos=0, bias="bullish")
        # Default 0.5 → fails
        assert _vd().validate(df)["fvg_valid"].iloc[10] is False
        # Loosened 0.1 → passes (0.2 > 0.1×ATR ≈ 0.1)
        assert _vd(min_size_atr_pct=0.1).validate(df)["fvg_valid"].iloc[10] is True


# ------------------------------------------------------------------ #
# Displacement check                                                   #
# ------------------------------------------------------------------ #

class TestDisplacementCheck:
    def test_bull_fvg_with_bull_displacement_at_c2_passes(self):
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bull", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bull")
        df = _set_bias(df, from_pos=0, bias="bullish")
        result = _vd().validate(df)
        assert result["fvg_valid"].iloc[10] is True

    def test_bull_fvg_with_no_displacement_fails(self):
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bull", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        # No displacement set at bar 9
        df = _set_bias(df, from_pos=0, bias="bullish")
        result = _vd().validate(df)
        assert result["fvg_valid"].iloc[10] is False
        assert result["fvg_invalid_reason"].iloc[10] == "displacement"

    def test_bull_fvg_with_bear_displacement_fails(self):
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bull", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bear")   # wrong direction
        df = _set_bias(df, from_pos=0, bias="bullish")
        result = _vd().validate(df)
        assert result["fvg_valid"].iloc[10] is False
        assert result["fvg_invalid_reason"].iloc[10] == "displacement"

    def test_bear_fvg_with_bear_displacement_passes(self):
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bear", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bear")
        df = _set_bias(df, from_pos=0, bias="bearish")
        result = _vd().validate(df)
        assert result["fvg_valid"].iloc[10] is True

    def test_displacement_at_wrong_bar_fails(self):
        """Displacement at c1 itself (bar 8) doesn't count — must be at c1+1."""
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bull", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        df = _set_disp(df, pos=8, disp_type="bull")   # at c1, not c2
        df = _set_bias(df, from_pos=0, bias="bullish")
        result = _vd().validate(df)
        assert result["fvg_valid"].iloc[10] is False
        assert result["fvg_invalid_reason"].iloc[10] == "displacement"


# ------------------------------------------------------------------ #
# Reason precedence                                                    #
# ------------------------------------------------------------------ #

class TestReasonPrecedence:
    def test_bias_reported_first_when_multiple_fail(self):
        """All three checks fail — reason should be 'bias' (the first one)."""
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bull", top=10.05, bottom=10.0, size=0.05, c1_idx=8)
        # No displacement, neutral bias, size below threshold — all fail
        result = _vd().validate(df)
        assert result["fvg_valid"].iloc[10] is False
        assert result["fvg_invalid_reason"].iloc[10] == "bias"

    def test_size_reported_when_bias_passes(self):
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bull", top=10.05, bottom=10.0, size=0.05, c1_idx=8)
        df = _set_bias(df, from_pos=0, bias="bullish")
        # Bias passes; size fails; displacement also fails
        result = _vd().validate(df)
        assert result["fvg_invalid_reason"].iloc[10] == "size"


# ------------------------------------------------------------------ #
# No-FVG bars                                                          #
# ------------------------------------------------------------------ #

class TestNoFVGBars:
    def test_no_fvg_means_valid_is_none(self):
        df = _blank_df()
        result = _vd().validate(df)
        for i in range(len(df)):
            assert result["fvg_valid"].iloc[i] is None
            assert result["fvg_invalid_reason"].iloc[i] is None

    def test_only_fvg_bar_is_validated(self):
        df = _blank_df()
        df = _set_fvg(df, pos=10, fvg_type="bull", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bull")
        df = _set_bias(df, from_pos=0, bias="bullish")
        result = _vd().validate(df)
        # Only bar 10 has a result
        assert result["fvg_valid"].iloc[10] is True
        for i in range(len(df)):
            if i != 10:
                assert result["fvg_valid"].iloc[i] is None


# ------------------------------------------------------------------ #
# Bias column override (Phase 1.6 'resolved_bias')                     #
# ------------------------------------------------------------------ #

class TestBiasColumnOverride:
    def test_can_use_resolved_bias_column(self):
        df = _blank_df()
        df["resolved_bias"] = ["bullish"] * len(df)
        df = _set_fvg(df, pos=10, fvg_type="bull", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bull")
        # structure_bias still neutral; resolved_bias is bullish
        result = _vd(bias_column="resolved_bias").validate(df)
        assert result["fvg_valid"].iloc[10] is True

    def test_missing_specified_bias_column_raises(self):
        df = _blank_df()
        with pytest.raises(ValueError, match="Missing"):
            _vd(bias_column="resolved_bias").validate(df)


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _blank_df()
        result = _vd().validate(df)
        assert "fvg_valid" in result.columns
        assert "fvg_invalid_reason" in result.columns

    def test_output_index_unchanged(self):
        df = _blank_df()
        pd.testing.assert_index_equal(_vd().validate(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _blank_df()
        _vd().validate(df)
        assert "fvg_valid" not in df.columns

    def test_existing_columns_preserved(self):
        df = _blank_df()
        result = _vd().validate(df)
        for col in ["high", "low", "close", "fvg_type", "displacement_type",
                    "structure_bias"]:
            assert col in result.columns


# ------------------------------------------------------------------ #
# Validation                                                            #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_negative_size_threshold_raises(self):
        with pytest.raises(ValueError, match="min_size_atr_pct"):
            FVGValidator(min_size_atr_pct=-0.1)

    def test_empty_bias_column_raises(self):
        with pytest.raises(ValueError, match="bias_column"):
            FVGValidator(bias_column="")

    def test_atr_period_zero_raises(self):
        with pytest.raises(ValueError, match="atr_period"):
            FVGValidator(atr_period=0)

    def test_missing_fvg_type_raises(self):
        df = _blank_df().drop(columns=["fvg_type"])
        with pytest.raises(ValueError, match="Missing"):
            _vd().validate(df)

    def test_missing_displacement_type_raises(self):
        df = _blank_df().drop(columns=["displacement_type"])
        with pytest.raises(ValueError, match="Missing"):
            _vd().validate(df)

    def test_missing_structure_bias_raises(self):
        df = _blank_df().drop(columns=["structure_bias"])
        with pytest.raises(ValueError, match="Missing"):
            _vd().validate(df)

    def test_non_datetime_index_raises(self):
        df = _blank_df()
        df.index = range(len(df))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _vd().validate(df)

    def test_convenience_function_works(self):
        df = _blank_df()
        result = validate_fvgs(df)
        assert "fvg_valid" in result.columns


# ------------------------------------------------------------------ #
# Multiple FVGs                                                         #
# ------------------------------------------------------------------ #

class TestMultipleFVGs:
    def test_each_fvg_validated_independently(self):
        df = _blank_df(n=30)
        # FVG 1 at bar 10 — should pass
        df = _set_fvg(df, pos=10, fvg_type="bull", top=15.0, bottom=10.0, size=5.0, c1_idx=8)
        df = _set_disp(df, pos=9, disp_type="bull")
        # FVG 2 at bar 20 — same setup but no displacement → should fail
        df = _set_fvg(df, pos=20, fvg_type="bull", top=15.0, bottom=10.0, size=5.0, c1_idx=18)
        df = _set_bias(df, from_pos=0, bias="bullish")
        result = _vd().validate(df)
        assert result["fvg_valid"].iloc[10] is True
        assert result["fvg_valid"].iloc[20] is False
        assert result["fvg_invalid_reason"].iloc[20] == "displacement"
