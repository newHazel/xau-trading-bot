"""
Tests for order_block_detector.py — Phase 2.8.

Critical properties:
  - Bullish OB = last bearish candle (close < open) before a bullish impulse.
  - Bearish OB = last bullish candle (close > open) before a bearish impulse.
  - Triggers: BOS (Phase 1.3) and/or FVG (Phase 2.3).
  - Doji candles (close == open) are skipped in the backward scan.
  - Lookback capped at max_lookback (default 10).
  - Same candle marked only once (first trigger wins).
  - No-trigger bars get ob_type=None.
"""

import pytest
import numpy as np
import pandas as pd

from core.smc.order_block_detector import (
    OrderBlockDetector,
    detect_order_blocks,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _blank_df(n=20, start="2026-01-05 10:00", freq="5min",
              open_=100.0, high=101.0, low=99.0, close=100.5):
    """Default: all bars are mildly bullish (close > open)."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open":   [open_] * n,
            "high":   [high]  * n,
            "low":    [low]   * n,
            "close":  [close] * n,
            "volume": [100.0] * n,
        },
        index=idx,
    )


def _add_bos_columns(df):
    df = df.copy()
    df["bos_bull"] = np.nan
    df["bos_bear"] = np.nan
    df["bos_bull_ref_bar"] = -1
    df["bos_bear_ref_bar"] = -1
    return df


def _add_fvg_columns(df):
    df = df.copy()
    n = len(df)
    df["fvg_type"]   = pd.Series([None] * n, dtype=object, index=df.index)
    df["fvg_top"]    = np.nan
    df["fvg_bottom"] = np.nan
    df["fvg_size"]   = np.nan
    df["fvg_c1_idx"] = np.nan
    return df


def _set_bar(df, pos, o, h, l, c):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("open")]  = float(o)
    df.iloc[pos, df.columns.get_loc("high")]  = float(h)
    df.iloc[pos, df.columns.get_loc("low")]   = float(l)
    df.iloc[pos, df.columns.get_loc("close")] = float(c)
    return df


def _set_bos(df, pos, direction, level):
    df = df.copy()
    col = f"bos_{direction}"
    df.iloc[pos, df.columns.get_loc(col)] = float(level)
    return df


def _set_fvg(df, pos, fvg_type, c1_idx):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("fvg_type")]   = fvg_type
    df.iloc[pos, df.columns.get_loc("fvg_c1_idx")] = float(c1_idx)
    return df


def _ob(**kwargs):
    return OrderBlockDetector(**kwargs)


# ------------------------------------------------------------------ #
# BOS-triggered Order Blocks — Bull                                    #
# ------------------------------------------------------------------ #

class TestBOSTriggerBull:
    """Bull BOS at bar j → scan backward for last bearish candle → bullish OB."""

    def _base(self):
        """Bars 0-19, all mildly bullish. Add BOS columns."""
        return _add_bos_columns(_blank_df())

    def test_bull_ob_from_bos(self):
        """Bar 5 is bearish; bar 8 has bull BOS → OB at bar 5."""
        df = self._base()
        df = _set_bar(df, 5, o=102, h=103, l=99, c=99.5)  # bearish
        df = _set_bos(df, 8, "bull", 105.0)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[5] == "bull"
        assert result["ob_top"].iloc[5] == 103.0
        assert result["ob_bottom"].iloc[5] == 99.0
        assert result["ob_trigger_bar"].iloc[5] == 8
        assert result["ob_trigger_type"].iloc[5] == "bos"

    def test_takes_nearest_bearish_candle(self):
        """Two bearish candles at bars 4,6 — BOS at bar 8 → OB at bar 6 (nearest)."""
        df = self._base()
        df = _set_bar(df, 4, o=102, h=103, l=99, c=99.5)  # bearish
        df = _set_bar(df, 6, o=102, h=103, l=98, c=99.0)  # bearish (nearer)
        df = _set_bos(df, 8, "bull", 105.0)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[6] == "bull"
        assert result["ob_type"].iloc[4] is None  # not this one

    def test_skips_doji(self):
        """Bar 7 is doji (close==open), bar 6 is bearish → OB at bar 6."""
        df = self._base()
        df = _set_bar(df, 7, o=100, h=101, l=99, c=100)   # doji
        df = _set_bar(df, 6, o=102, h=103, l=99, c=99.5)  # bearish
        df = _set_bos(df, 8, "bull", 105.0)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[6] == "bull"
        assert result["ob_type"].iloc[7] is None

    def test_no_ob_when_all_bullish(self):
        """All bars are bullish → no opposing candle found → no OB."""
        df = self._base()
        df = _set_bos(df, 8, "bull", 105.0)
        result = _ob().detect(df)
        # No bar should be marked as OB
        assert result["ob_type"].notna().sum() == 0

    def test_max_lookback_limits_scan(self):
        """Bearish candle at bar 0, BOS at bar 15. With max_lookback=5, too far."""
        df = self._base()
        df = _set_bar(df, 0, o=102, h=103, l=99, c=99.5)  # bearish
        df = _set_bos(df, 15, "bull", 105.0)
        result = _ob(max_lookback=5).detect(df)
        assert result["ob_type"].notna().sum() == 0

    def test_max_lookback_exact_boundary(self):
        """Bearish at bar 10, BOS at bar 14. Scan from 13 with lookback=4 → reaches bar 10."""
        df = self._base()
        df = _set_bar(df, 10, o=102, h=103, l=99, c=99.5)
        df = _set_bos(df, 14, "bull", 105.0)
        result = _ob(max_lookback=4).detect(df)
        assert result["ob_type"].iloc[10] == "bull"


# ------------------------------------------------------------------ #
# BOS-triggered Order Blocks — Bear                                    #
# ------------------------------------------------------------------ #

class TestBOSTriggerBear:
    """Bear BOS at bar j → scan backward for last bullish candle → bearish OB."""

    def _base(self):
        """All bars are mildly bearish (close < open)."""
        return _add_bos_columns(
            _blank_df(open_=101.0, high=102.0, low=99.0, close=100.0)
        )

    def test_bear_ob_from_bos(self):
        """Bar 5 is bullish; bar 8 has bear BOS → OB at bar 5."""
        df = self._base()
        df = _set_bar(df, 5, o=99, h=103, l=98, c=102)  # bullish
        df = _set_bos(df, 8, "bear", 95.0)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[5] == "bear"
        assert result["ob_top"].iloc[5] == 103.0
        assert result["ob_bottom"].iloc[5] == 98.0
        assert result["ob_trigger_bar"].iloc[5] == 8
        assert result["ob_trigger_type"].iloc[5] == "bos"

    def test_takes_nearest_bullish_candle(self):
        df = self._base()
        df = _set_bar(df, 4, o=99, h=103, l=98, c=102)  # bullish
        df = _set_bar(df, 6, o=99, h=104, l=98, c=103)  # bullish (nearer)
        df = _set_bos(df, 8, "bear", 95.0)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[6] == "bear"
        assert result["ob_type"].iloc[4] is None


# ------------------------------------------------------------------ #
# FVG-triggered Order Blocks                                           #
# ------------------------------------------------------------------ #

class TestFVGTrigger:
    """FVG at bar j (c3) → impulse candle c2 = c1+1 → scan from c2-1."""

    def _base(self):
        return _add_fvg_columns(_blank_df())

    def test_bull_ob_from_fvg(self):
        """Bull FVG: c1=5, c2=6 (impulse), scan from 5 backward.
        Bar 4 is bearish → OB at bar 4."""
        df = self._base()
        df = _set_bar(df, 4, o=102, h=103, l=99, c=99.5)  # bearish
        df = _set_fvg(df, 8, "bull", c1_idx=5)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[4] == "bull"
        assert result["ob_trigger_bar"].iloc[4] == 8
        assert result["ob_trigger_type"].iloc[4] == "fvg"

    def test_bear_ob_from_fvg(self):
        """Bear FVG: c1=5, c2=6, scan from 5. Bar 3 is bullish → OB at 3."""
        df = self._base()
        # Make bar 3 bullish (default bars are already bullish: c=100.5 > o=100)
        # but let's be explicit
        df = _set_bar(df, 3, o=99, h=103, l=98, c=102)  # clearly bullish
        # Bars 4-5 need to be bearish or doji so the scan finds bar 3
        df = _set_bar(df, 4, o=102, h=103, l=99, c=99.5)  # bearish
        df = _set_bar(df, 5, o=102, h=103, l=99, c=99.0)  # bearish
        df = _set_fvg(df, 8, "bear", c1_idx=5)
        result = _ob().detect(df)
        # scan from c2-1 = 5 backward. bar 5 bearish → skip. bar 4 bearish → skip.
        # bar 3 bullish → OB!
        assert result["ob_type"].iloc[3] == "bear"
        assert result["ob_trigger_type"].iloc[3] == "fvg"

    def test_fvg_impulse_candle_not_scanned(self):
        """The impulse candle itself (c2) is NOT scanned — scan starts at c2-1."""
        df = self._base()
        # c1=5, c2=6. Make bar 6 bearish, bar 5 default (bullish).
        # For a bull FVG, we need a bearish candle. Bar 6 IS the impulse,
        # should not be the OB. Bar 5 is bullish → not opposing.
        # Bar 4 default bullish → not opposing. No bearish candles → no OB.
        df = _set_bar(df, 6, o=102, h=103, l=99, c=99.5)  # bearish but c2
        df = _set_fvg(df, 8, "bull", c1_idx=5)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[6] is None  # c2 not scanned


# ------------------------------------------------------------------ #
# Both BOS + FVG triggers                                              #
# ------------------------------------------------------------------ #

class TestBothTriggers:
    """When both BOS and FVG columns present, both sources are used."""

    def _base(self):
        df = _blank_df()
        df = _add_bos_columns(df)
        df = _add_fvg_columns(df)
        return df

    def test_same_candle_marked_only_once(self):
        """BOS at bar 10 and FVG at bar 12 both point to bearish candle at bar 7.
        First trigger (BOS) wins."""
        df = self._base()
        df = _set_bar(df, 7, o=102, h=103, l=99, c=99.5)  # bearish
        # All bars 8-9 are default bullish → BOS scan from 9 finds bar 7
        df = _set_bos(df, 10, "bull", 105.0)
        # FVG: c1=8, c2=9. Scan from 8 → bar 7 is bearish
        df = _set_fvg(df, 12, "bull", c1_idx=8)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[7] == "bull"
        assert result["ob_trigger_type"].iloc[7] == "bos"  # BOS processed first

    def test_different_obs_from_different_triggers(self):
        """BOS finds one OB, FVG finds a different one."""
        df = self._base()
        # Bearish candle at bar 3 → for BOS at bar 5
        df = _set_bar(df, 3, o=102, h=103, l=99, c=99.5)
        df = _set_bos(df, 5, "bull", 105.0)
        # Bearish candle at bar 8 → for FVG at bar 12 (c1=9, c2=10)
        df = _set_bar(df, 8, o=102, h=103, l=98, c=99.0)
        df = _set_fvg(df, 12, "bull", c1_idx=9)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[3] == "bull"
        assert result["ob_trigger_type"].iloc[3] == "bos"
        assert result["ob_type"].iloc[8] == "bull"
        assert result["ob_trigger_type"].iloc[8] == "fvg"

    def test_works_with_only_bos(self):
        """Only BOS columns present, no FVG columns."""
        df = _add_bos_columns(_blank_df())
        df = _set_bar(df, 5, o=102, h=103, l=99, c=99.5)
        df = _set_bos(df, 8, "bull", 105.0)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[5] == "bull"

    def test_works_with_only_fvg(self):
        """Only FVG columns present, no BOS columns."""
        df = _add_fvg_columns(_blank_df())
        df = _set_bar(df, 4, o=102, h=103, l=99, c=99.5)
        df = _set_fvg(df, 8, "bull", c1_idx=5)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[4] == "bull"


# ------------------------------------------------------------------ #
# No-trigger bars                                                      #
# ------------------------------------------------------------------ #

class TestNoTrigger:
    def test_no_triggers_means_no_obs(self):
        df = _add_bos_columns(_blank_df())
        result = _ob().detect(df)
        assert result["ob_type"].notna().sum() == 0

    def test_only_ob_bar_gets_type(self):
        df = _add_bos_columns(_blank_df())
        df = _set_bar(df, 5, o=102, h=103, l=99, c=99.5)
        df = _set_bos(df, 8, "bull", 105.0)
        result = _ob().detect(df)
        for i in range(len(df)):
            if i != 5:
                assert result["ob_type"].iloc[i] is None


# ------------------------------------------------------------------ #
# Multiple OBs                                                         #
# ------------------------------------------------------------------ #

class TestMultipleOBs:
    def test_bull_and_bear_obs_in_same_df(self):
        df = _add_bos_columns(_blank_df())
        # Bearish candle at bar 3 → bull OB (from bull BOS at bar 5)
        df = _set_bar(df, 3, o=102, h=103, l=99, c=99.5)
        df = _set_bos(df, 5, "bull", 105.0)
        # Bullish candle at bar 10 → bear OB (from bear BOS at bar 13)
        df = _set_bar(df, 10, o=99, h=104, l=98, c=103)
        # Make bars 11-12 also bullish (default) but we need bearish for bear BOS
        # Actually for bear BOS we need to scan for a bullish candle, and defaults are bullish
        # So bar 12 (default: o=100, c=100.5, bullish) is nearer → OB at bar 12
        df = _set_bos(df, 13, "bear", 95.0)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[3] == "bull"
        assert result["ob_type"].iloc[12] == "bear"

    def test_consecutive_bos_find_separate_obs(self):
        df = _add_bos_columns(_blank_df())
        # Bearish candle at bar 2
        df = _set_bar(df, 2, o=102, h=103, l=99, c=99.5)
        df = _set_bos(df, 4, "bull", 105.0)
        # Bearish candle at bar 7
        df = _set_bar(df, 7, o=102, h=103, l=98, c=99.0)
        df = _set_bos(df, 9, "bull", 110.0)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[2] == "bull"
        assert result["ob_type"].iloc[7] == "bull"


# ------------------------------------------------------------------ #
# Edge cases                                                            #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_bos_at_bar_0_no_lookback(self):
        """BOS at bar 0 → no bars to scan → no OB."""
        df = _add_bos_columns(_blank_df())
        df = _set_bos(df, 0, "bull", 105.0)
        result = _ob().detect(df)
        assert result["ob_type"].notna().sum() == 0

    def test_bos_at_bar_1_scans_bar_0(self):
        """BOS at bar 1 → scan bar 0 only."""
        df = _add_bos_columns(_blank_df())
        df = _set_bar(df, 0, o=102, h=103, l=99, c=99.5)  # bearish
        df = _set_bos(df, 1, "bull", 105.0)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[0] == "bull"

    def test_fvg_c1_at_bar_0(self):
        """FVG with c1=0 → c2=1, scan from bar 0. Bar 0 if bearish → OB."""
        df = _add_fvg_columns(_blank_df())
        df = _set_bar(df, 0, o=102, h=103, l=99, c=99.5)  # bearish
        df = _set_fvg(df, 3, "bull", c1_idx=0)
        result = _ob().detect(df)
        assert result["ob_type"].iloc[0] == "bull"

    def test_all_doji_no_ob(self):
        """All bars are doji (close==open) → no opposing candle → no OB."""
        df = _blank_df(open_=100, high=101, low=99, close=100)
        df = _add_bos_columns(df)
        df = _set_bos(df, 8, "bull", 105.0)
        result = _ob().detect(df)
        assert result["ob_type"].notna().sum() == 0

    def test_ob_zone_uses_high_low(self):
        """OB zone = [low, high] of the OB candle."""
        df = _add_bos_columns(_blank_df())
        df = _set_bar(df, 5, o=102, h=106.5, l=97.3, c=99.5)
        df = _set_bos(df, 8, "bull", 105.0)
        result = _ob().detect(df)
        assert result["ob_top"].iloc[5] == pytest.approx(106.5)
        assert result["ob_bottom"].iloc[5] == pytest.approx(97.3)


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _add_bos_columns(_blank_df())
        result = _ob().detect(df)
        for col in ["ob_type", "ob_top", "ob_bottom", "ob_trigger_bar", "ob_trigger_type"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _add_bos_columns(_blank_df())
        pd.testing.assert_index_equal(_ob().detect(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _add_bos_columns(_blank_df())
        _ob().detect(df)
        assert "ob_type" not in df.columns

    def test_defaults_when_no_ob(self):
        df = _add_bos_columns(_blank_df())
        result = _ob().detect(df)
        assert (result["ob_trigger_bar"] == -1).all()
        assert result["ob_top"].isna().all()
        assert result["ob_bottom"].isna().all()


# ------------------------------------------------------------------ #
# Validation                                                            #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_no_trigger_columns_raises(self):
        df = _blank_df()  # no BOS or FVG columns
        with pytest.raises(ValueError, match="trigger source"):
            _ob().detect(df)

    def test_missing_ohlc_raises(self):
        df = _add_bos_columns(_blank_df()).drop(columns=["open"])
        with pytest.raises(ValueError, match="Missing"):
            _ob().detect(df)

    def test_non_datetime_index_raises(self):
        df = _add_bos_columns(_blank_df())
        df.index = range(len(df))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _ob().detect(df)

    def test_zero_lookback_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            OrderBlockDetector(max_lookback=0)

    def test_negative_lookback_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            OrderBlockDetector(max_lookback=-1)

    def test_convenience_function_works(self):
        df = _add_bos_columns(_blank_df())
        result = detect_order_blocks(df)
        assert "ob_type" in result.columns


# ------------------------------------------------------------------ #
# Accessor                                                             #
# ------------------------------------------------------------------ #

class TestAccessor:
    def test_get_order_blocks_returns_newest_first(self):
        df = _add_bos_columns(_blank_df())
        df = _set_bar(df, 3, o=102, h=103, l=99, c=99.5)
        df = _set_bos(df, 5, "bull", 105.0)
        df = _set_bar(df, 9, o=102, h=103, l=98, c=99.0)
        df = _set_bos(df, 11, "bull", 110.0)
        result = _ob().detect(df)
        obs = _ob().get_order_blocks(result)
        assert len(obs) == 2
        assert obs[0]["timestamp"] >= obs[1]["timestamp"]

    def test_filter_by_direction(self):
        df = _add_bos_columns(_blank_df())
        # Bull OB
        df = _set_bar(df, 3, o=102, h=103, l=99, c=99.5)
        df = _set_bos(df, 5, "bull", 105.0)
        # Bear OB — need bullish candle before bear BOS. Default bars are bullish.
        df = _set_bos(df, 12, "bear", 95.0)
        result = _ob().detect(df)
        bull_obs = _ob().get_order_blocks(result, direction="bull")
        bear_obs = _ob().get_order_blocks(result, direction="bear")
        assert all(o["ob_type"] == "bull" for o in bull_obs)
        assert all(o["ob_type"] == "bear" for o in bear_obs)

    def test_empty_when_no_obs(self):
        df = _add_bos_columns(_blank_df())
        result = _ob().detect(df)
        assert _ob().get_order_blocks(result) == []

    def test_n_limits_results(self):
        df = _add_bos_columns(_blank_df(n=30))
        for pos in [3, 7, 12, 17]:
            df = _set_bar(df, pos, o=102, h=103, l=99, c=99.5)
            df = _set_bos(df, pos + 2, "bull", 105.0 + pos)
        result = _ob().detect(df)
        obs = _ob().get_order_blocks(result, n=2)
        assert len(obs) == 2


# ------------------------------------------------------------------ #
# Custom lookback                                                      #
# ------------------------------------------------------------------ #

class TestCustomLookback:
    def test_lookback_1_only_immediate_predecessor(self):
        df = _add_bos_columns(_blank_df())
        # Bar 6 is bearish, bar 7 is default bullish, BOS at bar 8
        df = _set_bar(df, 6, o=102, h=103, l=99, c=99.5)
        df = _set_bos(df, 8, "bull", 105.0)
        # With lookback=1, scan only bar 7. Bar 7 is bullish → no OB
        result = _ob(max_lookback=1).detect(df)
        assert result["ob_type"].notna().sum() == 0

    def test_lookback_2_finds_it(self):
        df = _add_bos_columns(_blank_df())
        df = _set_bar(df, 6, o=102, h=103, l=99, c=99.5)
        df = _set_bos(df, 8, "bull", 105.0)
        # With lookback=2, scan bars 7,6. Bar 6 is bearish → OB
        result = _ob(max_lookback=2).detect(df)
        assert result["ob_type"].iloc[6] == "bull"
