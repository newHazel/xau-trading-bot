"""
Tests for multi_touch_lifecycle.py — Phase 2.7.

Critical properties:
  - A "touch" = transition from outside → inside the FVG zone.
  - Bull FVG: inside when bar low < top; bear FVG: inside when bar high > bottom.
  - touch_count is the total distinct touches over the FVG's life.
  - Tradeability depends on mode: backtest≤3, paper≤2, live≤1 (defaults).
  - full/invalidated mitigation state overrides tradeability to False.
  - No-FVG bars get touch_count=-1, touch_tradeable=None.
"""

import pytest
import numpy as np
import pandas as pd

from core.smc.multi_touch_lifecycle import (
    MultiTouchLifecycle,
    track_touches,
    MODE_DEFAULTS,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _blank_df(n=20, start="2026-01-05 10:00", freq="5min",
              high=30.0, low=29.0, close=29.5):
    """Default OHLC sits well ABOVE a typical bull gap [10,15]."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open":       [close] * n,
            "high":       [high]  * n,
            "low":        [low]   * n,
            "close":      [close] * n,
            "volume":     [100.0] * n,
            "fvg_type":   pd.Series([None] * n, dtype=object, index=idx),
            "fvg_top":    [np.nan] * n,
            "fvg_bottom": [np.nan] * n,
        },
        index=idx,
    )


def _set_fvg(df, pos, fvg_type, top, bottom):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("fvg_type")]   = fvg_type
    df.iloc[pos, df.columns.get_loc("fvg_top")]    = float(top)
    df.iloc[pos, df.columns.get_loc("fvg_bottom")] = float(bottom)
    return df


def _set_bar(df, pos, h, l, c=None):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("high")]  = float(h)
    df.iloc[pos, df.columns.get_loc("low")]   = float(l)
    if c is not None:
        df.iloc[pos, df.columns.get_loc("close")] = float(c)
    return df


def _add_mitigation(df, pos, state):
    """Add mitigation_state column (if needed) and set value at pos."""
    if "mitigation_state" not in df.columns:
        df = df.copy()
        df["mitigation_state"] = pd.Series([None] * len(df), dtype=object, index=df.index)
    else:
        df = df.copy()
    df.iloc[pos, df.columns.get_loc("mitigation_state")] = state
    return df


def _mt(**kwargs):
    return MultiTouchLifecycle(**kwargs)


# ------------------------------------------------------------------ #
# Touch counting — Bull FVG                                            #
# ------------------------------------------------------------------ #

class TestBullTouchCount:
    """Bull FVG [10, 15]. Inside = low < 15. Default bars have low=29 (outside)."""

    def _base(self):
        df = _blank_df()
        return _set_fvg(df, 5, "bull", top=15.0, bottom=10.0)

    def test_zero_touches_when_price_never_enters(self):
        df = self._base()  # all bars low=29, never < 15
        result = _mt().track(df)
        assert result["touch_count"].iloc[5] == 0

    def test_one_touch_single_bar(self):
        df = self._base()
        df = _set_bar(df, 8, h=20, l=14.0)  # low=14 < 15 → inside
        result = _mt().track(df)
        assert result["touch_count"].iloc[5] == 1

    def test_one_touch_consecutive_bars(self):
        """Two consecutive bars inside = still 1 touch (no exit between)."""
        df = self._base()
        df = _set_bar(df, 8, h=20, l=14.0)
        df = _set_bar(df, 9, h=20, l=13.0)
        result = _mt().track(df)
        assert result["touch_count"].iloc[5] == 1

    def test_two_touches_with_exit_between(self):
        """Bar 8 inside, bar 9 outside (low=29), bar 10 inside again."""
        df = self._base()
        df = _set_bar(df, 8, h=20, l=14.0)   # touch 1
        # bar 9 stays default (low=29) → outside
        df = _set_bar(df, 10, h=20, l=13.0)  # touch 2
        result = _mt().track(df)
        assert result["touch_count"].iloc[5] == 2

    def test_three_touches(self):
        df = self._base()
        df = _set_bar(df, 7, h=20, l=14.0)   # touch 1
        # bar 8 default → outside
        df = _set_bar(df, 9, h=20, l=12.0)   # touch 2
        # bar 10 default → outside
        df = _set_bar(df, 11, h=20, l=14.5)  # touch 3 (14.5 < 15)
        result = _mt().track(df)
        assert result["touch_count"].iloc[5] == 3

    def test_touch_at_exact_boundary_is_not_inside(self):
        """low == top → NOT inside (need low < top)."""
        df = self._base()
        df = _set_bar(df, 8, h=20, l=15.0)   # low=15 == top=15 → outside
        result = _mt().track(df)
        assert result["touch_count"].iloc[5] == 0

    def test_deep_wick_still_one_touch(self):
        """Single bar that wicks through entire gap is still 1 touch."""
        df = self._base()
        df = _set_bar(df, 8, h=20, l=5.0)
        result = _mt().track(df)
        assert result["touch_count"].iloc[5] == 1


# ------------------------------------------------------------------ #
# Touch counting — Bear FVG                                            #
# ------------------------------------------------------------------ #

class TestBearTouchCount:
    """Bear FVG [10, 15]. Inside = high > 10. Default bars have high=8 (outside)."""

    def _base(self):
        df = _blank_df(high=8.0, low=7.0, close=7.5)
        return _set_fvg(df, 5, "bear", top=15.0, bottom=10.0)

    def test_zero_touches_bear(self):
        df = self._base()  # all bars high=8 < 10 → never enters
        result = _mt().track(df)
        assert result["touch_count"].iloc[5] == 0

    def test_one_touch_bear(self):
        df = self._base()
        df = _set_bar(df, 8, h=11.0, l=7.0)  # high=11 > 10 → inside
        result = _mt().track(df)
        assert result["touch_count"].iloc[5] == 1

    def test_two_touches_bear(self):
        df = self._base()
        df = _set_bar(df, 8, h=11.0, l=7.0)   # touch 1
        # bar 9 default (high=8) → outside
        df = _set_bar(df, 10, h=12.0, l=7.0)  # touch 2
        result = _mt().track(df)
        assert result["touch_count"].iloc[5] == 2

    def test_bear_boundary_not_inside(self):
        """high == bottom → NOT inside (need high > bottom)."""
        df = self._base()
        df = _set_bar(df, 8, h=10.0, l=7.0)   # high=10 == bottom=10 → outside
        result = _mt().track(df)
        assert result["touch_count"].iloc[5] == 0


# ------------------------------------------------------------------ #
# Mode-dependent tradeability                                          #
# ------------------------------------------------------------------ #

class TestModeTradeability:
    """Bull FVG [10, 15] with various touch counts."""

    def _base_with_touches(self, n_touches):
        """Create a DF where the FVG gets exactly n_touches distinct touches."""
        # Need enough bars: fvg at 3, then pairs of (inside, outside)
        n_bars = 3 + 1 + n_touches * 2 + 2  # some padding
        df = _blank_df(n=n_bars)
        df = _set_fvg(df, 3, "bull", top=15.0, bottom=10.0)
        for t in range(n_touches):
            inside_bar = 5 + t * 2   # 5, 7, 9, 11, ...
            # outside bar is inside_bar + 1 (default low=29 → outside)
            df = _set_bar(df, inside_bar, h=20, l=14.0)
        return df

    def test_backtest_0_touches_tradeable(self):
        df = self._base_with_touches(0)
        result = _mt(mode="backtest").track(df)
        assert result["touch_tradeable"].iloc[3] == True
        assert result["touch_max_allowed"].iloc[3] == 3

    def test_backtest_3_touches_tradeable(self):
        df = self._base_with_touches(3)
        result = _mt(mode="backtest").track(df)
        assert result["touch_tradeable"].iloc[3] == True

    def test_backtest_4_touches_not_tradeable(self):
        df = self._base_with_touches(4)
        result = _mt(mode="backtest").track(df)
        assert result["touch_tradeable"].iloc[3] == False

    def test_paper_2_touches_tradeable(self):
        df = self._base_with_touches(2)
        result = _mt(mode="paper").track(df)
        assert result["touch_tradeable"].iloc[3] == True
        assert result["touch_max_allowed"].iloc[3] == 2

    def test_paper_3_touches_not_tradeable(self):
        df = self._base_with_touches(3)
        result = _mt(mode="paper").track(df)
        assert result["touch_tradeable"].iloc[3] == False

    def test_live_1_touch_tradeable(self):
        df = self._base_with_touches(1)
        result = _mt(mode="live").track(df)
        assert result["touch_tradeable"].iloc[3] == True
        assert result["touch_max_allowed"].iloc[3] == 1

    def test_live_2_touches_not_tradeable(self):
        df = self._base_with_touches(2)
        result = _mt(mode="live").track(df)
        assert result["touch_tradeable"].iloc[3] == False

    def test_custom_max_touches_override(self):
        df = self._base_with_touches(5)
        result = _mt(mode="live", max_touches=5).track(df)
        assert result["touch_tradeable"].iloc[3] == True
        assert result["touch_max_allowed"].iloc[3] == 5

    def test_custom_max_6_not_tradeable(self):
        df = self._base_with_touches(5)
        result = _mt(mode="live", max_touches=4).track(df)
        assert result["touch_tradeable"].iloc[3] == False


# ------------------------------------------------------------------ #
# Mitigation state override                                            #
# ------------------------------------------------------------------ #

class TestMitigationOverride:
    """If Phase 2.6 mitigation_state is full/invalidated → not tradeable."""

    def _base(self):
        df = _blank_df()
        df = _set_fvg(df, 5, "bull", top=15.0, bottom=10.0)
        return df

    def test_full_overrides_to_not_tradeable(self):
        df = self._base()
        df = _add_mitigation(df, 5, "full")
        result = _mt(mode="backtest").track(df)
        assert result["touch_tradeable"].iloc[5] == False

    def test_invalidated_overrides_to_not_tradeable(self):
        df = self._base()
        df = _add_mitigation(df, 5, "invalidated")
        result = _mt(mode="backtest").track(df)
        assert result["touch_tradeable"].iloc[5] == False

    def test_fresh_remains_tradeable(self):
        df = self._base()
        df = _add_mitigation(df, 5, "fresh")
        result = _mt(mode="backtest").track(df)
        assert result["touch_tradeable"].iloc[5] == True

    def test_tapped_remains_tradeable(self):
        df = self._base()
        df = _add_mitigation(df, 5, "tapped")
        df = _set_bar(df, 8, h=20, l=14.0)  # 1 touch
        result = _mt(mode="backtest").track(df)
        assert result["touch_tradeable"].iloc[5] == True

    def test_deep_with_low_touches_still_tradeable(self):
        df = self._base()
        df = _add_mitigation(df, 5, "deep")
        df = _set_bar(df, 8, h=20, l=11.0)  # 1 touch (deep fill)
        result = _mt(mode="backtest").track(df)
        assert result["touch_tradeable"].iloc[5] == True

    def test_no_mitigation_column_works(self):
        """Module works without Phase 2.6 columns present."""
        df = self._base()
        # No mitigation_state column
        result = _mt(mode="backtest").track(df)
        assert result["touch_tradeable"].iloc[5] == True
        assert result["touch_count"].iloc[5] == 0


# ------------------------------------------------------------------ #
# No-FVG bars                                                          #
# ------------------------------------------------------------------ #

class TestNoFVG:
    def test_no_fvg_means_defaults(self):
        df = _blank_df()
        result = _mt().track(df)
        for i in range(len(df)):
            assert result["touch_count"].iloc[i] == -1
            assert result["touch_tradeable"].iloc[i] is None
            assert result["touch_max_allowed"].iloc[i] == -1

    def test_only_fvg_bar_gets_counts(self):
        df = _blank_df()
        df = _set_fvg(df, 5, "bull", top=15.0, bottom=10.0)
        result = _mt().track(df)
        assert result["touch_count"].iloc[5] >= 0
        for i in range(len(df)):
            if i != 5:
                assert result["touch_count"].iloc[i] == -1


# ------------------------------------------------------------------ #
# Multiple FVGs                                                        #
# ------------------------------------------------------------------ #

class TestMultipleFVGs:
    def test_independent_touch_counts(self):
        """Two FVGs at different price levels get independent touch counts."""
        df = _blank_df(n=30, high=100.0, low=99.0, close=99.5)
        # FVG1 at bar 3: [10, 15] — 1 touch
        df = _set_fvg(df, 3, "bull", top=15.0, bottom=10.0)
        df = _set_bar(df, 6, h=100, l=14.0)  # touches FVG1 only

        # FVG2 at bar 10: [50, 55] — 2 touches
        df = _set_fvg(df, 10, "bull", top=55.0, bottom=50.0)
        df = _set_bar(df, 13, h=100, l=54.0)  # touches FVG2 only
        # bar 14 default (low=99 > 55) → outside
        df = _set_bar(df, 15, h=100, l=53.0)  # touches FVG2 again

        result = _mt(mode="backtest").track(df)
        assert result["touch_count"].iloc[3] == 1
        assert result["touch_count"].iloc[10] == 2

    def test_fill_bar_touching_both_fvgs(self):
        """A single bar that enters BOTH FVGs gives each a touch."""
        df = _blank_df(n=20, high=100.0, low=99.0, close=99.5)
        # FVG1 at bar 3: [10, 15]
        df = _set_fvg(df, 3, "bull", top=15.0, bottom=10.0)
        # FVG2 at bar 5: [20, 25]
        df = _set_fvg(df, 5, "bull", top=25.0, bottom=20.0)
        # Bar 8: low=12 → enters both [10,15] and [20,25]
        df = _set_bar(df, 8, h=100, l=12.0)

        result = _mt().track(df)
        assert result["touch_count"].iloc[3] == 1
        assert result["touch_count"].iloc[5] == 1


# ------------------------------------------------------------------ #
# FVG at edge positions                                                #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_fvg_at_last_bar_zero_touches(self):
        df = _blank_df()
        df = _set_fvg(df, len(df) - 1, "bull", top=15.0, bottom=10.0)
        result = _mt().track(df)
        assert result["touch_count"].iloc[-1] == 0
        assert result["touch_tradeable"].iloc[-1] == True

    def test_fvg_at_second_to_last_one_possible_bar(self):
        df = _blank_df()
        pos = len(df) - 2
        df = _set_fvg(df, pos, "bull", top=15.0, bottom=10.0)
        df = _set_bar(df, pos + 1, h=20, l=14.0)  # 1 touch
        result = _mt().track(df)
        assert result["touch_count"].iloc[pos] == 1


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _blank_df()
        result = _mt().track(df)
        for col in ["touch_count", "touch_tradeable", "touch_max_allowed"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _blank_df()
        pd.testing.assert_index_equal(_mt().track(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _blank_df()
        _mt().track(df)
        assert "touch_count" not in df.columns


# ------------------------------------------------------------------ #
# Validation                                                            #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            MultiTouchLifecycle(mode="invalid")

    def test_zero_max_touches_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            MultiTouchLifecycle(max_touches=0)

    def test_negative_max_touches_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            MultiTouchLifecycle(max_touches=-1)

    def test_missing_columns_raises(self):
        df = _blank_df().drop(columns=["fvg_top"])
        with pytest.raises(ValueError, match="Missing"):
            _mt().track(df)

    def test_non_datetime_index_raises(self):
        df = _blank_df()
        df.index = range(len(df))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _mt().track(df)

    def test_convenience_function_works(self):
        df = _blank_df()
        result = track_touches(df)
        assert "touch_count" in result.columns

    def test_convenience_function_with_mode(self):
        df = _blank_df()
        result = track_touches(df, mode="live")
        # live mode max_touches=1
        assert result["touch_max_allowed"].iloc[0] == -1  # no FVG → -1


# ------------------------------------------------------------------ #
# Accessor                                                             #
# ------------------------------------------------------------------ #

class TestAccessor:
    def test_get_tradeable_fvgs_excludes_exhausted(self):
        """4 touches in backtest mode (max=3) → not tradeable → excluded."""
        df = _blank_df(n=20)
        df = _set_fvg(df, 3, "bull", top=15.0, bottom=10.0)
        # 4 touches
        for i, bar in enumerate([5, 7, 9, 11]):
            df = _set_bar(df, bar, h=20, l=14.0)
        result = _mt(mode="backtest").track(df)
        tradeable = _mt(mode="backtest").get_tradeable_fvgs(result)
        assert len(tradeable) == 0

    def test_get_tradeable_fvgs_includes_fresh(self):
        df = _blank_df()
        df = _set_fvg(df, 5, "bull", top=15.0, bottom=10.0)
        result = _mt(mode="backtest").track(df)
        tradeable = _mt(mode="backtest").get_tradeable_fvgs(result)
        assert len(tradeable) == 1
        assert tradeable[0]["touch_count"] == 0

    def test_get_tradeable_newest_first(self):
        df = _blank_df(n=30, high=100.0, low=99.0, close=99.5)
        df = _set_fvg(df, 5, "bull", top=15.0, bottom=10.0)
        df = _set_fvg(df, 15, "bull", top=25.0, bottom=20.0)
        result = _mt().track(df)
        tradeable = _mt().get_tradeable_fvgs(result, n=10)
        assert len(tradeable) == 2
        assert tradeable[0]["confirm_ts"] >= tradeable[1]["confirm_ts"]

    def test_get_tradeable_empty_when_none(self):
        df = _blank_df()
        result = _mt().track(df)
        assert _mt().get_tradeable_fvgs(result) == []

    def test_get_tradeable_includes_mitigation_state_if_present(self):
        df = _blank_df()
        df = _set_fvg(df, 5, "bull", top=15.0, bottom=10.0)
        df = _add_mitigation(df, 5, "fresh")
        result = _mt().track(df)
        tradeable = _mt().get_tradeable_fvgs(result)
        assert len(tradeable) == 1
        assert tradeable[0]["mitigation_state"] == "fresh"


# ------------------------------------------------------------------ #
# Mode defaults                                                        #
# ------------------------------------------------------------------ #

class TestModeDefaults:
    def test_backtest_default_is_3(self):
        assert _mt(mode="backtest").max_touches == 3

    def test_paper_default_is_2(self):
        assert _mt(mode="paper").max_touches == 2

    def test_live_default_is_1(self):
        assert _mt(mode="live").max_touches == 1

    def test_mode_case_insensitive(self):
        assert _mt(mode="BACKTEST").mode == "backtest"
        assert _mt(mode="Paper").mode == "paper"
        assert _mt(mode="LIVE").mode == "live"
