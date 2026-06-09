"""
Tests for mitigation_tracker.py — Phase 2.6.

Critical properties:
  - State reflects the DEEPEST fill reached over the FVG's life.
  - Bull FVG fill = (top − low) / gap; invalidated if close < bottom.
  - Bear FVG fill = (high − bottom) / gap; invalidated if close > top.
  - Invalidation is terminal and overrides any prior fill.
  - State bands (default): tapped (0,0.25], partial (0.25,0.50],
    deep (0.50,1.0), full ≥1.0.
  - FVG-free bars produce mitigation_state=None.
"""

import pytest
import numpy as np
import pandas as pd

from core.smc.mitigation_tracker import MitigationTracker, track_mitigation


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _blank_df(n=20, start="2026-01-05 10:00", freq="5min",
              high=30.0, low=29.0, close=29.5):
    """Default OHLC sits well ABOVE a typical bull gap [10,15] so bars don't
    accidentally mitigate unless we set them to."""
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


def _set_bar(df, pos, h, l, c):
    df = df.copy()
    df.iloc[pos, df.columns.get_loc("high")]  = float(h)
    df.iloc[pos, df.columns.get_loc("low")]   = float(l)
    df.iloc[pos, df.columns.get_loc("close")] = float(c)
    return df


def _mt(**kwargs):
    return MitigationTracker(**kwargs)


# ------------------------------------------------------------------ #
# Bullish FVG mitigation states                                        #
# ------------------------------------------------------------------ #

class TestBullishStates:
    # Gap [10, 15], gap size = 5. fill = (15 - low) / 5.
    def _base(self):
        # FVG at bar 5; all later bars default high=30, low=29 (well above the gap)
        df = _blank_df()
        return _set_fvg(df, 5, "bull", top=15.0, bottom=10.0)

    def test_fresh_when_price_never_returns(self):
        df = self._base()  # all bars stay at low=29 (> 15) → never enters
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "fresh"
        assert result["mitigation_max_fill_pct"].iloc[5] == pytest.approx(0.0)

    def test_tapped_shallow_entry(self):
        df = self._base()
        df = _set_bar(df, 8, h=20, l=14.0, c=20)   # fill = (15-14)/5 = 0.2 ≤ 0.25
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "tapped"
        assert result["mitigation_max_fill_pct"].iloc[5] == pytest.approx(0.2)

    def test_partial_entry(self):
        df = self._base()
        df = _set_bar(df, 8, h=20, l=13.0, c=20)   # fill = (15-13)/5 = 0.4
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "partial"

    def test_deep_past_midpoint(self):
        df = self._base()
        df = _set_bar(df, 8, h=20, l=11.0, c=20)   # fill = (15-11)/5 = 0.8
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "deep"

    def test_full_fill_to_bottom(self):
        df = self._base()
        df = _set_bar(df, 8, h=20, l=10.0, c=20)   # fill = (15-10)/5 = 1.0
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "full"
        assert result["mitigation_max_fill_pct"].iloc[5] == pytest.approx(1.0)

    def test_full_when_wick_below_but_close_holds(self):
        df = self._base()
        df = _set_bar(df, 8, h=20, l=8.0, c=12.0)  # wick below 10 but close=12 ≥ 10
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "full"
        assert result["mitigation_max_fill_pct"].iloc[5] == pytest.approx(1.0)

    def test_invalidated_when_close_below_bottom(self):
        df = self._base()
        df = _set_bar(df, 8, h=20, l=8.0, c=9.0)   # close=9 < bottom=10
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "invalidated"
        assert result["mitigation_invalidated_bar"].iloc[5] == 8

    def test_deepest_fill_wins(self):
        df = self._base()
        df = _set_bar(df, 7, h=20, l=14.0, c=20)   # tapped (0.2)
        df = _set_bar(df, 9, h=20, l=11.0, c=20)   # deep (0.8) — deeper
        df = _set_bar(df, 11, h=20, l=14.5, c=20)  # shallow again — doesn't reduce state
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "deep"

    def test_first_touch_bar_recorded(self):
        df = self._base()
        df = _set_bar(df, 9, h=20, l=13.0, c=20)
        result = _mt().track(df)
        assert result["mitigation_first_touch_bar"].iloc[5] == 9

    def test_invalidation_is_terminal_overrides_prior_fill(self):
        df = self._base()
        df = _set_bar(df, 7, h=20, l=11.0, c=20)   # deep first
        df = _set_bar(df, 9, h=20, l=8.0, c=9.0)   # then invalidate
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "invalidated"


# ------------------------------------------------------------------ #
# Bearish FVG mitigation states                                        #
# ------------------------------------------------------------------ #

class TestBearishStates:
    # Gap [10, 15]; bear FVG sits ABOVE price. fill = (high - 10) / 5.
    def _base(self):
        # Default bars sit BELOW the gap (high=30 would be above — need to lower)
        df = _blank_df(high=8.0, low=7.0, close=7.5)   # well below gap bottom=10
        return _set_fvg(df, 5, "bear", top=15.0, bottom=10.0)

    def test_fresh_when_price_never_rises_into_gap(self):
        df = self._base()
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "fresh"

    def test_tapped_bear(self):
        df = self._base()
        df = _set_bar(df, 8, h=11.0, l=7.0, c=7.5)   # fill = (11-10)/5 = 0.2
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "tapped"

    def test_full_bear(self):
        df = self._base()
        df = _set_bar(df, 8, h=15.0, l=7.0, c=7.5)   # fill = (15-10)/5 = 1.0
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "full"

    def test_invalidated_bear_when_close_above_top(self):
        df = self._base()
        df = _set_bar(df, 8, h=16.0, l=7.0, c=15.5)  # close=15.5 > top=15
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "invalidated"


# ------------------------------------------------------------------ #
# No-FVG bars                                                          #
# ------------------------------------------------------------------ #

class TestNoFVG:
    def test_no_fvg_means_state_none(self):
        df = _blank_df()
        result = _mt().track(df)
        for i in range(len(df)):
            assert result["mitigation_state"].iloc[i] is None

    def test_only_fvg_bar_gets_state(self):
        df = _blank_df()
        df = _set_fvg(df, 5, "bull", top=15.0, bottom=10.0)
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[5] == "fresh"
        for i in range(len(df)):
            if i != 5:
                assert result["mitigation_state"].iloc[i] is None

    def test_fvg_at_last_bar_is_fresh(self):
        df = _blank_df()
        df = _set_fvg(df, len(df) - 1, "bull", top=15.0, bottom=10.0)
        result = _mt().track(df)
        assert result["mitigation_state"].iloc[-1] == "fresh"


# ------------------------------------------------------------------ #
# Accessor                                                             #
# ------------------------------------------------------------------ #

class TestAccessors:
    def test_get_unmitigated_excludes_full_and_invalidated(self):
        # Distinct gap levels so each fill bar only affects its own FVG.
        # Price sits at ~100; fill bars dip only as low as needed (≥ 15 so they
        # never touch FVG1's [10,15]).
        df = _blank_df(n=40, high=100.0, low=99.0, close=99.5)
        # FVG 1 (bar 5): gap [10,15] → stays fresh (no bar dips below 15)
        df = _set_fvg(df, 5, "bull", top=15.0, bottom=10.0)
        # FVG 2 (bar 15): gap [20,25] → fully filled at bar 18 (low=20)
        df = _set_fvg(df, 15, "bull", top=25.0, bottom=20.0)
        df = _set_bar(df, 18, h=100, l=20.0, c=100)
        # FVG 3 (bar 25): gap [30,35] → invalidated at bar 28 (close=29 < 30)
        df = _set_fvg(df, 25, "bull", top=35.0, bottom=30.0)
        df = _set_bar(df, 28, h=100, l=31.0, c=29.0)
        result = _mt().track(df)
        unmit = _mt().get_unmitigated_fvgs(result, n=10)
        states = {u["state"] for u in unmit}
        assert "full" not in states
        assert "invalidated" not in states
        # FVG1 (fresh) should be present
        assert any(u["state"] == "fresh" for u in unmit)

    def test_get_unmitigated_newest_first(self):
        df = _blank_df(n=40)
        df = _set_fvg(df, 5, "bull", top=15.0, bottom=10.0)
        df = _set_fvg(df, 15, "bull", top=15.0, bottom=10.0)
        result = _mt().track(df)
        unmit = _mt().get_unmitigated_fvgs(result, n=10)
        if len(unmit) > 1:
            assert unmit[0]["confirm_ts"] >= unmit[1]["confirm_ts"]

    def test_get_unmitigated_empty_when_none(self):
        df = _blank_df()
        result = _mt().track(df)
        assert _mt().get_unmitigated_fvgs(result) == []


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _blank_df()
        result = _mt().track(df)
        for col in ["mitigation_state", "mitigation_first_touch_bar",
                    "mitigation_max_fill_pct", "mitigation_invalidated_bar"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _blank_df()
        pd.testing.assert_index_equal(_mt().track(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _blank_df()
        _mt().track(df)
        assert "mitigation_state" not in df.columns

    def test_defaults_when_no_fvg(self):
        df = _blank_df()
        result = _mt().track(df)
        assert (result["mitigation_first_touch_bar"] == -1).all()
        assert (result["mitigation_invalidated_bar"] == -1).all()


# ------------------------------------------------------------------ #
# Validation                                                            #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_bad_threshold_order_raises(self):
        with pytest.raises(ValueError, match="tapped_max"):
            MitigationTracker(tapped_max=0.6, partial_max=0.5)

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError):
            MitigationTracker(tapped_max=0.25, partial_max=1.5)

    def test_missing_fvg_columns_raises(self):
        df = _blank_df().drop(columns=["fvg_top"])
        with pytest.raises(ValueError, match="Missing"):
            _mt().track(df)

    def test_missing_ohlc_raises(self):
        df = _blank_df().drop(columns=["close"])
        with pytest.raises(ValueError, match="Missing"):
            _mt().track(df)

    def test_non_datetime_index_raises(self):
        df = _blank_df()
        df.index = range(len(df))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _mt().track(df)

    def test_convenience_function_works(self):
        df = _blank_df()
        result = track_mitigation(df)
        assert "mitigation_state" in result.columns


# ------------------------------------------------------------------ #
# Custom thresholds                                                    #
# ------------------------------------------------------------------ #

class TestCustomThresholds:
    def test_custom_bands_shift_classification(self):
        # Gap [10,15]; low=13 → fill=0.4
        df = _blank_df()
        df = _set_fvg(df, 5, "bull", top=15.0, bottom=10.0)
        df = _set_bar(df, 8, h=20, l=13.0, c=20)
        # Default: 0.4 → partial (0.25,0.50]
        assert _mt().track(df)["mitigation_state"].iloc[5] == "partial"
        # With tapped_max=0.45 → 0.4 ≤ 0.45 → tapped
        assert _mt(tapped_max=0.45, partial_max=0.6).track(df)["mitigation_state"].iloc[5] == "tapped"
