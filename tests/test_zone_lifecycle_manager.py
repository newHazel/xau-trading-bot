"""
Tests for zone_lifecycle_manager.py — Phase 2.10.

Critical properties:
  - Each FVG and OB gets a unique zone_id.
  - Status lifecycle: active → tested → mitigated → expired → invalidated.
  - Expiry: age > max_age_bars → expired (unless already mitigated/invalidated).
  - Invalidation is terminal and overrides everything.
  - Works with FVG only, OB only, or both.
  - Integrates Phase 2.6 mitigation_state and Phase 2.7 touch_tradeable when present.
  - No-zone bars get zone_id=None.
"""

import pytest
import numpy as np
import pandas as pd

from core.smc.zone_lifecycle_manager import (
    ZoneLifecycleManager,
    track_zones,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _blank_df(n=20, start="2026-01-05 10:00", freq="5min",
              high=100.0, low=99.0, close=99.5):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open":   [99.5]  * n,
            "high":   [high]  * n,
            "low":    [low]   * n,
            "close":  [close] * n,
            "volume": [100.0] * n,
        },
        index=idx,
    )


def _add_fvg(df, pos, fvg_type, top, bottom):
    df = df.copy()
    if "fvg_type" not in df.columns:
        n = len(df)
        df["fvg_type"]   = pd.Series([None] * n, dtype=object, index=df.index)
        df["fvg_top"]    = np.nan
        df["fvg_bottom"] = np.nan
    df.iloc[pos, df.columns.get_loc("fvg_type")]   = fvg_type
    df.iloc[pos, df.columns.get_loc("fvg_top")]    = float(top)
    df.iloc[pos, df.columns.get_loc("fvg_bottom")] = float(bottom)
    return df


def _add_ob(df, pos, ob_type, top, bottom):
    df = df.copy()
    if "ob_type" not in df.columns:
        n = len(df)
        df["ob_type"]   = pd.Series([None] * n, dtype=object, index=df.index)
        df["ob_top"]    = np.nan
        df["ob_bottom"] = np.nan
    df.iloc[pos, df.columns.get_loc("ob_type")]   = ob_type
    df.iloc[pos, df.columns.get_loc("ob_top")]    = float(top)
    df.iloc[pos, df.columns.get_loc("ob_bottom")] = float(bottom)
    return df


def _add_mitigation(df, pos, state):
    df = df.copy()
    if "mitigation_state" not in df.columns:
        df["mitigation_state"] = pd.Series([None] * len(df), dtype=object, index=df.index)
    df.iloc[pos, df.columns.get_loc("mitigation_state")] = state
    return df


def _add_touch(df, pos, tradeable):
    df = df.copy()
    if "touch_tradeable" not in df.columns:
        df["touch_tradeable"] = pd.Series([None] * len(df), dtype=object, index=df.index)
    df.iloc[pos, df.columns.get_loc("touch_tradeable")] = tradeable
    return df


def _set_bar(df, pos, h=None, l=None, c=None):
    df = df.copy()
    if h is not None:
        df.iloc[pos, df.columns.get_loc("high")]  = float(h)
    if l is not None:
        df.iloc[pos, df.columns.get_loc("low")]   = float(l)
    if c is not None:
        df.iloc[pos, df.columns.get_loc("close")] = float(c)
    return df


def _zm(**kwargs):
    return ZoneLifecycleManager(**kwargs)


# ------------------------------------------------------------------ #
# Zone ID assignment                                                   #
# ------------------------------------------------------------------ #

class TestZoneID:
    def test_fvg_gets_zone_id(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        result = _zm().track(df)
        zid = result["zone_id"].iloc[5]
        assert zid is not None
        assert zid.startswith("ZN-FVG-")

    def test_ob_gets_zone_id(self):
        df = _blank_df()
        df = _add_ob(df, 5, "bull", 105, 100)
        result = _zm().track(df)
        zid = result["zone_id"].iloc[5]
        assert zid is not None
        assert zid.startswith("ZN-OB-")

    def test_ids_are_unique(self):
        df = _blank_df(n=30)
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _add_fvg(df, 10, "bear", 110, 105)
        df = _add_ob(df, 15, "bull", 102, 98)
        result = _zm().track(df)
        ids = result["zone_id"].dropna().tolist()
        assert len(ids) == 3
        assert len(set(ids)) == 3  # all unique

    def test_id_format(self):
        df = _blank_df(start="2026-01-05 10:30")
        df = _add_fvg(df, 0, "bull", 105, 100)
        result = _zm().track(df)
        zid = result["zone_id"].iloc[0]
        # ZN-FVG-20260105-1030-001
        assert "20260105" in zid
        assert "1030" in zid

    def test_no_zone_bars_have_none_id(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        result = _zm().track(df)
        for i in range(len(df)):
            if i != 5:
                assert result["zone_id"].iloc[i] is None


# ------------------------------------------------------------------ #
# FVG zone status with mitigation data                                 #
# ------------------------------------------------------------------ #

class TestFVGStatusWithMitigation:
    def test_fresh_fvg_is_active(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _add_mitigation(df, 5, "fresh")
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "active"

    def test_tapped_fvg_is_tested(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _add_mitigation(df, 5, "tapped")
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "tested"

    def test_partial_fvg_is_tested(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _add_mitigation(df, 5, "partial")
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "tested"

    def test_deep_fvg_is_mitigated(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _add_mitigation(df, 5, "deep")
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "mitigated"

    def test_full_fvg_is_mitigated(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _add_mitigation(df, 5, "full")
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "mitigated"

    def test_invalidated_fvg_is_invalidated(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _add_mitigation(df, 5, "invalidated")
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "invalidated"


# ------------------------------------------------------------------ #
# FVG zone status without mitigation data                              #
# ------------------------------------------------------------------ #

class TestFVGStatusWithoutMitigation:
    def test_active_when_no_close_through(self):
        """Price never closes below bull FVG bottom → active."""
        df = _blank_df()  # close=99.5, all above bottom=90
        df = _add_fvg(df, 5, "bull", 105, 90)
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "active"

    def test_invalidated_when_close_below_bull_bottom(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _set_bar(df, 10, c=98.0)  # close below bottom=100
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "invalidated"

    def test_invalidated_when_close_above_bear_top(self):
        df = _blank_df(high=90, low=85, close=87)
        df = _add_fvg(df, 5, "bear", 95, 90)
        df = _set_bar(df, 10, c=96.0)  # close above top=95
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "invalidated"


# ------------------------------------------------------------------ #
# FVG expiry                                                           #
# ------------------------------------------------------------------ #

class TestFVGExpiry:
    def test_expired_when_age_exceeds_max(self):
        df = _blank_df(n=30)
        df = _add_fvg(df, 5, "bull", 105, 90)  # age = 30-1-5 = 24
        df = _add_mitigation(df, 5, "fresh")
        result = _zm(fvg_max_age_bars=20).track(df)
        assert result["zone_status"].iloc[5] == "expired"
        assert result["zone_expiry_bar"].iloc[5] == 25  # 5 + 20

    def test_not_expired_within_max_age(self):
        df = _blank_df(n=20)
        df = _add_fvg(df, 5, "bull", 105, 90)  # age = 19-5 = 14
        df = _add_mitigation(df, 5, "fresh")
        result = _zm(fvg_max_age_bars=20).track(df)
        assert result["zone_status"].iloc[5] == "active"

    def test_invalidation_overrides_expiry(self):
        """Invalidated even if also expired."""
        df = _blank_df(n=30)
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _add_mitigation(df, 5, "invalidated")
        result = _zm(fvg_max_age_bars=10).track(df)
        assert result["zone_status"].iloc[5] == "invalidated"

    def test_mitigated_overrides_expiry(self):
        df = _blank_df(n=30)
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _add_mitigation(df, 5, "full")
        result = _zm(fvg_max_age_bars=10).track(df)
        assert result["zone_status"].iloc[5] == "mitigated"


# ------------------------------------------------------------------ #
# Touch tradeable integration                                          #
# ------------------------------------------------------------------ #

class TestTouchIntegration:
    def test_tapped_but_touch_exhausted_is_mitigated(self):
        """mitigation=tapped but touch_tradeable=False → mitigated."""
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _add_mitigation(df, 5, "tapped")
        df = _add_touch(df, 5, False)
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "mitigated"

    def test_tapped_and_touch_tradeable_is_tested(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _add_mitigation(df, 5, "tapped")
        df = _add_touch(df, 5, True)
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "tested"


# ------------------------------------------------------------------ #
# OB zone status                                                       #
# ------------------------------------------------------------------ #

class TestOBStatus:
    def test_ob_active_when_untouched(self):
        """Bull OB [98, 103]. Price stays above (low=99 > nothing enters)."""
        df = _blank_df()  # low=99
        df = _add_ob(df, 5, "bull", 103, 98)
        # Default bars: low=99, which is <= top=103 → enters zone
        # Need low > top=103 to stay out
        df2 = _blank_df(low=104, high=106, close=105)
        df2 = _add_ob(df2, 5, "bull", 103, 98)
        result = _zm().track(df2)
        assert result["zone_status"].iloc[5] == "active"

    def test_ob_tested_when_price_enters(self):
        """Price dips into bull OB zone but doesn't close below."""
        df = _blank_df(low=104, high=106, close=105)
        df = _add_ob(df, 5, "bull", 103, 98)
        df = _set_bar(df, 10, l=102.0)  # low=102 <= top=103 → enters zone
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "tested"

    def test_ob_invalidated_when_close_below(self):
        """Bull OB: close below bottom → invalidated."""
        df = _blank_df(low=104, high=106, close=105)
        df = _add_ob(df, 5, "bull", 103, 98)
        df = _set_bar(df, 10, l=95, c=97)  # close=97 < bottom=98
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "invalidated"

    def test_bear_ob_tested(self):
        """Bear OB [98, 103]. Price rises into zone."""
        df = _blank_df(high=96, low=94, close=95)
        df = _add_ob(df, 5, "bear", 103, 98)
        df = _set_bar(df, 10, h=99)  # high=99 >= bottom=98 → enters
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "tested"

    def test_bear_ob_invalidated(self):
        """Bear OB: close above top → invalidated."""
        df = _blank_df(high=96, low=94, close=95)
        df = _add_ob(df, 5, "bear", 103, 98)
        df = _set_bar(df, 10, h=105, c=104)  # close=104 > top=103
        result = _zm().track(df)
        assert result["zone_status"].iloc[5] == "invalidated"


# ------------------------------------------------------------------ #
# OB expiry                                                            #
# ------------------------------------------------------------------ #

class TestOBExpiry:
    def test_ob_expired(self):
        df = _blank_df(n=30, low=104, high=106, close=105)
        df = _add_ob(df, 3, "bull", 103, 98)  # age = 29-3 = 26
        result = _zm(ob_max_age_bars=20).track(df)
        assert result["zone_status"].iloc[3] == "expired"
        assert result["zone_expiry_bar"].iloc[3] == 23  # 3 + 20

    def test_ob_not_expired_within_max(self):
        df = _blank_df(n=20, low=104, high=106, close=105)
        df = _add_ob(df, 3, "bull", 103, 98)  # age = 19-3 = 16
        result = _zm(ob_max_age_bars=20).track(df)
        assert result["zone_status"].iloc[3] == "active"


# ------------------------------------------------------------------ #
# Both FVG and OB zones                                                #
# ------------------------------------------------------------------ #

class TestBothZoneTypes:
    def test_fvg_and_ob_coexist(self):
        df = _blank_df(n=30)
        df = _add_fvg(df, 5, "bull", 105, 90)
        df = _add_ob(df, 10, "bear", 110, 105)
        result = _zm().track(df)
        assert result["zone_type"].iloc[5] == "fvg"
        assert result["zone_type"].iloc[10] == "ob"
        assert result["zone_id"].iloc[5] != result["zone_id"].iloc[10]

    def test_fvg_and_ob_at_same_bar(self):
        """FVG and OB at the same bar get separate zone entries.
        Note: FVG is processed first, so it takes the bar. OB overwrites."""
        df = _blank_df(n=20, low=104, high=106, close=105)
        df = _add_fvg(df, 5, "bull", 105, 90)
        df = _add_ob(df, 5, "bear", 110, 105)
        result = _zm().track(df)
        # OB processed after FVG, overwrites the bar
        assert result["zone_type"].iloc[5] == "ob"


# ------------------------------------------------------------------ #
# Zone metadata                                                        #
# ------------------------------------------------------------------ #

class TestZoneMetadata:
    def test_zone_direction(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _add_fvg(df, 10, "bear", 110, 105)
        result = _zm().track(df)
        assert result["zone_direction"].iloc[5] == "bull"
        assert result["zone_direction"].iloc[10] == "bear"

    def test_zone_top_bottom(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105.5, 100.3)
        result = _zm().track(df)
        assert result["zone_top"].iloc[5] == pytest.approx(105.5)
        assert result["zone_bottom"].iloc[5] == pytest.approx(100.3)

    def test_zone_age_bars(self):
        df = _blank_df(n=20)
        df = _add_fvg(df, 5, "bull", 105, 90)
        result = _zm().track(df)
        assert result["zone_age_bars"].iloc[5] == 14  # 19 - 5


# ------------------------------------------------------------------ #
# Output format                                                        #
# ------------------------------------------------------------------ #

class TestOutputFormat:
    def test_output_has_required_columns(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        result = _zm().track(df)
        for col in ["zone_id", "zone_type", "zone_direction", "zone_top",
                     "zone_bottom", "zone_status", "zone_age_bars", "zone_expiry_bar"]:
            assert col in result.columns

    def test_output_index_unchanged(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        pd.testing.assert_index_equal(_zm().track(df).index, df.index)

    def test_output_is_copy_not_inplace(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        _zm().track(df)
        assert "zone_id" not in df.columns

    def test_defaults_when_no_zone(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        result = _zm().track(df)
        assert result["zone_age_bars"].iloc[0] == -1
        assert result["zone_expiry_bar"].iloc[0] == -1
        assert np.isnan(result["zone_top"].iloc[0])


# ------------------------------------------------------------------ #
# Validation                                                            #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_no_zone_source_raises(self):
        df = _blank_df()
        with pytest.raises(ValueError, match="zone source"):
            _zm().track(df)

    def test_missing_ohlc_raises(self):
        df = _blank_df().drop(columns=["close"])
        df = _add_fvg(df, 5, "bull", 105, 100)
        with pytest.raises(ValueError, match="Missing"):
            _zm().track(df)

    def test_non_datetime_index_raises(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        df.index = range(len(df))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _zm().track(df)

    def test_invalid_fvg_max_age_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            ZoneLifecycleManager(fvg_max_age_bars=0)

    def test_invalid_ob_max_age_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            ZoneLifecycleManager(ob_max_age_bars=-1)

    def test_convenience_function_works(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        result = track_zones(df)
        assert "zone_id" in result.columns


# ------------------------------------------------------------------ #
# Accessor                                                             #
# ------------------------------------------------------------------ #

class TestAccessor:
    def test_get_active_zones(self):
        df = _blank_df(n=30)
        df = _add_fvg(df, 5, "bull", 105, 90)
        df = _add_mitigation(df, 5, "fresh")
        df = _add_fvg(df, 10, "bear", 95, 90)
        df = _add_mitigation(df, 10, "invalidated")
        result = _zm().track(df)
        active = _zm().get_active_zones(result)
        assert len(active) == 1
        assert active[0]["status"] == "active"

    def test_filter_by_zone_type(self):
        df = _blank_df(n=30, low=104, high=106, close=105)
        df = _add_fvg(df, 5, "bull", 105, 90)
        df = _add_mitigation(df, 5, "fresh")
        df = _add_ob(df, 10, "bull", 103, 98)
        result = _zm().track(df)
        fvg_zones = _zm().get_active_zones(result, zone_type="fvg")
        ob_zones  = _zm().get_active_zones(result, zone_type="ob")
        assert all(z["zone_type"] == "fvg" for z in fvg_zones)
        assert all(z["zone_type"] == "ob" for z in ob_zones)

    def test_filter_by_direction(self):
        df = _blank_df(n=30)
        df = _add_fvg(df, 5, "bull", 105, 90)
        df = _add_mitigation(df, 5, "fresh")
        df = _add_fvg(df, 10, "bear", 110, 105)
        df = _add_mitigation(df, 10, "fresh")
        result = _zm().track(df)
        bull = _zm().get_active_zones(result, direction="bull")
        bear = _zm().get_active_zones(result, direction="bear")
        assert all(z["direction"] == "bull" for z in bull)
        assert all(z["direction"] == "bear" for z in bear)

    def test_newest_first(self):
        df = _blank_df(n=30)
        df = _add_fvg(df, 5, "bull", 105, 90)
        df = _add_mitigation(df, 5, "fresh")
        df = _add_fvg(df, 15, "bull", 106, 91)
        df = _add_mitigation(df, 15, "fresh")
        result = _zm().track(df)
        active = _zm().get_active_zones(result)
        assert len(active) == 2
        assert active[0]["timestamp"] >= active[1]["timestamp"]

    def test_empty_when_no_active(self):
        df = _blank_df()
        df = _add_fvg(df, 5, "bull", 105, 100)
        df = _add_mitigation(df, 5, "invalidated")
        result = _zm().track(df)
        assert _zm().get_active_zones(result) == []
