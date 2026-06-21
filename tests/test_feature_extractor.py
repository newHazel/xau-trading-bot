"""Tests for Phase 12.1 feature extractor (core/ml/feature_extractor.py)."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from core.engine.signal_pipeline import PipelineSignal
from core.engine.rulebook_engine import (
    RulebookDecision, MANDATORY_CONDITIONS, OPTIONAL_CONDITIONS, INDICATOR_CONDITIONS,
)
from core.engine.signal_grader import GradeResult
from core.ml.feature_extractor import extract_features, feature_names


def _grade(core=14, ind=6, net_rr=2.5):
    return GradeResult(
        grade="B", score=core + ind, mandatory_passed=True, failed_mandatory=[],
        passed_optional=["ob_valid"], failed_optional=[], net_rr=net_rr, detail="",
        core_score=core, indicator_score=ind, passed_indicators=["vwap_aligned"],
    )


def _signal(direction="long", entry=100.0, sl=99.0, tp1=102.0, tp2=103.5,
            grade="B", score=20, indicators=True):
    mand = {c: True for c in MANDATORY_CONDITIONS}          # approved → all True
    opt = {c: (c == "ob_valid") for c in OPTIONAL_CONDITIONS}
    ind = {c: (c == "vwap_aligned") for c in INDICATOR_CONDITIONS} if indicators else None
    decision = RulebookDecision(
        approved=True, direction=direction, grade=_grade(), rejection=None,
        mandatory_results=mand, optional_results=opt, indicator_results=ind,
    )
    return PipelineSignal(
        setup_id="ETH-20260501-1200-LONG", direction=direction, entry=entry, sl=sl,
        tp1=tp1, tp2=tp2, lot_size=0.1, grade=grade, score=score,
        timestamp=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc), bar_index=10,
        approved=True, decision=decision,
    )


def _history(n=40, base=100.0):
    idx = pd.date_range("2026-05-01T08:00:00Z", periods=n, freq="5min", tz="UTC")
    closes = [base + (i % 5) * 0.3 for i in range(n)]
    df = pd.DataFrame({
        "open": closes,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
    }, index=idx)
    return {"5m": df}


def test_feature_names_are_stable_and_complete():
    names = feature_names()
    assert len(names) == len(set(names)), "duplicate feature columns"
    # 15 mandatory + 10 optional + 4 indicator booleans must all be present
    for c in MANDATORY_CONDITIONS:
        assert f"mand_{c}" in names
    for c in OPTIONAL_CONDITIONS:
        assert f"opt_{c}" in names
    for c in INDICATOR_CONDITIONS:
        assert f"ind_{c}" in names


def test_extract_returns_exactly_the_declared_columns():
    feats = extract_features(_signal(), history=_history(), config={})
    assert set(feats.keys()) == set(feature_names())


def test_core_fields_mapped_correctly():
    feats = extract_features(_signal(direction="long"), history=_history(), config={})
    assert feats["is_long"] == 1.0
    assert feats["grade_ordinal"] == 2.0          # "B"
    assert feats["score"] == 20.0
    assert feats["core_score"] == 14.0
    assert feats["indicator_score"] == 6.0
    assert feats["net_rr"] == 2.5
    assert feats["opt_ob_valid"] == 1.0
    assert feats["opt_dxy_aligned"] == 0.0
    assert feats["ind_vwap_aligned"] == 1.0
    # approved signal → every mandatory boolean is True
    assert all(feats[f"mand_{c}"] == 1.0 for c in MANDATORY_CONDITIONS)


def test_geometry_and_volatility():
    feats = extract_features(_signal(entry=100.0, sl=99.0, tp1=102.0, tp2=103.5),
                             history=_history(), config={})
    assert feats["sl_distance"] == 1.0
    assert math.isclose(feats["tp1_r"], 2.0)
    assert math.isclose(feats["tp2_r"], 3.5)
    assert math.isfinite(feats["atr"]) and feats["atr"] > 0
    assert math.isfinite(feats["sl_distance_atr"])
    assert 0.0 <= feats["rsi"] <= 100.0


def test_short_direction_geometry_sign():
    feats = extract_features(_signal(direction="short", entry=100.0, sl=101.0,
                                     tp1=98.0, tp2=96.5), history=_history(), config={})
    assert feats["is_long"] == 0.0
    assert math.isclose(feats["tp1_r"], 2.0)      # (entry-tp1)/risk = 2/1
    assert math.isclose(feats["tp2_r"], 3.5)


def test_sweep_and_fvg_enrichment():
    sweep = {"type": "EQL", "level": 99.0}
    fvg = {"top": 100.5, "bottom": 99.8, "state": "fresh",
           "confirm_ts": "2026-05-01T11:30:00+00:00"}
    feats = extract_features(_signal(), history=_history(), config={}, sweep=sweep, fvg=fvg)
    assert feats["sweep_type_eql"] == 1.0
    assert feats["sweep_type_eqh"] == 0.0
    assert feats["sweep_type_known"] == 1.0
    assert feats["fvg_state_fresh"] == 1.0
    assert feats["fvg_state_tapped"] == 0.0
    assert math.isclose(feats["fvg_height"], 0.7, abs_tol=1e-9)
    assert math.isfinite(feats["fvg_height_atr"])


def test_missing_indicator_results_coerced_to_zero():
    feats = extract_features(_signal(indicators=False), history=_history(), config={})
    for c in INDICATOR_CONDITIONS:
        assert feats[f"ind_{c}"] == 0.0


def test_no_history_yields_nan_volatility_not_crash():
    feats = extract_features(_signal(), history=None, config={})
    assert math.isnan(feats["atr"])
    assert math.isnan(feats["rsi"])
    # geometry that does not need history is still computed
    assert feats["sl_distance"] == 1.0


def test_no_outcome_leakage_columns():
    feats = extract_features(_signal(), history=_history(), config={})
    forbidden = {"r_multiple", "net_pnl", "gross_pnl", "exit_type", "exit_price",
                 "outcome", "win", "tp1_before_sl", "net_r", "bar_exit"}
    assert not (forbidden & set(feats.keys())), "feature row must carry NO outcome fields"
