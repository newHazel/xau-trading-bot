"""Tests for Phase 12.3/12.4 trainer + evaluator (core/ml/trainer.py).

Pure functions are tested directly; the model path uses whatever backend make_model()
resolves (LightGBM if installed, else sklearn HistGB) on small synthetic data."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.ml import trainer as T
from core.ml.feature_extractor import feature_names


def test_walk_forward_folds_no_lookahead():
    folds = T.walk_forward_folds(400, n_folds=4)
    assert len(folds) == 4
    prev_test_start = -1
    for tr, te in folds:
        assert tr.max() < te.min(), "train must be strictly before test (no leakage)"
        assert te.min() > prev_test_start, "test blocks must move forward in time"
        prev_test_start = te.min()
    # expanding window: each fold's train set grows
    assert folds[0][0].max() < folds[-1][0].max()


def test_expectancy_pf_math():
    # 2 wins (+2R) and 3 losses (-1R): PF = 4 / 3, expectancy = (4-3)/5 = 0.2
    r = np.array([2.0, 2.0, -1.0, -1.0, -1.0])
    m = T.expectancy_pf(r)
    assert m["n"] == 5
    assert math.isclose(m["win_rate"], 2 / 5)
    assert math.isclose(m["pf"], 4 / 3, rel_tol=1e-9)
    assert math.isclose(m["expectancy"], 0.2, rel_tol=1e-9)
    assert math.isclose(m["total_r"], 1.0)


def test_expectancy_pf_all_wins_is_inf():
    assert T.expectancy_pf(np.array([2.0, 1.0]))["pf"] == float("inf")
    assert T.expectancy_pf(np.array([]))["n"] == 0


def test_resolved_subset_drops_unresolved():
    df = pd.DataFrame({"tp1_before_sl": [1.0, 0.0, np.nan, 1.0], "x": [1, 2, 3, 4]})
    out = T.resolved_subset(df)
    assert len(out) == 3
    assert out["tp1_before_sl"].tolist() == [1, 0, 1]
    assert out["tp1_before_sl"].dtype.kind in "iu"


def test_feature_matrix_excludes_labels_and_ids():
    cols = feature_names()
    df = pd.DataFrame({c: np.linspace(0, 1, 5) for c in cols})
    # add label/id columns that must be ignored
    df["tp1_before_sl"] = 1
    df["net_r"] = 2.0
    df["setup_id"] = "x"
    df["entry"] = 100.0
    X, used = T.feature_matrix(df)
    assert used == [c for c in cols if c in df.columns]
    assert X.shape == (5, len(used))
    assert "tp1_before_sl" not in used and "net_r" not in used and "entry" not in used


def test_pick_r_column_prefers_net_r_when_populated():
    df = pd.DataFrame({"net_r": [1.0, -1.0, 2.0, np.nan], "win_r": [2.0, -1.0, 2.0, -1.0]})
    assert T.pick_r_column(df) == "net_r"
    df2 = pd.DataFrame({"net_r": [np.nan, np.nan, np.nan, 1.0], "win_r": [2.0, -1.0, 2.0, -1.0]})
    assert T.pick_r_column(df2) == "win_r"  # net_r mostly missing → fall back


def test_threshold_sweep_filtering_improves_expectancy():
    # construct p_win that ranks trades: higher p_win => more likely a win (+2R) else -1R
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 600)
    win = rng.uniform(0, 1, 600) < p           # win probability == p
    r = np.where(win, 2.0, -1.0)
    sweep = T.threshold_sweep(p, r)
    base = sweep[0]
    high = [row for row in sweep if row["threshold"] >= 0.6][0]
    assert base["threshold"] == 0.0
    assert high["expectancy"] > base["expectancy"], "filtering to high P_win must raise expectancy"
    assert high["n"] < base["n"], "a higher threshold keeps fewer signals"


def test_recommend_threshold_respects_signal_floor():
    sweep = [
        {"threshold": 0.0, "n": 100, "expectancy": 0.10, "pf": 1.2, "win_rate": 0.4, "total_r": 10, "kept_pct": 100},
        {"threshold": 0.5, "n": 60, "expectancy": 0.40, "pf": 1.8, "win_rate": 0.5, "total_r": 24, "kept_pct": 60},
        {"threshold": 0.9, "n": 3, "expectancy": 5.00, "pf": 9.9, "win_rate": 0.9, "total_r": 15, "kept_pct": 3},
    ]
    rec = T.recommend_threshold(sweep, min_signal_frac=0.30, min_signals=20)
    assert rec["threshold"] == 0.5, "must skip the 3-signal lucky spike and pick the robust tau"


def test_fit_predict_oos_learns_a_real_signal():
    # synthetic separable data: feature 0 drives the win probability; others are noise.
    rng = np.random.default_rng(7)
    n = 500
    x0 = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, (n, 4))
    X = np.column_stack([x0, noise])
    p = 1 / (1 + np.exp(-2.5 * x0))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    oos, backend = T.fit_predict_oos(X, y, n_folds=4)
    scored = np.isfinite(oos)
    assert scored.sum() > 0 and backend is not None
    # the model should rank winners above losers out-of-sample
    win_mean = np.nanmean(oos[scored & (y == 1)])
    loss_mean = np.nanmean(oos[scored & (y == 0)])
    assert win_mean > loss_mean, f"OOS P_win should separate classes ({win_mean:.3f} vs {loss_mean:.3f})"
