"""Tests for backtesting/bootstrap.py — small-sample CIs, diff CIs, Holm correction."""

import math

from backtesting.bootstrap import bootstrap_ci, bootstrap_diff, holm_threshold


class TestBootstrapCI:
    def test_point_matches_inputs(self):
        rs = [2.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, -1.0]  # 50% win, +2/-1, total +4
        r = bootstrap_ci(rs, n_resamples=2000, seed=1)
        assert r["point"]["win_rate"] == 0.5
        assert r["point"]["total_r"] == 4.0
        assert r["point"]["expectancy"] == 0.5
        assert r["point"]["profit_factor"] == 2.0  # 8 / 4

    def test_ci_brackets_point(self):
        rs = [1.5, -1.0, 2.0, -1.0, 1.8, -1.0, 2.2, 1.0, -1.0, 1.5]
        r = bootstrap_ci(rs, n_resamples=3000, seed=2)
        lo, hi = r["ci"]["expectancy"]
        assert lo <= r["point"]["expectancy"] <= hi
        lo_t, hi_t = r["ci"]["total_r"]
        assert lo_t <= r["point"]["total_r"] <= hi_t

    def test_clear_winner_low_p_no_edge(self):
        rs = [3.0] * 25 + [-1.0] * 5  # strongly positive
        r = bootstrap_ci(rs, n_resamples=3000, seed=3)
        assert r["p_no_edge"] < 0.05

    def test_clear_loser_high_p_no_edge(self):
        rs = [-1.0] * 25 + [2.0] * 3  # strongly negative
        r = bootstrap_ci(rs, n_resamples=3000, seed=4)
        assert r["p_no_edge"] > 0.9

    def test_insufficient_below_two(self):
        r = bootstrap_ci([1.5], n_resamples=1000)
        assert r["insufficient"] is True
        assert r["n"] == 1

    def test_deterministic(self):
        rs = [1.0, -1.0, 2.0, -1.0, 1.5, 0.5]
        a = bootstrap_ci(rs, n_resamples=1500, seed=42)
        b = bootstrap_ci(rs, n_resamples=1500, seed=42)
        assert a["ci"]["expectancy"] == b["ci"]["expectancy"]
        assert a["p_no_edge"] == b["p_no_edge"]

    def test_all_winners_pf_inf_tracked(self):
        r = bootstrap_ci([1.0, 2.0, 1.5, 3.0], n_resamples=1000, seed=5)
        assert r["point"]["profit_factor"] == float("inf")
        assert r["pf_inf_share"] > 0.5  # most resamples are all-winners too


class TestBootstrapDiff:
    def test_treatment_better(self):
        treat = [2.0, 2.0, 2.0, -1.0, 2.0, 2.0, 2.0, -1.0]   # exp ~1.25
        base = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]  # exp 0
        d = bootstrap_diff(treat, base, metric="expectancy", n_resamples=3000, seed=6)
        assert d["point_diff"] > 0
        assert d["p_treatment_worse_or_equal"] < 0.5

    def test_treatment_worse(self):
        treat = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
        base = [2.0, 2.0, -1.0, 2.0, 2.0, -1.0]
        d = bootstrap_diff(treat, base, metric="expectancy", n_resamples=3000, seed=7)
        assert d["point_diff"] < 0
        assert d["p_treatment_worse_or_equal"] > 0.5

    def test_insufficient(self):
        d = bootstrap_diff([1.0], [2.0, 1.0], metric="expectancy")
        assert d["insufficient"] is True


class TestHolm:
    def test_all_significant(self):
        assert holm_threshold([0.001, 0.002, 0.003], alpha=0.05) == [True, True, True]

    def test_none_significant(self):
        assert holm_threshold([0.5, 0.6, 0.7], alpha=0.05) == [False, False, False]

    def test_step_down_partial(self):
        # 3 tests at alpha 0.05: smallest must beat 0.05/3=0.0167; next 0.05/2=0.025; last 0.05
        surv = holm_threshold([0.01, 0.04, 0.9], alpha=0.05)
        assert surv[0] is True       # 0.01 <= 0.0167
        assert surv[1] is False      # 0.04 > 0.025 → fails, and step-down stops
        assert surv[2] is False

    def test_preserves_input_order(self):
        # the largest p-value is first in the input; result must map back to input order
        surv = holm_threshold([0.9, 0.001, 0.002], alpha=0.05)
        assert surv[0] is False
        assert surv[1] is True
        assert surv[2] is True

    def test_empty(self):
        assert holm_threshold([], alpha=0.05) == []
