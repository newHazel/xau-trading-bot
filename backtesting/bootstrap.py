"""
Bootstrap confidence intervals for backtest metrics — small-sample honesty.

At ~30 trades a point-estimate PF / win% / expectancy is almost meaningless: the
95% CI (and the probability the true total-R is <= 0) is what tells you whether an
"edge" is real or just noise. We resample the per-trade R list WITH REPLACEMENT
(same sample size) `n_resamples` times and read percentiles off the distribution.

Deterministic: a fixed seed makes the CIs reproducible across runs (important for a
re-score/aggregate layer that may run many times over the same checkpoints).

Pure stdlib (random) — fast enough for ~30-200 trades × 5000 resamples.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple


def _profit_factor(rs: List[float]) -> float:
    gross_win = sum(r for r in rs if r > 0)
    gross_loss = abs(sum(r for r in rs if r <= 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def _point_stats(rs: List[float]) -> Dict[str, float]:
    n = len(rs)
    if n == 0:
        return {"win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "total_r": 0.0}
    wins = sum(1 for r in rs if r > 0)
    return {
        "win_rate": wins / n,
        "profit_factor": _profit_factor(rs),
        "expectancy": sum(rs) / n,
        "total_r": sum(rs),
    }


def _percentile(vals: List[float], q: float) -> float:
    """Nearest-rank percentile over the finite values (NaN/inf dropped)."""
    clean = sorted(v for v in vals if v == v and v not in (float("inf"), float("-inf")))
    if not clean:
        return float("nan")
    idx = int(round(q * (len(clean) - 1)))
    idx = max(0, min(len(clean) - 1, idx))
    return clean[idx]


_KEYS = ("win_rate", "profit_factor", "expectancy", "total_r")


def bootstrap_ci(
    r_values: List[float],
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """95% (default) bootstrap CIs for win_rate, profit_factor, expectancy, total_r,
    plus p_no_edge = P(resampled total_r <= 0) — a direct 'probability there is no edge'.

    Returns {n, insufficient, point:{...}, ci:{key:(lo,hi)}, p_no_edge, pf_inf_share}.
    With < 2 trades the CI collapses to the point estimate and insufficient=True.
    """
    rs = [float(r) for r in r_values]
    n = len(rs)
    point = _point_stats(rs)
    if n < 2:
        return {
            "n": n, "insufficient": True, "point": point,
            "ci": {k: (point[k], point[k]) for k in _KEYS},
            "p_no_edge": 1.0 if point["total_r"] <= 0 else 0.0,
            "pf_inf_share": 0.0,
        }

    rng = random.Random(seed)
    dist: Dict[str, List[float]] = {k: [] for k in _KEYS}
    n_no_edge = 0
    pf_inf = 0
    for _ in range(n_resamples):
        sample = [rs[rng.randrange(n)] for _ in range(n)]  # resample w/ replacement, same N
        st = _point_stats(sample)
        for k in _KEYS:
            v = st[k]
            if k == "profit_factor" and v == float("inf"):
                pf_inf += 1
                continue  # exclude inf from the percentile, count it separately
            dist[k].append(v)
        if st["total_r"] <= 0:
            n_no_edge += 1

    lo, hi = alpha / 2.0, 1.0 - alpha / 2.0
    ci: Dict[str, Tuple[float, float]] = {
        k: (_percentile(dist[k], lo), _percentile(dist[k], hi)) for k in _KEYS
    }
    return {
        "n": n, "insufficient": False, "point": point, "ci": ci,
        "p_no_edge": n_no_edge / n_resamples,
        "pf_inf_share": pf_inf / n_resamples,
    }


def bootstrap_diff(
    r_treatment: List[float],
    r_baseline: List[float],
    metric: str = "expectancy",
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 7,
) -> Dict[str, Any]:
    """CI for the DIFFERENCE (treatment - baseline) of a metric, via independent
    resampling of each arm. If the CI excludes 0 the lever moved the metric beyond
    noise. Used by the variant ablation so we don't promote a lever on a point diff.

    metric in {win_rate, profit_factor, expectancy, total_r}. Returns
    {point_diff, ci:(lo,hi), p_treatment_worse_or_equal, n_t, n_b, insufficient}.
    """
    a = [float(r) for r in r_treatment]
    b = [float(r) for r in r_baseline]
    if len(a) < 2 or len(b) < 2:
        pa = _point_stats(a).get(metric, 0.0)
        pb = _point_stats(b).get(metric, 0.0)
        return {"point_diff": pa - pb, "ci": (pa - pb, pa - pb),
                "p_treatment_worse_or_equal": 1.0, "n_t": len(a), "n_b": len(b),
                "insufficient": True}
    rng = random.Random(seed)
    diffs: List[float] = []
    worse = 0
    for _ in range(n_resamples):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        va = _point_stats(sa).get(metric, 0.0)
        vb = _point_stats(sb).get(metric, 0.0)
        if va in (float("inf"), float("-inf")) or vb in (float("inf"), float("-inf")):
            continue
        d = va - vb
        diffs.append(d)
        if d <= 0:
            worse += 1
    if not diffs:
        return {"point_diff": 0.0, "ci": (float("nan"), float("nan")),
                "p_treatment_worse_or_equal": 1.0, "n_t": len(a), "n_b": len(b),
                "insufficient": True}
    pa = _point_stats(a).get(metric, 0.0)
    pb = _point_stats(b).get(metric, 0.0)
    return {
        "point_diff": pa - pb,
        "ci": (_percentile(diffs, alpha / 2.0), _percentile(diffs, 1.0 - alpha / 2.0)),
        "p_treatment_worse_or_equal": worse / len(diffs),
        "n_t": len(a), "n_b": len(b), "insufficient": False,
    }


def holm_threshold(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni step-down: given K p-values, return a boolean per input (in the
    ORIGINAL order) marking which survive at family-wise alpha. Use to keep the variant
    ablation honest when testing several levers at once (multiple-testing correction)."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    survive = [False] * m
    for rank, idx in enumerate(order):
        adj = alpha / (m - rank)
        if p_values[idx] <= adj:
            survive[idx] = True
        else:
            break  # step-down: once one fails, all larger p-values fail too
    return survive
