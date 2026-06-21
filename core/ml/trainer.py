"""
Phase 12.3/12.4 — Trainer + edge evaluator (SKELETON).

Trains the confidence model on the labeled dataset from Phase 12.2 and answers the ONLY
question that matters (the project's hard-won lesson): does filtering approved setups by
the model's P_win ACTUALLY raise profit-factor / expectancy vs taking them all? Accuracy
is irrelevant — a low-win% / high-R:R edge is judged by PF & expectancy.

Design choices baked in:
  * TARGET = tp1_before_sl (binary). The model estimates P(setup reaches TP1 before SL).
  * MODEL  = gradient boosting on tabular features. Prefers LightGBM; falls back to
    sklearn's HistGradientBoostingClassifier (both CPU, both handle NaN natively) so the
    code runs even before `pip install -r requirements-ml.txt`.
  * VALIDATION = WALK-FORWARD (chronological, expanding window). NEVER shuffle a time
    series — train on the past, test on the future, or you leak and fool yourself.
  * EVALUATION = a threshold sweep on the out-of-sample predictions, scored by realized R
    (net_r if present, else the binary win_r), reported as signals / win% / PF / expectancy
    and compared to the unfiltered baseline. That is the 12.4 evaluator.

Pure functions here (split / sweep / metrics) are dependency-light and unit-tested; only
make_model()/fit_predict_oos() touch the ML backend (lazy import).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.ml.feature_extractor import feature_names

# Columns that are LABELS or identifiers — never features. (feature_names() already
# excludes these, but we hard-guard against accidental leakage.)
LABEL_COLS = {"triggered", "outcome", "tp1_before_sl", "win_r", "fill_offset",
              "resolve_offset", "net_r", "net_r_exit_type"}
ID_COLS = {"instrument", "symbol", "setup_id", "gpos", "ts", "direction",
           "entry", "sl", "tp1", "tp2", "grade"}

DEFAULT_TARGET = "tp1_before_sl"


def load_dataset(path: str) -> pd.DataFrame:
    """Load a Phase 12.2 dataset CSV (or parquet). Parses ts as tz-aware UTC."""
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def resolved_subset(df: pd.DataFrame, target: str = DEFAULT_TARGET) -> pd.DataFrame:
    """Keep only TRAINABLE rows: a real entry fill AND a resolved binary outcome.

    Drops NO_FILL (never triggered) and OPEN (censored) — they have target == NaN and
    must NOT be imputed as losses (survivorship bias).
    """
    if target not in df.columns:
        raise KeyError(f"target column '{target}' not in dataset")
    out = df[df[target].notna()].copy()
    out[target] = out[target].astype(int)
    return out


def feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Build the X matrix from the stable feature schema, guarding against leakage.

    Uses exactly the columns from feature_extractor.feature_names() that are present,
    and asserts none of them is a label/id column.
    """
    cols = [c for c in feature_names() if c in df.columns]
    leak = (set(cols) & (LABEL_COLS | ID_COLS))
    if leak:
        raise ValueError(f"leakage: feature columns overlap labels/ids: {sorted(leak)}")
    X = df[cols].astype(float).to_numpy()
    return X, cols


def pick_r_column(df: pd.DataFrame) -> str:
    """Choose the realized-R column for edge evaluation.

    Prefer net_r (cost-aware, the number the whole project judges by) when it is present
    and populated for most resolved rows; otherwise fall back to win_r (the cost-free
    binary R: ~+tp1_r for a win, -1.0 for a loss).
    """
    if "net_r" in df.columns and df["net_r"].notna().mean() >= 0.5:
        return "net_r"
    return "win_r"


def expectancy_pf(r: np.ndarray) -> Dict[str, float]:
    """Profit-factor / expectancy of a set of realized R-multiples. r>0 = win (matches
    backtesting.metrics.compute_metrics)."""
    r = np.asarray([x for x in r if x is not None and np.isfinite(x)], dtype=float)
    n = len(r)
    if n == 0:
        return {"n": 0, "win_rate": float("nan"), "pf": float("nan"),
                "expectancy": float("nan"), "total_r": 0.0}
    wins = r[r > 0]
    gross_win = float(wins.sum())
    gross_loss = float(-r[r <= 0].sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    return {"n": n, "win_rate": float(len(wins) / n), "pf": pf,
            "expectancy": float(r.mean()), "total_r": float(r.sum())}


def walk_forward_folds(n_rows: int, n_folds: int = 4) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window chronological folds over time-SORTED rows.

    Returns [(train_idx, test_idx), ...]: fold k trains on all rows before a cut and tests
    on the next block. NO future row ever appears in a train set (max(train) < min(test)).
    """
    if n_rows < 2 or n_folds < 1:
        return []
    edges = [round(n_rows * k / (n_folds + 1)) for k in range(n_folds + 2)]
    folds = []
    for i in range(n_folds):
        tr_end = edges[i + 1]
        te_end = edges[i + 2]
        if tr_end <= 0 or te_end <= tr_end:
            continue
        folds.append((np.arange(0, tr_end), np.arange(tr_end, te_end)))
    return folds


def make_model(random_state: int = 42) -> Tuple[Any, str]:
    """Gradient-boosting classifier. LightGBM if installed, else sklearn HistGB. Both
    expose predict_proba and handle NaN features natively."""
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            random_state=random_state, verbose=-1,
        )
        return model, "lightgbm"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, random_state=random_state,
        )
        return model, "sklearn-histgb"


def fit_predict_oos(
    X: np.ndarray, y: np.ndarray, n_folds: int = 4
) -> Tuple[np.ndarray, Optional[str]]:
    """Walk-forward out-of-sample P_win for every row whose fold could be trained.

    Rows in the first (train-only) block, or in a fold whose train set has a single class,
    stay NaN (no honest OOS prediction). Returns (oos_pwin, backend_name).
    """
    n = len(X)
    oos = np.full(n, np.nan, dtype=float)
    backend = None
    for tr, te in walk_forward_folds(n, n_folds):
        if len(np.unique(y[tr])) < 2:
            continue  # can't train a classifier on one class
        model, backend = make_model()
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])
        # column index of the positive class (1)
        classes = list(getattr(model, "classes_", [0, 1]))
        pos = classes.index(1) if 1 in classes else (proba.shape[1] - 1)
        oos[te] = proba[:, pos]
    return oos, backend


def threshold_sweep(
    p_win: np.ndarray, r: np.ndarray, thresholds: Optional[np.ndarray] = None
) -> List[Dict[str, float]]:
    """For each threshold tau, score the kept set (P_win >= tau) by realized R.

    Row 0 (tau=0.0) is the BASELINE = take every scored setup. Rows are comparable only
    over the same OOS population (rows with a finite P_win AND a finite R).
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 0.86, 0.05), 2)
    p = np.asarray(p_win, dtype=float)
    rr = np.asarray(r, dtype=float)
    mask_scored = np.isfinite(p) & np.isfinite(rr)
    p, rr = p[mask_scored], rr[mask_scored]
    base_n = len(p)
    rows = []
    for tau in thresholds:
        keep = p >= tau
        m = expectancy_pf(rr[keep])
        m["threshold"] = float(tau)
        m["kept_pct"] = float(100.0 * m["n"] / base_n) if base_n else 0.0
        rows.append(m)
    return rows


def recommend_threshold(
    sweep: List[Dict[str, float]], min_signal_frac: float = 0.30, min_signals: int = 20
) -> Optional[Dict[str, float]]:
    """Pick the tau that MAXIMIZES expectancy while keeping enough signals (so we don't
    curve-fit to a handful of lucky trades). Returns None if nothing qualifies."""
    if not sweep:
        return None
    base_n = max((row["n"] for row in sweep), default=0)
    floor = max(min_signals, int(min_signal_frac * base_n))
    eligible = [row for row in sweep
                if row["n"] >= floor and np.isfinite(row.get("expectancy", float("nan")))]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r["expectancy"])
