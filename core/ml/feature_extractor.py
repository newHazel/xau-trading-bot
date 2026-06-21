"""
Phase 12.1 — Feature extractor (SKELETON).

Turns ONE completed setup into a fixed, ordered, leakage-safe feature row for the
ML confidence model. Pure function, lightweight (no GPU, no heavy compute) — safe
to call once per emitted signal inside the dataset builder or, later, live.

WHAT IS A FEATURE (safe) vs A LABEL (forbidden here):
    Features = everything known AT the moment the alert is emitted: the rulebook
               boolean vectors, the grade/score, the trade geometry (entry/SL/TP),
               volatility/momentum context, session/time, the captured sweep/FVG.
    Labels   = anything about how the trade ENDED (win/loss, exit price, R). Those
               come ONLY from core/ml/labeler.py over FUTURE bars. This module must
               never read an outcome — see the leakage note at the bottom.

WHY THE MANDATORY BOOLEANS LOOK CONSTANT: for an *approved* signal all 15 mandatory
conditions are True by definition, so on an approved-only dataset those columns
carry no variance (the trainer will drop them). The real predictive signal lives in
the 10 OPTIONAL boosters, the 4 INDICATOR confirmations, the grade/score, net R:R,
the ATR-normalized geometry, RSI, and the session/time features — i.e. the things
the grader uses to LABEL quality (A+/A/B) but NOT to gate approval. That gap is
exactly what the model can exploit on top of the rulebook.

The column set is STABLE and ORDERED (see feature_names()). Genuinely-missing
numerics are emitted as NaN (LightGBM handles NaN natively); booleans as 1.0/0.0.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import the canonical condition name lists so the feature columns can NEVER drift
# from the rulebook (if a condition is added/renamed, the columns follow).
from core.engine.rulebook_engine import (
    MANDATORY_CONDITIONS,
    OPTIONAL_CONDITIONS,
    INDICATOR_CONDITIONS,
)
from core.engine.pipeline_hooks import compute_atr, compute_rsi

NAN = float("nan")

# Ordinal encoding of the grade label (higher = better quality per the grader).
_GRADE_ORDINAL = {"A+": 4, "A": 3, "B": 2, "C": 1, "D": 0}

# Sweep liquidity-pool types we one-hot (lower-cased). Unknown/other -> all zero
# with sweep_type_known=0, so the column width stays fixed regardless of vocabulary.
_SWEEP_TYPES = ["eqh", "eql", "pdh", "pdl", "asia_high", "asia_low", "swing_high", "swing_low"]

# FVG mitigation states the engine treats as tradeable (see mitigation_tracker).
_FVG_STATES = ["fresh", "tapped", "partial", "deep"]

# ---- prefixes keep the namespace obvious in the final CSV header ---- #
_MAND_PREFIX = "mand_"
_OPT_PREFIX = "opt_"
_IND_PREFIX = "ind_"


def _scalar_feature_names() -> List[str]:
    return [
        "is_long",
        "grade_ordinal",
        "score",
        "core_score",
        "indicator_score",
        "net_rr",
        # --- geometry (ATR-normalized ones are scale-free → poolable across coins) ---
        "sl_distance",          # |entry - sl| in price units (instrument-scaled)
        "sl_distance_atr",      # |entry - sl| / ATR  (scale-free)
        "tp1_r",                # reward(TP1) / risk  (~2.0 by design)
        "tp2_r",                # reward(TP2) / risk  (~3.5 by design)
        "atr",                  # exec-TF ATR at signal time (instrument-scaled)
        "atr_pct",              # ATR / entry  (scale-free volatility regime)
        "rsi",                  # exec-TF RSI at signal time (0-100)
        # --- session / time (cyclical, scale-free) ---
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "minute_of_day",
        # --- captured sweep / FVG geometry (filled when the dicts are supplied) ---
        "sweep_type_known",
        "fvg_height", "fvg_height_atr", "fvg_age_bars",
    ]


def feature_names() -> List[str]:
    """The full, ordered list of feature column names. Stable across runs.

    Order: scalars -> sweep one-hot -> fvg-state one-hot -> mandatory -> optional
    -> indicator. Import this to write/read the dataset with a guaranteed schema.
    """
    cols: List[str] = list(_scalar_feature_names())
    cols += [f"sweep_type_{t}" for t in _SWEEP_TYPES]
    cols += [f"fvg_state_{s}" for s in _FVG_STATES]
    cols += [f"{_MAND_PREFIX}{c}" for c in MANDATORY_CONDITIONS]
    cols += [f"{_OPT_PREFIX}{c}" for c in OPTIONAL_CONDITIONS]
    cols += [f"{_IND_PREFIX}{c}" for c in INDICATOR_CONDITIONS]
    return cols


def _b(x: Any) -> float:
    """Coerce a truthy/None value to 1.0 / 0.0 (None -> 0.0)."""
    return 1.0 if bool(x) else 0.0


def _f(x: Any) -> float:
    """Coerce to float, mapping None / non-finite to NaN."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v if math.isfinite(v) else NAN


def _exec_df(history: Any, exec_tf: str):
    if not isinstance(history, dict):
        return None
    df = history.get(exec_tf)
    if df is None or getattr(df, "empty", True):
        return None
    return df


def _fvg_age_bars(fvg: Dict[str, Any], df) -> float:
    """Bars between the FVG's confirm timestamp and the last bar of the window.

    Capped at the window length when the FVG formed before the window starts.
    Returns NaN on any tz/format mismatch (so a bad value never poses as 'fresh').
    """
    if df is None or "confirm_ts" not in fvg:
        return NAN
    try:
        import pandas as pd  # local import keeps module import cheap
        ts = pd.to_datetime(fvg["confirm_ts"], utc=True)
        pos = df.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            return float(len(df))  # older than the window → at least this many bars
        return float(len(df) - 1 - pos)
    except Exception:
        return NAN


def extract_features(
    signal: Any,
    history: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    sweep: Optional[Dict[str, Any]] = None,
    fvg: Optional[Dict[str, Any]] = None,
    exec_tf: str = "5m",
) -> Dict[str, float]:
    """Build the fixed feature row for one emitted setup.

    Args:
        signal:  a PipelineSignal (has .direction/.entry/.sl/.tp1/.tp2/.grade/.score
                 and .decision: RulebookDecision with the boolean vectors + grade).
        history: the {tf: DataFrame} window fed to on_bar at emit time (for ATR/RSI).
                 May be None — ATR/RSI/age features then come back NaN.
        config:  the assembled pipeline config (reads atr_period / rsi period if set).
        sweep:   the captured sweep dict (runner._captured['sweep']) — optional.
        fvg:     the captured FVG dict (runner._captured['fvg']) — optional.
        exec_tf: execution timeframe key into `history` (default "5m").

    Returns:
        Dict[str, float] covering EXACTLY feature_names(), in that key set. Missing
        numerics are NaN; booleans are 1.0/0.0. Never raises on partial input.
    """
    cfg = config or {}
    feats: Dict[str, float] = {name: NAN for name in feature_names()}

    # ---- top-level signal fields ----
    direction = str(getattr(signal, "direction", "") or "").lower()
    feats["is_long"] = 1.0 if direction == "long" else 0.0
    feats["grade_ordinal"] = float(_GRADE_ORDINAL.get(str(getattr(signal, "grade", "")), NAN))
    feats["score"] = _f(getattr(signal, "score", None))

    entry = _f(getattr(signal, "entry", None))
    sl = _f(getattr(signal, "sl", None))
    tp1 = _f(getattr(signal, "tp1", None))
    tp2 = _f(getattr(signal, "tp2", None))

    # ---- rulebook decision (the dense boolean payload + sub-scores) ----
    decision = getattr(signal, "decision", None)
    mand = getattr(decision, "mandatory_results", None) or {}
    opt = getattr(decision, "optional_results", None) or {}
    ind = getattr(decision, "indicator_results", None) or {}
    grade_res = getattr(decision, "grade", None)
    feats["core_score"] = _f(getattr(grade_res, "core_score", None))
    feats["indicator_score"] = _f(getattr(grade_res, "indicator_score", None))
    net_rr = _f(getattr(grade_res, "net_rr", None))
    feats["net_rr"] = net_rr

    for c in MANDATORY_CONDITIONS:
        feats[f"{_MAND_PREFIX}{c}"] = _b(mand.get(c))
    for c in OPTIONAL_CONDITIONS:
        feats[f"{_OPT_PREFIX}{c}"] = _b(opt.get(c))
    # indicator_results is None on legacy rows → coerce to all-False, fixed width
    for c in INDICATOR_CONDITIONS:
        feats[f"{_IND_PREFIX}{c}"] = _b(ind.get(c))

    # ---- geometry (risk = |entry - sl|) ----
    risk = abs(entry - sl) if (math.isfinite(entry) and math.isfinite(sl)) else NAN
    feats["sl_distance"] = risk
    if math.isfinite(risk) and risk > 0:
        if math.isfinite(tp1):
            feats["tp1_r"] = (tp1 - entry) / risk if direction == "long" else (entry - tp1) / risk
        if math.isfinite(tp2):
            feats["tp2_r"] = (tp2 - entry) / risk if direction == "long" else (entry - tp2) / risk

    # ---- volatility / momentum context from the exec-TF window ----
    df = _exec_df(history, exec_tf)
    atr = NAN
    if df is not None:
        atr = _f(compute_atr(df, int(cfg.get("atr_period", 14) or 14)))
        feats["rsi"] = _f(compute_rsi(df, int(cfg.get("rsi_period", 14) or 14)))
    feats["atr"] = atr
    if math.isfinite(atr) and atr > 0:
        if math.isfinite(risk):
            feats["sl_distance_atr"] = risk / atr
        if math.isfinite(entry) and entry != 0:
            feats["atr_pct"] = atr / abs(entry)

    # ---- session / time (cyclical encodings are scale-free + wrap-correct) ----
    ts = getattr(signal, "timestamp", None)
    if isinstance(ts, datetime):
        hour = ts.hour + ts.minute / 60.0
        feats["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
        feats["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
        dow = ts.weekday()
        feats["dow_sin"] = math.sin(2 * math.pi * dow / 7.0)
        feats["dow_cos"] = math.cos(2 * math.pi * dow / 7.0)
        feats["minute_of_day"] = float(ts.hour * 60 + ts.minute)

    # ---- captured sweep one-hot ----
    if isinstance(sweep, dict):
        stype = str(sweep.get("type", "")).lower()
        known = False
        for t in _SWEEP_TYPES:
            hit = 1.0 if stype == t else 0.0
            feats[f"sweep_type_{t}"] = hit
            known = known or bool(hit)
        feats["sweep_type_known"] = 1.0 if known else 0.0
    else:
        for t in _SWEEP_TYPES:
            feats[f"sweep_type_{t}"] = 0.0
        feats["sweep_type_known"] = 0.0

    # ---- captured FVG geometry + state one-hot ----
    if isinstance(fvg, dict):
        top = _f(fvg.get("top"))
        bottom = _f(fvg.get("bottom"))
        height = abs(top - bottom) if (math.isfinite(top) and math.isfinite(bottom)) else NAN
        feats["fvg_height"] = height
        if math.isfinite(height) and math.isfinite(atr) and atr > 0:
            feats["fvg_height_atr"] = height / atr
        feats["fvg_age_bars"] = _fvg_age_bars(fvg, df)
        fstate = str(fvg.get("state", "")).lower()
        for s in _FVG_STATES:
            feats[f"fvg_state_{s}"] = 1.0 if fstate == s else 0.0
    else:
        for s in _FVG_STATES:
            feats[f"fvg_state_{s}"] = 0.0

    return feats


# --------------------------------------------------------------------------- #
# LEAKAGE GUARD (read before adding any feature)
#
#   * Never read trade outcome here: exit_price, r_multiple, net_pnl, exit_type,
#     bar_exit, "did it win" — these live on labeler/TradeRecord and are the TARGET.
#   * Every feature above is computed from the SIGNAL bar and bars BEFORE it
#     (compute_atr/compute_rsi use df up to the last window bar = the signal bar;
#     sweep/fvg were captured at/before the signal bar). No future bars are touched.
#   * `setup_id` / `timestamp` are identifiers — keep them as join keys in the
#     dataset builder, do NOT feed them to the model as features.
# --------------------------------------------------------------------------- #
