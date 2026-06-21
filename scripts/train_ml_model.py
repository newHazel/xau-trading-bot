"""
Phase 12.3/12.4 — Train + evaluate the confidence model (SKELETON).

Reads the Phase 12.2 dataset, trains a gradient-boosting P_win model with WALK-FORWARD
(chronological) validation, and prints the edge table that decides everything:

    does filtering approved setups by P_win >= tau raise PF / expectancy vs taking them all?

Then it fits a final calibrated model on all data and saves it for the 12.5 inference gate.
Light enough to run on the dev laptop (sklearn fallback) or RunPod (LightGBM); training is
seconds-to-minutes on a few hundred-to-thousand rows — NO GPU.

    python scripts/train_ml_model.py --dataset data/processed/ml/ml_dataset_all.csv
    python scripts/train_ml_model.py --dataset .../ml_dataset_eth.csv --n-folds 5

Output: <out-dir>/ml_model.pkl (joblib: model + features + chosen threshold + metadata)
        <out-dir>/ml_report.json (the sweep + recommendation, for the record).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from core.ml import trainer as T

MIN_SAMPLES = 150  # below this, results are noise — warn loudly (the project's #1 lesson)


def _fmt(m: dict) -> str:
    pf = m["pf"]
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (f"{m['threshold']:>5.2f}{m['n']:>8d}{m['kept_pct']:>8.0f}%"
            f"{m['win_rate']*100:>8.1f}{pf_s:>8}{m['expectancy']:>10.3f}{m['total_r']:>9.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train + evaluate the Phase 12 confidence model.")
    p.add_argument("--dataset", default=str(_ROOT / "data" / "processed" / "ml" / "ml_dataset_all.csv"))
    p.add_argument("--target", default=T.DEFAULT_TARGET)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--out-dir", default=str(_ROOT / "data" / "processed" / "ml"))
    p.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    p.add_argument("--r-column", default="auto", choices=["auto", "net_r", "win_r"],
                   help="Realized-R column to SCORE the edge table by. 'auto' prefers net_r "
                        "(cost-aware but understates TP1-then-SL); 'win_r' matches the model's "
                        "tp1_before_sl target exactly (win=+~2R, loss=-1R).")
    a = p.parse_args()

    df = T.load_dataset(a.dataset)
    res = T.resolved_subset(df, a.target)
    if "ts" in res.columns:
        res = res.sort_values("ts").reset_index(drop=True)  # chronological for walk-forward

    n = len(res)
    print(f"=== train confidence model | dataset={Path(a.dataset).name} | "
          f"resolved rows={n} | target={a.target} ===")
    if n < a.min_samples:
        print(f"⚠️  ONLY {n} resolved rows (< {a.min_samples}). Results below are NOISE, not an "
              f"edge — gather more forward/backtest data before trusting any threshold.")
    if n < 2 * (a.n_folds + 1):
        print("🔴 Too few rows for walk-forward validation. Build a bigger dataset first.")
        sys.exit(1)

    X, feat_cols = T.feature_matrix(res)
    y = res[a.target].to_numpy().astype(int)
    base_rate = y.mean()
    print(f"  features={len(feat_cols)} | base win-rate={base_rate*100:.1f}% | folds={a.n_folds}")

    # --- walk-forward out-of-sample P_win ---
    oos, backend = T.fit_predict_oos(X, y, n_folds=a.n_folds)
    scored = np.isfinite(oos).sum()
    print(f"  model backend: {backend} | OOS-scored rows: {scored}/{n}")
    if scored == 0:
        print("🔴 No fold could be trained (single-class folds). Need more/balanced data.")
        sys.exit(1)

    # --- edge evaluation: sweep thresholds, score by realized R ---
    r_col = T.pick_r_column(res) if a.r_column == "auto" else a.r_column
    if r_col not in res.columns:
        r_col = T.pick_r_column(res)
    r = res[r_col].to_numpy(dtype=float)
    sweep = T.threshold_sweep(oos, r)
    print(f"\n  EDGE TABLE (realized R from '{r_col}'; row tau=0.00 = BASELINE = take all):")
    print(f"  {'tau':>5}{'signals':>8}{'kept':>9}{'win%':>8}{'PF':>8}{'expR':>10}{'totalR':>9}")
    for row in sweep:
        print(f"  {_fmt(row)}")

    baseline = sweep[0] if sweep else None
    rec = T.recommend_threshold(sweep)
    print("\n  --- verdict ---")
    if rec and baseline:
        better = (rec["pf"] >= baseline["pf"] and rec["expectancy"] > baseline["expectancy"])
        print(f"  best tau={rec['threshold']:.2f}: keeps {rec['n']} signals "
              f"({rec['kept_pct']:.0f}%), PF {rec['pf']:.2f} vs baseline {baseline['pf']:.2f}, "
              f"expR {rec['expectancy']:+.3f} vs {baseline['expectancy']:+.3f}")
        print(f"  → ML filter {'ADDS edge on this sample (consider 12.5 deploy, flag-OFF first)' if better else 'does NOT clearly beat taking all setups — do not deploy yet'}.")
    else:
        print("  no threshold kept enough signals to judge — gather more data.")

    # --- fit final model on ALL data + calibrate, then save for the 12.5 gate ---
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, backend = T.make_model()
    final, calibrated = model, False
    try:
        from sklearn.calibration import CalibratedClassifierCV
        cc = CalibratedClassifierCV(model, method="isotonic", cv=3)
        cc.fit(X, y)
        final, calibrated = cc, True
    except Exception as exc:
        print(f"  (calibration skipped: {type(exc).__name__}: {exc}; saving uncalibrated)")
        final.fit(X, y)

    rec_tau = rec["threshold"] if rec else 0.5
    payload = {"model": final, "features": feat_cols, "backend": backend,
               "calibrated": calibrated, "target": a.target, "r_column": r_col,
               "recommended_threshold": rec_tau, "n_train": n, "base_rate": float(base_rate)}
    try:
        import joblib
        joblib.dump(payload, out_dir / "ml_model.pkl")
        print(f"\n  saved model → {out_dir / 'ml_model.pkl'} (backend={backend}, "
              f"calibrated={calibrated}, tau={rec_tau:.2f})")
    except Exception as exc:
        print(f"  ⚠️  could not save model ({type(exc).__name__}: {exc})")

    report = {"dataset": str(a.dataset), "resolved_rows": n, "backend": backend,
              "r_column": r_col, "base_win_rate": float(base_rate),
              "recommended_threshold": rec_tau, "sweep": sweep}
    (out_dir / "ml_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"  saved report → {out_dir / 'ml_report.json'}")


if __name__ == "__main__":
    main()
