"""
Phase 12.2 — ML dataset builder (RunPod-ready, SKELETON).

Fans the faithful SequenceRunner across CPU cores to turn historical candles into a
LABELED training dataset for the Phase 12 confidence model:

    for each instrument (gold + the 9-coin crypto fleet):
        GEN   (parallel, chunked, checkpointed)  — replay SequenceRunner, and at every
              emitted A+/A/B setup snapshot the 12.1 feature row (+ captured sweep/FVG)
        LABEL (single pass over the FULL exec df) — for each setup compute the
              tp1_before_sl binary label (and optionally net_r) from FUTURE bars only
        WRITE one CSV per instrument + a combined CSV  (data/processed/, gitignored)

It deliberately REUSES the verified machinery in scripts/backtest_sequence_parallel.py
(_load = CSV-prefer-then-DB, _win = past-only window, _chunks, the crypto_pct VARIANT
+ one-level deep-merge) so labels match live/backtest behavior and can't drift.

WHY THIS IS THE HEAVY STEP: signal GENERATION (the per-bar SMC detector stack) is the
slow part — exactly what RunPod's many cores are for. Labeling is cheap. Run this on a
RunPod CPU pod (NOT a GPU pod — gradient boosting needs no GPU; see Phase 12 notes).

    # 9 crypto coins (fully reproducible from committed data/candles CSVs — no API):
    python scripts/build_ml_dataset.py --instruments crypto --jobs 16

    # add gold (needs XAUUSD candles in the SQLite DB or a fetched CSV):
    python scripts/build_ml_dataset.py --instruments all --jobs 16 --net-r

    # prove chunked == sequential before trusting a big run:
    python scripts/build_ml_dataset.py --instruments crypto --verify

RunPod tuning: a RunPod pod gives DEDICATED vCPUs (unlike a shared Railway container),
so set --jobs to the pod's vCPU count. Mount a persistent volume and point --out-dir at
it so a preemption/restart resumes from the per-chunk checkpoints instead of regenerating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# allow importing the sibling backtest script (pure helpers, no side effects on import)
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import pandas as pd

from core.logging.db import get_db
from core.engine.sequence_runner import SequenceRunner
from core.engine.pipeline_config import assemble_pipeline_config
from core.ml.feature_extractor import extract_features, feature_names
from core.ml import labeler as L

import backtest_sequence_parallel as bsp  # _load, _win, _chunks, VARIANTS

# Authoritative crypto universe — keep in sync with scripts/live_alerts_crypto.py
# DEFAULT_SYMBOLS and scripts/export_candles_csv.py (all 9 have committed candle CSVs).
CRYPTO_SYMBOLS = [
    "ETHUSDT", "DOGEUSDT", "SOLUSDT", "LINKUSDT",
    "AVAXUSDT", "NEARUSDT", "SUIUSDT", "SANDUSDT", "ZECUSDT",
]
GOLD_SYMBOL = "XAUUSD"

# Anchored to the repo root so the script is CWD-agnostic (like bsp._load's CSV path) —
# safe to launch from anywhere on a RunPod pod / cron, not only from the repo root.
OUT_DEFAULT = str(_ROOT / "data" / "processed" / "ml")
DB_DEFAULT = str(_ROOT / "data" / "database" / "trading_bot.sqlite")


def _default_jobs() -> int:
    """Default worker count. Prefer the cgroup/cpuset-aware affinity count (correct on a
    RunPod/Linux pod) over os.cpu_count() (which reports HOST cores). Capped at 8 so a bare
    run on the dev laptop doesn't saturate it — on RunPod pass --jobs = the pod's vCPUs."""
    try:
        n = len(os.sched_getaffinity(0))   # Linux: respects cpuset/quota
    except (AttributeError, OSError):
        n = os.cpu_count() or 2            # macOS/other: no sched_getaffinity
    return max(1, min(n, 8))


def _run_tag(cfg: dict, exec_tf: str, warmup: int, window: int) -> str:
    """Short digest of the load-bearing params. Embedded in the checkpoint filename so a
    re-run with a CHANGED config/exec_tf/warmup/window can NEVER silently reuse stale rows
    from a previous run on the same persistent volume (the high-severity resume trap)."""
    blob = json.dumps({"cfg": cfg, "tf": exec_tf, "wu": warmup, "w": window},
                      sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:8]

# Meta/id columns written alongside the feature columns. NOT model inputs — join keys.
_META_COLS = ["instrument", "symbol", "setup_id", "gpos", "ts", "direction",
              "entry", "sl", "tp1", "tp2", "grade"]
# Label columns appended in the LABEL pass.
_LABEL_COLS = ["triggered", "outcome", "tp1_before_sl", "win_r",
               "fill_offset", "resolve_offset", "net_r", "net_r_exit_type"]


def _merge(cfg: dict, overrides: dict) -> dict:
    """One-level deep-merge — identical to backtest_sequence_parallel._gen_chunk so a
    crypto config (rr_tiers / costs / spread sub-dicts) is built the same way live is."""
    out = dict(cfg)
    for k, v in overrides.items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _instrument_plan(which: str) -> list:
    """Return [(instrument_key, symbol, cfg, cost_overrides), ...].

    instrument_key labels the output; cfg/cost_overrides replicate the LIVE config so
    the dataset matches deployed behavior:
        gold   = assemble_pipeline_config (absolute costs, kill-zone enforced)
        crypto = + VARIANTS['crypto_pct'] (percent costs, 24/7, freshness, price-sanity)
    """
    base = dict(assemble_pipeline_config(str(_ROOT / "config")))
    crypto_ov = bsp.VARIANTS["crypto_pct"]
    plan = []
    want_gold = which in ("gold", "all")
    want_crypto = which in ("crypto", "all")
    if want_gold:
        plan.append(("gold", GOLD_SYMBOL, dict(base), None))  # None → absolute cost model
    if want_crypto:
        crypto_cfg = _merge(base, crypto_ov)
        for sym in CRYPTO_SYMBOLS:
            plan.append((sym.replace("USDT", "").lower(), sym, dict(crypto_cfg), crypto_ov))
    if not plan:  # explicit symbol list (comma-separated) → treat as crypto-config
        crypto_cfg = _merge(base, crypto_ov)
        for sym in [s.strip().upper() for s in which.split(",") if s.strip()]:
            plan.append((sym.replace("USDT", "").lower(), sym, dict(crypto_cfg), crypto_ov))
    return plan


def _gen_ml_chunk(task):
    """Worker (top-level for spawn-safety): emit feature rows for [real_start, real_end).

    Mirrors backtest_sequence_parallel._gen_chunk but records the FULL 12.1 feature row
    per emitted setup (reading the runner's captured sweep/FVG for geometry features).
    Checkpoints to disk so a kill/preemption resumes.
    """
    (ikey, symbol, cfg, real_start, real_end, warmup, window, exec_tf, db_path, out_dir, run_tag) = task
    t0 = time.time()
    # checkpoint key includes run_tag (digest of exec_tf/warmup/window/cfg) AND the chunk's
    # [real_start,real_end) bounds → a re-run with changed params/chunking can't reuse stale
    # or wrong-range rows from a previous run on the same volume (the resume corruption trap).
    ckpt = os.path.join(out_dir, f"ml_{ikey}_{run_tag}_{real_start:08d}_{real_end:08d}.json")
    if os.path.exists(ckpt):
        try:
            with open(ckpt) as f:
                out = json.load(f)
            if all(real_start <= int(r["gpos"]) < real_end for r in out):  # belt-and-braces
                print(f"  [{ikey} {real_start}-{real_end}] {len(out)} rows (cached)", flush=True)
                return (ikey, out)
        except Exception:
            pass  # corrupt/partial/out-of-range → regenerate

    db = get_db(db_path)
    full = {tf: bsp._load(db, symbol, tf) for tf in bsp._TFS}
    exec_df = full.get(exec_tf)
    if exec_df is None or exec_df.empty:
        print(f"  [{ikey}] no {exec_tf} candles for {symbol} — skipped", flush=True)
        return (ikey, [])

    runner = SequenceRunner(cfg, execution_tf=exec_tf, account_balance=10000.0,
                            tradeable_grades=("A+", "A", "B"))
    feed_start = max(window, real_start - warmup)
    out = []
    for gpos in range(feed_start, real_end):
        ts = exec_df.index[gpos]
        hist = {tf: bsp._win(df, ts, window) for tf, df in full.items() if not df.empty}
        bar = {"timestamp": ts.to_pydatetime(), "bar_index": gpos, "symbol": symbol}
        # one bad candle must not kill a multi-hour run — skip the bar, keep the chunk.
        try:
            sig = runner.on_bar(bar, hist)
        except Exception as exc:
            print(f"  [{ikey}] on_bar error @gpos={gpos}: {type(exc).__name__}: {exc} — skip bar", flush=True)
            continue
        if sig is None or gpos < real_start:
            continue
        # captured zones are still on the runner right after an approved emit (reset
        # only happens on rejection or cooldown-complete) → richest geometry features.
        try:
            feats = extract_features(
                sig, history=hist, config=cfg,
                sweep=runner._captured.get("sweep"), fvg=runner._captured.get("fvg"),
                exec_tf=exec_tf,
            )
        except Exception as exc:
            print(f"  [{ikey}] feature error @gpos={gpos}: {type(exc).__name__}: {exc} — skip", flush=True)
            continue
        row = {
            "instrument": ikey, "symbol": symbol, "setup_id": sig.setup_id,
            "gpos": gpos, "ts": ts.isoformat(), "direction": sig.direction,
            "entry": sig.entry, "sl": sig.sl, "tp1": sig.tp1,
            "tp2": sig.tp2 if sig.tp2 is not None else sig.tp1, "grade": sig.grade,
        }
        row.update(feats)
        out.append(row)

    os.makedirs(out_dir, exist_ok=True)
    with open(ckpt, "w") as f:
        json.dump(out, f)
    print(f"  [{ikey} {real_start}-{real_end}] {len(out)} rows ({time.time()-t0:.0f}s)", flush=True)
    return (ikey, out)


def _label_rows(rows: list, exec_df: pd.DataFrame, cost_overrides, exec_tf: str,
                entry_expiry_bars: int, max_hold_bars: int, want_net_r: bool) -> list:
    """LABEL pass over the FULL exec df (so future bars cross chunk boundaries safely).

    For each setup at global index gpos, future bars = exec_df.iloc[gpos+1:]. Computes
    the conservative tp1_before_sl binary label, and optionally the cost-aware net_r.
    """
    highs = exec_df["high"].to_numpy()
    lows = exec_df["low"].to_numpy()
    n = len(exec_df)
    mh = None if max_hold_bars <= 0 else max_hold_bars
    labeled = []
    for r in rows:
        try:
            gpos = int(r["gpos"])
            fh = list(highs[gpos + 1:n])
            fl = list(lows[gpos + 1:n])
            lab = L.label_binary(r["direction"], float(r["entry"]), float(r["sl"]),
                                 float(r["tp1"]), fh, fl,
                                 entry_expiry_bars=entry_expiry_bars, max_hold_bars=mh)
            r["triggered"] = lab["triggered"]
            r["outcome"] = lab["outcome"]
            r["tp1_before_sl"] = lab["tp1_before_sl"]
            r["win_r"] = lab["win_r"]
            r["fill_offset"] = lab["fill_offset"]
            r["resolve_offset"] = lab["resolve_offset"]
            r["net_r"] = None
            r["net_r_exit_type"] = None
            if want_net_r and lab["triggered"]:
                sig_dict = {"setup_id": r["setup_id"], "direction": r["direction"],
                            "entry": r["entry"], "sl": r["sl"], "tp1": r["tp1"],
                            "tp2": r["tp2"], "grade": r["grade"], "execution_tf": exec_tf}
                # SAME entry-trigger window as the binary label so they agree on 'triggered'.
                nr = L.label_net_r(sig_dict, exec_df.iloc[gpos:], cost_overrides=cost_overrides,
                                   entry_expiry_bars=entry_expiry_bars)
                r["net_r"] = nr["net_r"]
                r["net_r_exit_type"] = nr["exit_type"]
            labeled.append(r)
        except Exception as exc:
            print(f"  label error setup={r.get('setup_id')}: {type(exc).__name__}: {exc} — skip", flush=True)
    return labeled


def _write_csv(rows: list, path: Path) -> None:
    cols = _META_COLS + feature_names() + _LABEL_COLS
    df = pd.DataFrame(rows)
    for c in cols:  # guarantee a stable, complete header even on empty/partial runs
        if c not in df.columns:
            df[c] = pd.NA
    df = df[cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {len(df)} rows → {path}", flush=True)


def _verify(plan, exec_tf, warmup, window, jobs, db_path, out_dir):
    """Prove chunked GEN == sequential GEN (signals identical) before a big run."""
    ikey, symbol, cfg, _ov = plan[0]
    db = get_db(db_path)
    exec_df = bsp._load(db, symbol, exec_tf)
    n = len(exec_df)
    if n == 0:
        print(f"VERIFY: no candles for {symbol} — cannot verify."); return
    rs = max(window, n - 300)
    rt = _run_tag(cfg, exec_tf, warmup, window)
    one = _gen_ml_chunk((ikey, symbol, cfg, rs, n, warmup, window, exec_tf, db_path, out_dir + "_v1", rt))[1]
    tasks = [(ikey, symbol, cfg, cs, ce, warmup, window, exec_tf, db_path, out_dir + "_v3", rt)
             for cs, ce in bsp._chunks(rs, n, 100)]
    with Pool(jobs) as pool:
        many = [r for _l, rs_ in pool.map(_gen_ml_chunk, tasks) for r in rs_]
    k = lambda r: (r["gpos"], round(float(r["entry"]), 2), r["direction"])
    s1, s3 = sorted(map(k, one)), sorted(map(k, many))
    print(f"\n=== VERIFY ({ikey}) ===\n  1 chunk: {len(s1)} | chunked: {len(s3)}")
    print(f"  IDENTICAL: {s1 == s3}  {'✅ safe for the big run' if s1 == s3 else '🔴 MISMATCH'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build the Phase 12 ML training dataset.")
    p.add_argument("--instruments", default="crypto",
                   help="'gold' | 'crypto' | 'all' | a comma-separated symbol list.")
    p.add_argument("--execution-tf", default="5m")
    p.add_argument("--total-bars", type=int, default=0, help="0 = use all available bars.")
    p.add_argument("--chunk-bars", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=450)
    p.add_argument("--window", type=int, default=350)
    p.add_argument("--jobs", type=int, default=_default_jobs(),
                   help="Parallel workers. On RunPod set to the pod's vCPU count.")
    p.add_argument("--entry-expiry-bars", type=int, default=L.DEFAULT_ENTRY_EXPIRY_BARS)
    p.add_argument("--max-hold-bars", type=int, default=0, help="0 = resolve to end of data.")
    p.add_argument("--net-r", action="store_true", help="Also compute the cost-aware net_r label (slower).")
    p.add_argument("--db-path", default=DB_DEFAULT)
    p.add_argument("--out-dir", default=OUT_DEFAULT,
                   help="Checkpoints + CSVs. Put on a persistent volume on RunPod.")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()

    plan = _instrument_plan(a.instruments)
    if not plan:
        print("No instruments resolved — check --instruments."); return
    os.makedirs(a.out_dir, exist_ok=True)

    if a.verify:
        _verify(plan, a.execution_tf, a.warmup, a.window, a.jobs, a.db_path, a.out_dir)
        return

    print(f"=== build ML dataset | instruments={[k for k,_s,_c,_o in plan]} | "
          f"exec_tf={a.execution_tf} | jobs={a.jobs} | net_r={a.net_r} ===", flush=True)

    # --- GEN phase: fan every instrument's chunks across the pool together ---
    tasks, db = [], get_db(a.db_path)
    spans, skipped = {}, []  # ikey -> (symbol, cfg, cost_overrides, exec_df) ; skipped ikeys
    for ikey, symbol, cfg, ov in plan:
        exec_df = bsp._load(db, symbol, a.execution_tf)
        if exec_df.empty:
            # LOUD, not silent: crypto ships committed CSVs; gold needs the DB/CSV staged
            # on the pod. An --instruments all run must not quietly produce a gold-less set.
            print(f"  ⚠️  [{ikey}] NO {a.execution_tf} candles for {symbol} — SKIPPED "
                  f"(crypto ships committed CSVs; gold needs the SQLite DB or a fetched CSV)",
                  flush=True)
            skipped.append(ikey)
            continue
        n = len(exec_df)
        range_start = max(a.window, n - a.total_bars) if a.total_bars > 0 else a.window
        run_tag = _run_tag(cfg, a.execution_tf, a.warmup, a.window)
        spans[ikey] = (symbol, cfg, ov, exec_df)
        for cs, ce in bsp._chunks(range_start, n, a.chunk_bars):
            tasks.append((ikey, symbol, cfg, cs, ce, a.warmup, a.window,
                          a.execution_tf, a.db_path, a.out_dir, run_tag))
    if not tasks:
        print("🔴 No candle data found for ANY requested instrument — nothing to do.")
        sys.exit(1)

    t0 = time.time()
    with Pool(a.jobs) as pool:
        results = pool.map(_gen_ml_chunk, tasks)
    print(f"\n  GEN done in {time.time()-t0:.0f}s ({len(tasks)} chunks)", flush=True)

    by = {ikey: [] for ikey in spans}
    for ikey, rows in results:
        if ikey in by:
            by[ikey].extend(rows)

    # --- LABEL phase + write ---
    all_rows = []
    for ikey, (symbol, cfg, ov, exec_df) in spans.items():
        rows = sorted(by[ikey], key=lambda r: r["gpos"])
        rows = _label_rows(rows, exec_df, ov, a.execution_tf,
                           a.entry_expiry_bars, a.max_hold_bars, a.net_r)
        _write_csv(rows, Path(a.out_dir) / f"ml_dataset_{ikey}.csv")
        all_rows.extend(rows)

    if all_rows:
        _write_csv(all_rows, Path(a.out_dir) / "ml_dataset_all.csv")
        # quick label sanity so the user sees if there's enough signal to train on
        triggered = [r for r in all_rows if r.get("triggered")]
        resolved = [r for r in triggered if r.get("tp1_before_sl") is not None]
        wins = sum(1 for r in resolved if r["tp1_before_sl"] == 1)
        wr = (100.0 * wins / len(resolved)) if resolved else 0.0
        print(f"\n  === dataset summary ===")
        print(f"  total setups: {len(all_rows)} | triggered: {len(triggered)} | "
              f"resolved (labelable): {len(resolved)} | win-rate: {wr:.1f}%")
        print(f"  NOTE: train on the {len(resolved)} RESOLVED rows; "
              f"drop NO_FILL/OPEN (untriggered/censored).")

    # Fail LOUD (non-zero exit) if any REQUESTED instrument was skipped for lack of data,
    # so an `--instruments all` run can't quietly hand back an incomplete (e.g. gold-less)
    # dataset that looks complete.
    if skipped:
        print(f"\n⚠️  INCOMPLETE: {len(skipped)} requested instrument(s) had NO data and "
              f"were SKIPPED → {skipped}. Stage their candles and re-run, or use "
              f"--instruments crypto (the committed, reproducible set).", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
