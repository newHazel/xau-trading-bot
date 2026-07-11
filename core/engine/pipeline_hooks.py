"""
Pipeline stage hooks — SKELETON wiring the existing detectors into SignalPipeline.

Each `make_*_hook(...)` factory instantiates the real components ONCE and returns
a stage hook `(ctx, bar, history) -> None` that runs them and writes results onto
the PipelineContext. This is the layer that connects Phases 1–11 to the pipeline.

CONTRACT for `history` (what the caller passes to SignalPipeline.process_bar):
    history = {
        "4h": <pd.DataFrame>,   # UTC DatetimeIndex + open/high/low/close/volume
        "1h": <pd.DataFrame>,
        "15m": <pd.DataFrame>,
        "5m": <pd.DataFrame>,   # execution TF (or "1m" in overlap)
        "1m": <pd.DataFrame>,
        "dxy": <pd.DataFrame>,  # optional, for the DXY filter
    }
Each hook reads only the timeframes it needs and is defensive: if a timeframe is
missing or too short, it leaves the relevant ctx fields at their defaults (so an
unfinished wiring produces no signal rather than crashing).

Everything marked `# TODO(user):` needs your judgment — these are the spots where
the skeleton intentionally stops short of a decision (config values, multi-TF
orchestration choices, confirmation-window logic, ATR source, account balance).

No heavy computation is triggered at import time; detectors run only when a hook
is invoked on real data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from core.engine.signal_pipeline import PipelineContext, StageHook


# ====================================================================== #
# Shared helpers                                                          #
# ====================================================================== #

# structure layer speaks bullish/bearish; the rulebook speaks long/short
_BIAS_TO_DIR = {"bullish": "long", "bearish": "short", "neutral": "neutral"}


def _tf(history: Any, name: str) -> Optional[pd.DataFrame]:
    """Safely pull a timeframe DataFrame from the history dict."""
    if not isinstance(history, dict):
        return None
    df = history.get(name)
    if df is None or not hasattr(df, "empty") or df.empty:
        return None
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder ATR of the most recent `period` bars. Lightweight (no GPU)."""
    if df is None or len(df) < period + 1:
        return 0.0
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder ATR (RMA of TR) — same smoothing compute_rsi already uses, not a plain
    # SMA of the last `period` true ranges. The old tr.tail(period).mean() was an SMA
    # despite the "Wilder" docstring, inconsistent with the RSI calc.
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return float(atr.iloc[-1])


def atr_series(df: pd.DataFrame, period: int = 14, tail: int = 50) -> list:
    """Rolling-ATR series (last `tail` values) for the VolatilityFilter.

    The filter compares the latest ATR to the median of the series, so we hand
    it a window of recent rolling-ATR readings. Lightweight / vectorized.
    """
    if df is None or len(df) < period + 2:
        return []
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder ATR series (RMA of TR), consistent with compute_atr above — not a plain
    # rolling SMA. The VolatilityFilter compares the latest reading to the median.
    roll = tr.ewm(alpha=1.0 / period, adjust=False).mean().dropna()
    return [float(x) for x in roll.tail(tail)]


def compute_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder RSI of the latest bar (0-100). Used by the momentum-confirmation gate:
    a long taken while RSI is still very low = catching a falling knife. Lightweight."""
    if df is None or len(df) < period + 1:
        return 50.0
    close = df["close"].astype(float)
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_dn = float(dn.iloc[-1])
    if last_dn == 0:
        return 100.0
    rs = float(up.iloc[-1]) / last_dn
    return 100.0 - 100.0 / (1.0 + rs)


def compute_macd_alignment(df: pd.DataFrame, is_long: bool,
                           fast: int = 12, slow: int = 26, signal: int = 9) -> bool:
    """Pine-v4 trigger alignment: MACD histogram momentum in the trade direction AND
    the macd line on the right side of its signal (or crossing it this bar).
    Warm-up (<2 bars of hist) never blocks."""
    close = df["close"].astype(float)
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    if len(hist) < 2:
        return True
    h0, h1 = float(hist.iloc[-1]), float(hist.iloc[-2])
    m0, s0 = float(macd.iloc[-1]), float(sig.iloc[-1])
    m1, s1 = float(macd.iloc[-2]), float(sig.iloc[-2])
    if is_long:
        return h0 > h1 and (m0 > s0 or (m1 <= s1 and m0 > s0))
    return h0 < h1 and (m0 < s0 or (m1 >= s1 and m0 < s0))


def near_any_ob(price: float, blocks: list, tol: float) -> bool:
    """True when price sits inside any OB zone widened by tol (Pine htfPoiATR)."""
    return any(b["bottom"] - tol <= price <= b["top"] + tol for b in blocks)


def compute_ema_bias(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> str:
    """Execution-TF trend bias from EMA50/EMA200: 'long' if price>EMA200 AND EMA50>EMA200,
    'short' if price<EMA200 AND EMA50<EMA200, else 'neutral'. Used by the trend_gate to
    block COUNTER-TREND entries (the longs that bled in a down regime). Lightweight."""
    if df is None or len(df) < slow:
        return "neutral"
    close = df["close"].astype(float)
    ema_f = float(close.ewm(span=fast, adjust=False).mean().iloc[-1])
    ema_s = float(close.ewm(span=slow, adjust=False).mean().iloc[-1])
    price = float(close.iloc[-1])
    if price > ema_s and ema_f > ema_s:
        return "long"
    if price < ema_s and ema_f < ema_s:
        return "short"
    return "neutral"


def rejection_confirms(o: float, h: float, l: float, c: float,
                       fvg_lo: float, fvg_hi: float, is_long: bool,
                       atr: float, min_body_atr: float = 0.3) -> bool:
    """A REAL confirmation candle: a decisive REJECTION at the POI (vs the weak body-color
    gate). LONG: the candle dipped to the proximal FVG edge (fvg_hi) and CLOSED back ABOVE
    it, with a body >= min_body_atr*ATR. SHORT mirror: wicked up to fvg_lo, closed BELOW it.
    A bounce/breakout that closes on the wrong side of the proximal edge does NOT confirm
    (the ETH 11:50 short: its fill candle closed ABOVE the entry → not a rejection)."""
    body = abs(c - o)
    strong = (body >= min_body_atr * atr) if atr > 0 else (body > 0)
    if is_long:
        reject = (c > o) and (l <= fvg_hi) and (c >= fvg_hi)
    else:
        reject = (c < o) and (h >= fvg_lo) and (c <= fvg_lo)
    return bool(strong and reject)


def _collect_levels(liq_df: pd.DataFrame, swing_df: pd.DataFrame, smc_dir: str) -> list:
    """Build a [{price, type}] list of liquidity pools for the TP-target finder.

    Pulls the most recent EQH/EQL/PDH/PDL levels plus the last few swing
    highs/lows. The target finder filters these by direction relative to entry.
    """
    levels: list = []
    for col, typ in [("eqh_level", "eqh"), ("eql_level", "eql"),
                     ("pdh", "pdh"), ("pdl", "pdl")]:
        if col in liq_df.columns:
            s = liq_df[col].dropna()
            if not s.empty:
                levels.append({"price": float(s.iloc[-1]), "type": typ})
    for col, typ in [("swing_high", "swing_high"), ("swing_low", "swing_low")]:
        if col in swing_df.columns:
            s = swing_df[col].dropna()
            for v in s.iloc[-3:]:
                levels.append({"price": float(v), "type": typ})
    return levels


# ====================================================================== #
# STAGE 1 — structure (4H macro bias + 1H permission + premium/discount)  #
# ====================================================================== #

def select_fvg(candidates: list, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict]:
    """Pick the best tradeable FVG zone for entry.

    Legacy (fvg_freshness_enabled=false): returns the newest matching zone
    (candidates are newest-first), preserving the original behavior exactly.

    Enabled (default): scores every candidate by PROXIMITY (distance from the
    current price, in ATR units) and FRESHNESS (bars since the gap formed), and
    picks the best. Zones beyond the distance/age caps are heavily PENALIZED but
    never discarded — so a valid setup is still taken when nothing fresher/nearer
    exists ("don't give up on the old, just prefer the fresh"). This stops the
    engine pinning a stale prior-day gap far from price and waiting forever for an
    impossible retrace (the 2026-06-10 ~4180 short it missed)."""
    if not candidates:
        return None
    if not bool(config.get("fvg_freshness_enabled", True)):
        return candidates[0]  # legacy: newest-first

    price = float(df["close"].iloc[-1])
    atr = compute_atr(df, 14)
    if not atr or atr != atr:  # 0.0 or NaN guard
        rng = float((df["high"].astype(float) - df["low"].astype(float)).tail(14).mean())
        atr = rng if rng and rng == rng else 1.0

    w_dist = float(config.get("fvg_distance_weight", 1.0))
    w_age = float(config.get("fvg_age_weight", 0.5))
    max_dist_atr = float(config.get("fvg_max_distance_atr", 4.0))
    max_age_bars = int(config.get("fvg_max_age_bars", 120))
    OVER = 100.0  # soft penalty for breaching a cap: deprioritize, never exclude

    def _near_dist(u: Dict) -> float:
        lo = min(u["top"], u["bottom"]); hi = max(u["top"], u["bottom"])
        if price < lo:
            return lo - price
        if price > hi:
            return price - hi
        return 0.0  # price already inside the zone

    def _age_bars(u: Dict) -> int:
        ts = u.get("confirm_ts")
        if ts is None or ts != ts:  # None or NaT
            return 0
        try:
            return int((df.index > ts).sum())
        except (TypeError, ValueError):  # tz mismatch / bad ts → distance-only score
            return 0

    def _score(u: Dict) -> float:
        dist_atr = _near_dist(u) / atr if atr > 0 else _near_dist(u)
        age = _age_bars(u)
        pen = w_dist * dist_atr + w_age * (age / max(max_age_bars, 1))
        if dist_atr > max_dist_atr:
            pen += OVER
        if age > max_age_bars:
            pen += OVER
        return pen

    return min(candidates, key=_score)


def make_structure_hook(config: Optional[Dict[str, Any]] = None,
                        htf: str = "4h", mtf: str = "1h") -> StageHook:
    config = config or {}
    from core.structure.swing_detector import SwingDetector
    from core.structure.market_structure import MarketStructure
    from core.structure.premium_discount import PremiumDiscountAnalyzer
    from core.structure.bias_conflict_resolver import HTFConflictResolver

    swings = SwingDetector(config.get("fractal_windows"))
    structure = MarketStructure(relaxed_bias=config.get("relaxed_structure_bias", False))
    pd_zone = PremiumDiscountAnalyzer()
    resolver = HTFConflictResolver()

    def hook(ctx: PipelineContext, bar: Any, history: Any) -> None:
        df_htf = _tf(history, htf)
        df_mtf = _tf(history, mtf)
        if df_htf is None or df_mtf is None:
            return

        # 4H + 1H bias
        htf_swings = swings.detect(df_htf, htf)
        htf_struct = structure.classify(htf_swings)
        bias_4h = structure.get_current_bias(htf_struct)

        mtf_swings = swings.detect(df_mtf, mtf)
        mtf_struct = structure.classify(mtf_swings)
        bias_1h = structure.get_current_bias(mtf_struct)

        # The structure layer speaks "bullish"/"bearish"; the rulebook speaks
        # "long"/"short". Translate so htf_bias / 15m_aligned can actually match.
        combined = resolver.combine(bias_4h, bias_1h)   # bullish/bearish/neutral
        combined_dir = _BIAS_TO_DIR.get(combined, "neutral")
        bias_1h_dir = _BIAS_TO_DIR.get(bias_1h, "neutral")

        # htf_bias gate. Default: the 1H bias ALONE (the diagnostic showed the
        # 4H+1H-consensus requirement was the #1 blocker at ~80%). The 4H still
        # acts as confluence toward A+ via the rest of the score. Set
        # require_4h_agreement=true to restore the strict 4H+1H consensus mode.
        require_4h = config.get("require_4h_agreement", False)
        gate_dir = combined_dir if require_4h else bias_1h_dir
        ctx.htf_bias = gate_dir
        if gate_dir in ("long", "short"):
            ctx.direction = gate_dir
        # record whether 4H agrees, for downstream confluence/booster use
        ctx.extra["htf_4h_aligned"] = (combined_dir == bias_1h_dir and combined_dir in ("long", "short"))

        # 15m alignment + premium/discount zone.
        ctx.structure_15m = bias_1h_dir if bias_1h_dir in ("long", "short") else None
        # price_zone: default on the 1H (MTF) dealing range. Opt-in (backtest-gated,
        # default OFF → live unchanged): compute it on the 15m execution range the
        # entries actually trade. The coarser 1H range flips halves slowly and can both
        # over-reject (15m already in discount but 1H still premium) and under-reject
        # relative to a true 15m zone — the price_zone gate is mandatory, so this
        # mislabels which setups survive (audit finding). Backtest to pick.
        if config.get("price_zone_on_15m", False):
            df_15m = _tf(history, "15m")
            if df_15m is not None and len(df_15m) >= 30:
                sw_15m = swings.detect(df_15m, "15m")
                ctx.price_zone = pd_zone.get_current_zone(pd_zone.analyze(sw_15m))
            else:
                ctx.price_zone = pd_zone.get_current_zone(pd_zone.analyze(mtf_swings))
        else:
            pd_df = pd_zone.analyze(mtf_swings)
            ctx.price_zone = pd_zone.get_current_zone(pd_df)  # "premium"/"discount"/"equilibrium"

    return hook


# ====================================================================== #
# STAGE 2 — SMC (liquidity sweep → FVG → order block, confirmations)      #
# ====================================================================== #

def make_smc_hook(config: Optional[Dict[str, Any]] = None,
                  execution_tf: str = "5m") -> StageHook:
    config = config or {}
    from core.structure.swing_detector import SwingDetector
    from core.structure.market_structure import MarketStructure
    from core.structure.choch_detector import CHoCHDetector
    from core.smc.liquidity_detector import LiquidityDetector
    from core.smc.sweep_detector import SweepDetector
    from core.smc.fvg_detector import FVGDetector
    from core.smc.order_block_detector import OrderBlockDetector
    from core.smc.displacement_detector import DisplacementDetector
    from core.smc.mitigation_tracker import MitigationTracker

    swings = SwingDetector(config.get("fractal_windows"))
    structure = MarketStructure(relaxed_bias=config.get("relaxed_structure_bias", False))
    choch = CHoCHDetector()
    liquidity = LiquidityDetector()
    obs = OrderBlockDetector()
    mitigation = MitigationTracker()
    # Detectors. Default construction = current live behavior (their built-in defaults
    # already match smc_rules.yaml EXCEPT displacement break_lookback: code=3, YAML=5).
    # Opt-in (default OFF → live unchanged): make smc_rules.yaml AUTHORITATIVE so
    # fvg_min_atr_ratio / sweep window / displacement thresholds become real, tunable
    # count levers — previously these YAML keys were read by NOTHING (audit finding).
    # Enabling this also applies YAML break_lookback (3→5, a stricter grade-only
    # displacement booster), so re-baseline the backtest after flipping it on.
    _swp_cfg = config.get("sweep", {}) or {}
    # Sweep-quality gate (default OFF -> live unchanged): honor the previously-dead
    # smc_rules.yaml key min_wick_penetration_atr_multiplier — a wick must penetrate
    # the level by >= mult x prior ATR to register as a sweep (Pine-v3.2 parity).
    _pen_mult = (float(_swp_cfg.get("min_wick_penetration_atr_multiplier", 0.05))
                 if config.get("sweep_min_penetration_enabled", False) else 0.0)
    if config.get("wire_detector_config", False):
        _swp = _swp_cfg
        _dsp = config.get("displacement", {}) or {}
        sweeps = SweepDetector(window=int(_swp.get("confirmation_window_candles", 5)),
                               min_penetration_atr_mult=_pen_mult)
        fvgs = FVGDetector(
            atr_period=int(_dsp.get("atr_period", 14)),
            size_threshold_atr_pct=float(config.get("fvg_min_atr_ratio", 0.3)),
        )
        displacement = DisplacementDetector(
            body_atr_threshold=float(_dsp.get("min_body_atr_multiplier", 1.2)),
            body_range_threshold=float(_dsp.get("min_body_to_range_ratio", 0.60)),
            break_lookback=int(_dsp.get("breakout_lookback_candles", 3)),
            atr_period=int(_dsp.get("atr_period", 14)),
        )
    else:
        sweeps = SweepDetector(min_penetration_atr_mult=_pen_mult)
        fvgs = FVGDetector()
        displacement = DisplacementDetector()
    # how recently (in execution-TF bars) a sweep must have fired to count
    recency = int(config.get("trigger_recency_bars", 10))
    # sweep_early lever (default OFF → live unchanged): arm on the PROVISIONAL wick
    # (before the close-back) when no confirmed sweep is available, gated by a breakout
    # guard. Only the backtest 'crypto_sweep' variant flips this on.
    sweep_early = bool(config.get("sweep_early", False))
    sweep_early_max_wick_atr = float(config.get("sweep_early_max_wick_atr", 0.5))
    # CHoCH confirms slower than a sweep — give it a wider window
    choch_recency = int(config.get("choch_recency_bars", 20))
    # the "micro" CHoCH lives on the timeframe BELOW execution (doc: "CHoCH on 5m")
    _ltf_below = {"4h": "1h", "1h": "15m", "15m": "5m", "5m": "1m", "1m": "1m"}
    micro_tf = _ltf_below.get(execution_tf, execution_tf)

    def hook(ctx: PipelineContext, bar: Any, history: Any) -> None:
        df = _tf(history, execution_tf)
        if df is None or len(df) < 30:
            return

        n = len(df)
        last_pos = n - 1
        is_long = ctx.direction == "long"
        smc_dir = "bull" if is_long else "bear"

        prepared = swings.detect(df, execution_tf)

        # --- liquidity sweep (needs swing + liquidity columns) ---
        liq_df = liquidity.detect(prepared)
        sweep_df = sweeps.detect(liq_df)
        last_sweep = sweeps.get_last_sweep(sweep_df, direction=smc_dir)
        ctx.sweep = last_sweep
        # Confirmed if the sweep fired within the recency window (close already back).
        ctx.sweep_confirmed = (
            last_sweep is not None and (last_pos - last_sweep["confirm_pos"]) <= recency
        )
        # sweep_early fallback: if no CONFIRMED sweep armed the step, optionally arm on
        # the freshest PROVISIONAL wick — but only if it's a rejection wick (real liquidity
        # grab), not a breakout. The captured level is the genuine swept level, so SL/entry
        # stay correct. Default OFF (sweep_early False) → this whole branch is skipped.
        if sweep_early and not ctx.sweep_confirmed:
            pend = sweeps.get_last_pending_sweep(sweep_df, direction=smc_dir)
            if pend is not None and (last_pos - pend["confirm_pos"]) <= recency:
                lvl = pend["level"]
                ext = pend["wick_extreme"]   # the pierce extreme (high for bear, low for bull)
                wc = pend["wick_close"]       # close on the wick bar (still BEYOND lvl — unconfirmed)
                pen = (ext - lvl) if smc_dir == "bear" else (lvl - ext)
                if pen > 0:
                    # (1) REJECTION: the wick bar gave back >= half its overshoot (closed in the
                    #     half nearer the level). A breakout marubozu closes near the extreme;
                    #     a stop-grab rejects back toward the level. (For a PENDING sweep the close
                    #     is still beyond the level — it hasn't fully closed back yet — so we test
                    #     the give-back fraction, not close-vs-level which is always beyond here.)
                    mid = (lvl + ext) / 2.0
                    reverted = (wc <= mid) if smc_dir == "bear" else (wc >= mid)
                    # (2) PENETRATION CAP: a large thrust past the level = conviction/continuation,
                    #     not a stop-grab. Skip if ATR isn't available yet.
                    atr = compute_atr(df)
                    pen_ok = (atr <= 0) or (pen <= sweep_early_max_wick_atr * atr)
                    if reverted and pen_ok:
                        ctx.sweep = pend
                        ctx.sweep_confirmed = True

        # --- micro-CHoCH: a recent change of character in the trade direction, on
        #     the execution TF OR the lower "micro" TF (doc: "CHoCH on 5m"). Either. ---
        structured = structure.classify(prepared)
        c_exec = choch.get_last_choch(choch.detect(structured), direction=smc_dir)
        micro = (c_exec is not None and (last_pos - c_exec["confirm_pos"]) <= choch_recency)

        if not micro and micro_tf != execution_tf:
            ltf_df = _tf(history, micro_tf)
            if ltf_df is not None and len(ltf_df) >= 30:
                ltf_struct = structure.classify(swings.detect(ltf_df, micro_tf))
                c_ltf = choch.get_last_choch(choch.detect(ltf_struct), direction=smc_dir)
                # LTF has more bars per unit time → allow a proportionally wider window
                if c_ltf is not None and (len(ltf_df) - 1 - c_ltf["confirm_pos"]) <= choch_recency * 3:
                    micro = True
        ctx.micro_choch = micro

        # --- FVG: trade the most recent UNMITIGATED FVG of our direction ---
        # (get_last_fvg returns the latest FVG even if already filled; here we pick
        #  the freshest tradeable one so the setup zone is genuinely actionable.)
        fvg_df = fvgs.detect(df)
        mitig_df = mitigation.track(fvg_df)
        unmitigated = mitigation.get_unmitigated_fvgs(mitig_df, n=10)  # newest-first
        # get_unmitigated_fvgs returns {fresh, tapped, partial, deep} — every zone
        # that still has gap left. We deliberately trade the STRICTER subset here
        # and exclude 'deep' (fill 0.50–0.80): that matches the LIVE gate in
        # config/mitigation_rules.yaml (max_allowed_fill_percent_for_live: 0.50 ==
        # partial_max; 'deep' only fits the 0.80 paper cap), and agrees with
        # zone_lifecycle_manager, which classifies 'deep' as 'mitigated'. Adding
        # 'deep' would loosen the entry zone and INCREASE signal frequency — a
        # behavior change to backtest first, not a behavior-preserving cleanup.
        tradeable_states = {"fresh", "tapped", "partial"}  # excludes 'deep' by design
        candidates = [u for u in unmitigated
                      if u["fvg_type"] == smc_dir and u["state"] in tradeable_states]
        # Prefer the freshest + nearest zone (config: fvg_freshness_enabled);
        # falls back to older/further zones rather than discarding them.
        chosen_fvg = select_fvg(candidates, df, config)
        ctx.fvg = chosen_fvg
        ctx.fvg_valid = chosen_fvg is not None
        # mandatory fvg_freshness: the chosen zone is in the tradeable freshness set
        # (fresh/tapped/partial). Behaviour-equivalent to `chosen_fvg is not None` today
        # (the candidate filter already excludes deep/mitigated), but reads the zone STATE
        # explicitly instead of being a blind alias of fvg_valid — so it correctly rejects
        # if an over-mitigated zone ever reaches here. (Genuine 'fresh' = optional booster.)
        ctx.fvg_fresh = bool(isinstance(chosen_fvg, dict)
                             and chosen_fvg.get("state") in tradeable_states)
        # (The optional fvg_fresh BOOSTER is derived separately from the TRADED zone's state
        #  in SignalPipeline._build_optional — genuine 'fresh' vs 'tapped'/'partial' — bug #11.)
        if chosen_fvg is not None:
            # --- retrace into the FVG zone + confirmation candle ---
            # NOTE: ctx.retraced_to_zone here is computed against the FRESHLY
            # re-selected FVG. It is authoritative only on the NON-PINNED path
            # (SignalPipeline.process_bar, which has no setup capture). Under
            # SequenceRunner, once an FVG is captured the runner OVERWRITES
            # ctx.retraced_to_zone against the PINNED zone (sequence_runner.on_bar,
            # ~L110-119), so on later bars this line's result is superseded. Keep
            # it: the non-pinned path depends on it, and on the bar the FVG is
            # first captured it equals the pinned value (same zone).
            lo = min(chosen_fvg["top"], chosen_fvg["bottom"])
            hi = max(chosen_fvg["top"], chosen_fvg["bottom"])
            recent_low = float(df["low"].iloc[-3:].min())
            recent_high = float(df["high"].iloc[-3:].max())
            ctx.retraced_to_zone = (recent_low <= hi and recent_high >= lo)

            o = float(df["open"].iloc[-1])
            c = float(df["close"].iloc[-1])
            # confirmation: DEFAULT = body color (green=long / red=short — the known-weak gate).
            # With confirm_gate ON (default OFF → live unchanged): require a real REJECTION
            # candle at the POI — a decisive body (>= k*ATR) that closed back THROUGH the
            # proximal FVG edge (reclaimed the level). Targets the entry-quality root cause:
            # a weak green/red 'confirmation' let the bot sell into a bounce that never rejected
            # (the ETH 11:50 short: the fill candle CLOSED above the entry → would be blocked here).
            if config.get("confirm_gate", False):
                ctx.confirmation_candle = rejection_confirms(
                    o, float(df["high"].iloc[-1]), float(df["low"].iloc[-1]), c,
                    lo, hi, is_long, compute_atr(df),
                    float(config.get("confirm_min_body_atr", 0.3)))
            else:
                ctx.confirmation_candle = (c > o) if is_long else (c < o)

        # --- order block (needs FVG/BOS columns) ---
        ob_df = obs.detect(fvg_df)
        order_blocks = obs.get_order_blocks(ob_df) if hasattr(obs, "get_order_blocks") else None
        ctx.order_block = order_blocks
        ctx.ob_valid = bool(order_blocks)

        # --- displacement strength ---
        disp_df = displacement.detect(df)
        if hasattr(displacement, "get_last_displacement"):
            # Pass direction=smc_dir: get_last_displacement defaults to "bull", so for a
            # SHORT setup the old call returned bull displacements and strong_displacement
            # was always False. Recency reads the real key 'confirm_pos' — the old code read
            # 'pos' (absent) so `"pos" in last_disp` was False → in_window always True.
            last_disp = displacement.get_last_displacement(disp_df, direction=smc_dir)
            if isinstance(last_disp, dict):
                pos = int(last_disp.get("confirm_pos", last_pos))
                in_window = (last_pos - pos) <= recency
                ctx.strong_displacement = (last_disp.get("type") == smc_dir) and in_window

        # --- volume confirmation: a volume spike on the latest candle (optional booster) ---
        vol = df["volume"].astype(float)
        if len(vol) >= 21 and vol.iloc[-21:-1].mean() > 0:
            spike_ratio = float(config.get("volume_spike_ratio", 1.3))
            ctx.volume_confirmation = bool(vol.iloc[-1] >= vol.iloc[-21:-1].mean() * spike_ratio)

        # hand liquidity pools to the risk stage (for TP2 target search)
        ctx.extra["liquidity_levels"] = _collect_levels(liq_df, prepared, smc_dir)

        # --- momentum-confirmation gate (momentum_gate, default OFF) ---
        # Don't take an entry while momentum is still AGAINST it: a long entered at very
        # low RSI is catching a falling knife (the 2026-06-15 losers entered at RSI 32/39
        # and kept dropping; the winner was RSI 56). Only sets momentum_ok when the flag
        # is on — otherwise it stays absent and _emit treats it as True (live unchanged).
        if config.get("momentum_gate", False):
            rsi = compute_rsi(df, int(config.get("rsi_period", 14)))
            ctx.extra["rsi"] = rsi
            ctx.extra["momentum_ok"] = (rsi >= float(config.get("rsi_long_min", 45.0))) if is_long \
                else (rsi <= float(config.get("rsi_short_max", 55.0)))

        # --- trend gate (trend_gate, default OFF): block COUNTER-TREND entries ---
        # The backtest showed longs bleed in a down regime. Require the execution-TF
        # EMA50/EMA200 trend bias to AGREE with the trade direction (a 'neutral' bias is
        # allowed — only an OPPOSING trend blocks). Sets trend_ok; filter_hook folds it into
        # no_blocking_filters. Default OFF → trend_ok absent → live unchanged.
        if config.get("trend_gate", False):
            bias = compute_ema_bias(df, int(config.get("trend_ema_fast", 50)),
                                    int(config.get("trend_ema_slow", 200)))
            ctx.extra["trend_bias"] = bias
            ctx.extra["trend_ok"] = (bias != "short") if is_long else (bias != "long")

        # --- MACD trigger alignment (macd_gate, default OFF): Pine-v4 useMACD parity.
        # Histogram momentum + line-vs-signal must agree with the trade direction at
        # the trigger. filter_hook folds macd_ok into no_blocking_filters. ---
        if config.get("macd_gate", False):
            ctx.extra["macd_ok"] = compute_macd_alignment(df, is_long)

        # --- HTF Order-Block confluence (htf_ob_gate, default OFF): Pine-v4 htfPoiMode
        # 'Required'. The current price must sit inside/near (±htf_ob_atr_mult x ATR) a
        # recent SAME-direction 1H order block — entries floating far from any HTF POI
        # are blocked. filter_hook folds htf_ob_ok into no_blocking_filters. ---
        if config.get("htf_ob_gate", False):
            ok = True   # fail-open: if the HTF frame can't be scored, don't block
            htf_df = _tf(history, str(config.get("htf_ob_tf", "1h")))
            if htf_df is not None and len(htf_df) >= 30:
                try:
                    # OrderBlockDetector needs FVG (or BOS) trigger columns first —
                    # same chain the exec-TF path uses: obs.detect(fvgs.detect(df)).
                    ob_df = obs.detect(fvgs.detect(htf_df))
                    blocks = obs.get_order_blocks(ob_df, direction=smc_dir, n=3)
                    if blocks:
                        tol = float(config.get("htf_ob_atr_mult", 1.0)) * compute_atr(df)
                        ok = near_any_ob(float(df["close"].iloc[-1]), blocks, tol)
                    else:
                        ok = False   # no recent same-direction HTF OB → gate blocks
                except Exception:
                    ok = True        # detector hiccup on the HTF frame → fail-open
            ctx.extra["htf_ob_ok"] = ok

    return hook


# ====================================================================== #
# STAGE 3 — filters (session / news / DXY / market-state)                 #
# ====================================================================== #

def make_filter_hook(config: Optional[Dict[str, Any]] = None,
                     execution_tf: str = "5m") -> StageHook:
    config = config or {}
    from core.filters.session_filter import SessionFilter
    from core.filters.dxy_filter import DXYFilter
    from core.filters.news_filter import NewsFilter
    from core.filters.spread_filter import SpreadFilter
    from core.filters.volatility_filter import VolatilityFilter
    from core.filters.market_state_filter import MarketStateFilter
    from core.filters.correlation_spike_filter import CorrelationSpikeFilter

    session = SessionFilter(config.get("session", config))
    dxy = DXYFilter(config.get("dxy", config))
    news = NewsFilter(config.get("news", config))
    spread_f = SpreadFilter(config.get("spread", config))
    vol_f = VolatilityFilter(config.get("volatility", config))
    state_f = MarketStateFilter(config.get("market_state", config))
    corr_f = CorrelationSpikeFilter(config.get("correlation", config))

    # funding_filter (default OFF, ORTHOGONAL signal): block a FRESH trade on the
    # crowded perp side (long into crowded-long funding / short into crowded-short),
    # read from history["funding"]. Only the backtest crypto_funding variant turns it
    # on → live unchanged. Thesis: avoids squeeze-prone counter-trend longs that bled.
    funding_filter = bool(config.get("funding_filter", False))
    funding_window = int(config.get("funding_window", 90))
    funding_hi = float(config.get("funding_hi_pct", 0.80))
    funding_lo = float(config.get("funding_lo_pct", 0.20))

    # Load the news calendar once. Prefer an explicit CSV path; else the default.
    news_csv = config.get("news_csv_path")
    if news_csv:
        news.load_from_csv(news_csv)
    else:
        news.ensure_loaded()

    def hook(ctx: PipelineContext, bar: Any, history: Any) -> None:
        ts = ctx.timestamp
        if ts is None:
            return

        ctx.in_kill_zone = session.is_trade_allowed(ts)
        active = session.get_active_sessions(ts) if hasattr(session, "get_active_sessions") else []
        # overlap = two sessions active simultaneously (London + NY)
        ctx.overlap_session = len(active) >= 2

        # News block window around tiered events.
        ctx.news_clear = not news.is_blocked(ts)

        # --- the four "blocking" filters ---
        df = _tf(history, execution_tf)
        spread_ok = vol_ok = state_ok = corr_ok = True
        if df is not None:
            atr = compute_atr(df)
            # Tick data has no real spread column → use the configured default + ATR cap.
            # Pass price so "percent" cost mode can derive a price-proportional spread
            # (a $0.08 coin must not be judged by a gold-sized absolute spread).
            price = float(df["close"].iloc[-1])
            spread_ok = spread_f.is_trade_allowed(spread=None, atr=atr, price=price)
            vol_ok = vol_f.is_trade_allowed(atr_series(df))
            state_ok = state_f.is_trade_allowed(
                highs=df["high"].astype(float).tolist(),
                lows=df["low"].astype(float).tolist(),
                closes=df["close"].astype(float).tolist(),
                opens=df["open"].astype(float).tolist(),
                atr=atr,
            )
            # clean_market_state is an optional booster (trending/acceptable state).
            ctx.clean_market_state = state_ok

            dxy_df = _tf(history, "dxy")
            if dxy_df is not None:
                xau_closes = df["close"].astype(float).tolist()
                dxy_closes = dxy_df["close"].astype(float).tolist()
                corr_ok = not corr_f.is_spike(xau_closes, dxy_closes)

        # funding (orthogonal positioning signal): block the crowded perp side when the
        # flag is on AND the per-coin funding series is present in history. Default OFF →
        # funding_ok stays True and no_blocking_filters is unchanged.
        funding_ok = True
        if funding_filter:
            from core.data.funding_provider import funding_regime, funding_blocks
            fdf = history.get("funding") if isinstance(history, dict) else None
            regime, rate = funding_regime(fdf, ts, funding_window, funding_hi, funding_lo)
            ctx.extra["funding_regime"] = regime
            ctx.extra["funding_rate"] = rate
            funding_ok = not funding_blocks(regime, ctx.direction)

        # trend gate (set in smc_hook, which runs before this filter): block a
        # counter-trend entry. Absent (trend_gate OFF) → defaults True → unchanged.
        trend_ok = bool(ctx.extra.get("trend_ok", True))
        # Pine-v4 levers (absent when their flags are OFF -> default True -> unchanged)
        macd_ok = bool(ctx.extra.get("macd_ok", True))
        htf_ob_ok = bool(ctx.extra.get("htf_ob_ok", True))

        # no_blocking_filters covers the AUXILIARY filters only. kill_zone and
        # news_clear are their own separate mandatory conditions — don't double-count
        # them here (that would fail two mandatories at once on any off-session bar).
        ctx.no_blocking_filters = bool(spread_ok and vol_ok and state_ok and corr_ok
                                       and funding_ok and trend_ok and macd_ok and htf_ob_ok)

        # DXY alignment (optional booster). Needs a DXY DataFrame in history.
        dxy_df = _tf(history, "dxy")
        if dxy_df is not None:
            dxy_closes = dxy_df["close"].astype(float).tolist()
            ctx.dxy_aligned = bool(dxy.is_aligned(dxy_closes, ctx.direction))

    return hook


# ====================================================================== #
# STAGE 4 — indicators (Phase 11: VWAP / EMA / RSI-div / Volume Profile)  #
# ====================================================================== #

def make_indicator_hook(config: Optional[Dict[str, Any]] = None,
                        execution_tf: str = "5m") -> StageHook:
    config = config or {}
    from core.indicators.vwap import SessionalVWAP
    from core.indicators.ema import EMACalculator
    from core.indicators.rsi_divergence import RSIDivergenceDetector
    from core.indicators.volume_profile import VolumeProfile

    # NOTE: these indicators are STATEFUL (fed candle-by-candle).
    # TODO(user): in a live/replay loop, maintain ONE instance and feed each new
    #             closed candle once. The skeleton below rebuilds from the history
    #             window each call — correct but O(n) per bar; fine for wiring, not
    #             for a tight backtest loop.

    def hook(ctx: PipelineContext, bar: Any, history: Any) -> None:
        df = _tf(history, execution_tf)
        if df is None:
            return

        atr = compute_atr(df)
        vwap = SessionalVWAP(config.get("vwap"))
        ema = EMACalculator(config.get("ema"))
        rsi = RSIDivergenceDetector(config.get("rsi"))
        vp = VolumeProfile(config.get("volume_profile"))

        last_vwap = last_ema = last_vp = None
        for ts, row in df.iterrows():
            candle = {
                "timestamp": ts, "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
            }
            last_vwap = vwap.update(candle, atr=atr or 1.0)
            last_ema = ema.update(candle)
            rsi.update(candle)
            r = vp.update(candle, atr=atr or 1.0)
            if r is not None:
                last_vp = r

        ctx.vwap_reading = last_vwap
        ctx.ema_reading = last_ema
        ctx.divergence = rsi.detect_divergence()
        ctx.volume_profile_reading = last_vp

    return hook


# ====================================================================== #
# STAGE 5 — risk (SL → liquidity target → TP → position size)             #
# ====================================================================== #

def _sk_grid_depth(config: Dict[str, Any]) -> Optional[float]:
    """SK-System entry grid (Stefan Kassing 'Korrekturlevel').

    Flag-gated (``sk_entry_grid``), default OFF → live unchanged. When ON, the FVG
    entry depth is resolved from the SK retracement grid ``[0.50, 0.559, 0.618, 0.667]``
    (0.559 and 0.667 are the non-standard SK levels; 0.50/0.618 = the "golden pocket")
    INSTEAD of the single ``entry_depth_pct``. This is the "bridge" between SK's deep
    retrace entry and the bot's existing OTE lever — same FVG-depth mechanism, SK's
    exact levels. (SK proper anchors on the swing 0→A leg; anchoring on the FVG here is
    the cheap approximation. The true swing-anchored version is a future ``sk_fib_anchor``
    lever.) Because the pipeline emits ONE entry per signal, ``sk_entry_grid_mode``
    collapses the grid to a single depth:

        "mean"    -> laddered average across the pocket (default; most SK-faithful to
                     "place limit orders at every level")
        "golden"  -> the grid level nearest 0.618 (golden pocket)
        "shallow" -> shallowest level (most fills, worst price)
        "deep"    -> deepest level (best price, fewest fills)
        <float>   -> use that exact depth

    Returns the clamped ``[0, 0.95]`` depth, or ``None`` when the lever is OFF (the
    caller then falls back to ``entry_depth_pct``).
    """
    if not config.get("sk_entry_grid", False):
        return None
    levels = config.get("sk_entry_grid_levels") or [0.50, 0.559, 0.618, 0.667]
    levels = sorted(float(x) for x in levels)
    mode = config.get("sk_entry_grid_mode", "mean")
    if isinstance(mode, bool):                       # guard: bool is a subclass of int
        mode = "mean"
    if isinstance(mode, (int, float)):
        depth = float(mode)
    elif mode == "golden":
        depth = min(levels, key=lambda x: abs(x - 0.618))
    elif mode == "shallow":
        depth = levels[0]
    elif mode == "deep":
        depth = levels[-1]
    else:                                            # "mean" (default) and any unknown
        depth = sum(levels) / len(levels)
    return max(0.0, min(0.95, depth))


def make_risk_hook(config: Optional[Dict[str, Any]] = None,
                   account_balance: float = 10000.0,
                   execution_tf: str = "5m") -> StageHook:
    config = config or {}
    from core.risk.stop_loss import StopLossCalculator
    from core.risk.take_profit import TakeProfitCalculator
    from core.risk.liquidity_target_finder import LiquidityTargetFinder
    from core.risk.position_sizer import PositionSizer
    from core.risk.rr_calculator import RRCalculator

    sl_calc = StopLossCalculator(config)
    tp_calc = TakeProfitCalculator(config)
    liq_finder = LiquidityTargetFinder(config)
    sizer = PositionSizer(config, config.get("costs", config))
    rr_calc = RRCalculator(config, config.get("costs", config))  # F1: net-of-costs R:R
    rr_min = config.get("rr_tiers", {}).get("min_to_enter", 2.0)

    def hook(ctx: PipelineContext, bar: Any, history: Any) -> None:
        df = _tf(history, execution_tf)
        if df is None or ctx.sweep is None or ctx.fvg is None:
            return  # need a sweep + FVG before sizing a trade

        atr = compute_atr(df)
        if atr <= 0:
            return

        is_long = ctx.direction == "long"
        fvg_bottom = min(ctx.fvg["top"], ctx.fvg["bottom"])
        fvg_top = max(ctx.fvg["top"], ctx.fvg["bottom"])
        # Entry into the FVG. DEFAULT (entry_depth_pct=0.0) = the proximal near edge
        # (current live behavior). Opt-in (backtest-gated, default OFF → live unchanged):
        # enter DEEPER toward the distal edge — an OTE-style deeper discount/premium fill
        # (closer to the ICT 0.62-0.79 "golden pocket"). A deeper entry gives a better
        # price → better R:R and potentially higher win%, at the cost of needing a deeper
        # retrace to fill (fewer fills). Clamp to [0, 0.95] so entry stays inside the gap.
        # SK-System entry grid (Stefan Kassing) OVERRIDES the single OTE depth when the
        # sk_entry_grid lever is on (flag-gated, default OFF → live unchanged). See
        # _sk_grid_depth. Falls back to entry_depth_pct (the existing OTE lever) otherwise.
        sk_depth = _sk_grid_depth(config)
        depth = sk_depth if sk_depth is not None \
            else max(0.0, min(0.95, float(config.get("entry_depth_pct", 0.0))))
        gap = fvg_top - fvg_bottom
        entry = (fvg_top - depth * gap) if is_long else (fvg_bottom + depth * gap)

        sweep_level = ctx.sweep.get("level") if isinstance(ctx.sweep, dict) else None
        sl_res = sl_calc.calculate(
            direction=ctx.direction, entry=entry, atr=atr,
            sweep_low=sweep_level if is_long else None,
            sweep_high=sweep_level if not is_long else None,
            fvg_bottom=fvg_bottom, fvg_top=fvg_top,
        )
        if not sl_res.valid:
            return

        # Real liquidity pools collected by the SMC stage feed the TP2 search.
        levels = ctx.extra.get("liquidity_levels", [])
        liq_res = liq_finder.find(ctx.direction, entry, sl_res.sl_distance, levels)
        liq_price = liq_res.tp2_target.price if liq_res.tp2_target else None

        tp_res = tp_calc.calculate(
            direction=ctx.direction, entry=entry,
            sl_distance=sl_res.sl_distance, liquidity_target_price=liq_price,
        )

        size = sizer.calculate(account_balance=account_balance, entry=entry, sl=sl_res.sl_price)

        ctx.entry = entry
        ctx.sl = sl_res.sl_price
        ctx.tp1 = tp_res.tp1
        ctx.tp2 = tp_res.tp2
        # F1: R:R gate + display must be NET of execution costs (spread + slippage),
        # not raw price geometry. RRCalculator subtracts costs; rr.valid is the
        # net >= min_to_enter gate. (Previously ctx.net_rr = tp_res.tp2_r was gross.)
        rr = rr_calc.calculate(
            direction=ctx.direction, entry=entry, sl=sl_res.sl_price, tp=tp_res.tp2,
            is_news_time=not ctx.news_clear,
            is_high_volatility=not getattr(ctx, "clean_market_state", True),
        )
        ctx.net_rr = rr.net_rr
        ctx.rr_minimum_ok = rr.valid
        ctx.extra["gross_rr"] = rr.gross_rr  # keep gross too, for display transparency
        size_ok = bool(size and getattr(size, "valid", False))
        ctx.lot_size = getattr(size, "lot_size", 0.01) if size_ok else 0.01
        # Honor the sizer's rejection: a valid=False result means the risk could NOT be
        # validated against risk_per_trade/max (SL too small, account can't afford 0.01).
        # The emit paths suppress the signal on sizing_valid=False instead of firing a
        # forced 0.01 lot whose risk was never checked.
        ctx.extra["sizing_valid"] = size_ok
        ctx.liquidity_target_clear = liq_price is not None
        # multiple confluence booster: 2+ of OB / displacement / DXY / fresh FVG
        ctx.multiple_confluence = sum([
            ctx.ob_valid, ctx.strong_displacement, ctx.dxy_aligned, ctx.fvg_fresh,
        ]) >= 2

    return hook


# ====================================================================== #
# Convenience: build all hooks at once                                    #
# ====================================================================== #

def build_default_hooks(config: Optional[Dict[str, Any]] = None,
                        account_balance: float = 10000.0,
                        execution_tf: str = "5m") -> Dict[str, StageHook]:
    """Return a dict of all five hooks, ready to splat into SignalPipeline.

        pipe = SignalPipeline(rulebook, **build_default_hooks(cfg))
    """
    config = config or {}
    return {
        "structure_hook": make_structure_hook(config),
        "smc_hook": make_smc_hook(config, execution_tf),
        "filter_hook": make_filter_hook(config, execution_tf),
        "indicator_hook": make_indicator_hook(config, execution_tf),
        "risk_hook": make_risk_hook(config, account_balance, execution_tf),
    }
