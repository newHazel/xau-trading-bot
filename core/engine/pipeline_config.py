"""
Pipeline config assembler.

Loads the project's YAML config (config/*.yaml) and flattens it into the single
dict the SignalPipeline hooks expect. Without this, the pipeline runs on bare
defaults — e.g. SessionFilter gets no kill-zone hours, so `kill_zone` never
passes and no A/A+ signal can ever form.

Use this instead of a hand-written inline config:

    from core.engine.pipeline_config import assemble_pipeline_config
    cfg = assemble_pipeline_config("config")
    pipe = SignalPipeline(RulebookEngine(cfg), **build_default_hooks(cfg, ...))
"""

from __future__ import annotations

from typing import Any, Dict


def assemble_pipeline_config(config_dir: str = "config") -> Dict[str, Any]:
    from core.utils.config_loader import load_config

    cfg = load_config(config_dir)
    raw = cfg.raw

    risk = dict(raw.get("risk", {}))
    smc = dict(raw.get("smc_rules", {}))
    costs = dict(raw.get("execution_costs", {}))
    costs.setdefault("point_value_per_lot", 100.0)

    master: Dict[str, Any] = {}
    # Flatten SMC + risk keys to the top level (StopLoss/TP/LiquidityFinder/
    # PositionSizer/grader read them directly).
    master.update(smc)
    master.update(risk)

    # Sub-sections the filter/detector hooks request by name.
    master.update({
        "costs": costs,
        "session": raw.get("sessions", {}),     # SessionFilter ← kill-zone hours
        "news": raw.get("news", {}),             # NewsFilter
        "dxy": smc,                              # DXYFilter (dxy_required, lookback)
        "spread": dict(costs),                   # SpreadFilter (own dict — never alias costs)
        "correlation": smc,                     # CorrelationSpikeFilter
        "volatility": raw.get("volatility", {}),
        "market_state": raw.get("market_state", {}),
        "rr_tiers": risk.get("rr_tiers", {}),
        "state_machine": raw.get("state_machine", {}),  # cooldown/expiry pacing (read by the live engine)
    })

    # Optional: swing fractal windows per timeframe, if the project defines them.
    tf = raw.get("timeframes", {})
    if isinstance(tf, dict) and "fractal_windows" in tf:
        master["fractal_windows"] = tf["fractal_windows"]
    elif "fractal_windows" in smc:
        master["fractal_windows"] = smc["fractal_windows"]

    return master
