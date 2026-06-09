"""
Web chart server — FastAPI.

Endpoints
    GET /                       → the chart page (static/index.html)
    GET /api/meta               → symbols + timeframes available in the DB
    GET /api/chart?symbol&tf&limit → candles + indicator series + signals + zones

Indicators are computed server-side from the existing core.indicators classes
(lightweight — VWAP/EMA/RSI over a few thousand candles is milliseconds).

Run:
    python -m webchart.server
    python -m webchart.server --port 8000 --db data/database/trading_bot.sqlite
Then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

_ROOT = Path(__file__).parent
_STATIC = _ROOT / "static"
DB_PATH = str(_ROOT.parent / "data" / "database" / "trading_bot.sqlite")

# Load .env so the live-refresh fetchers (Twelve Data) get their API key.
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT.parent / ".env")
except Exception:
    pass

app = FastAPI(title="XAU Web Chart")

# --- live refresh: fetch latest closed candles on demand (throttled) ---
import time as _time
_last_live_fetch: Dict[tuple, float] = {}
_LIVE_THROTTLE_SEC = 30


def _store_candles(df, symbol: str, tf: str, source: str) -> int:
    conn = _connect()
    try:
        rows = [
            (symbol, tf, ts.isoformat(), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]), source)
            for ts, r in df.iterrows()
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO candles "
            "(symbol,timeframe,timestamp,open,high,low,close,volume,source) "
            "VALUES (?,?,?,?,?,?,?,?,?)", rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def _refresh_latest(symbol: str, tf: str) -> int:
    """Pull the latest CLOSED candles for symbol+tf and upsert to the DB.

    Throttled to once per _LIVE_THROTTLE_SEC per symbol+tf so rapid chart
    refreshes don't burn the data-provider quota. XAUUSDT → Bybit (free public);
    everything else → Twelve Data (spot, uses the free key).
    """
    key = (symbol, tf)
    now = _time.time()
    if now - _last_live_fetch.get(key, 0.0) < _LIVE_THROTTLE_SEC:
        return 0
    _last_live_fetch[key] = now
    try:
        if symbol == "XAUUSDT":
            from core.data.bybit_fetcher import BybitFetcher
            res = BybitFetcher().fetch_latest_candles(symbol, tf, 60)
        else:
            from core.data.twelvedata_fetcher import TwelveDataFetcher
            res = TwelveDataFetcher().fetch_latest_candles(symbol, tf, 60)
        status = getattr(res.status, "value", res.status)
        if status == "ok" and res.data is not None and not res.data.empty:
            return _store_candles(res.data, symbol, tf, res.source)
    except Exception:
        pass
    return 0


# ------------------------------------------------------------------ #
# DB helpers                                                          #
# ------------------------------------------------------------------ #

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _epoch(ts_str: str) -> int:
    """ISO timestamp → UNIX seconds (Lightweight Charts wants seconds)."""
    import pandas as pd
    return int(pd.Timestamp(ts_str).timestamp())


# ------------------------------------------------------------------ #
# Indicator computation (server-side, lightweight)                    #
# ------------------------------------------------------------------ #

def _candle_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        out.append({
            "timestamp": r["timestamp"],
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]),
            "volume": float(r["volume"]),
        })
    return out


def _compute_indicators(candles: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    """Return {vwap, ema50, ema200, rsi} as [{time, value}] series for the chart."""
    from core.indicators.vwap import SessionalVWAP
    from core.indicators.ema import EMACalculator
    from core.indicators.rsi_divergence import RSIDivergenceDetector

    vwap = SessionalVWAP()
    ema = EMACalculator({"fast_period": 50, "slow_period": 200})
    rsi = RSIDivergenceDetector({"period": 14})

    vwap_s, ema50_s, ema200_s, rsi_s = [], [], [], []
    for c in candles:
        t = _epoch(c["timestamp"])
        vr = vwap.update(c, atr=1.0)
        vwap_s.append({"time": t, "value": round(vr.vwap, 4)})
        er = ema.update(c)
        ema50_s.append({"time": t, "value": round(er.ema_fast, 4)})
        ema200_s.append({"time": t, "value": round(er.ema_slow, 4)})
        rr = rsi.update(c)
        if rr is not None:
            rsi_s.append({"time": t, "value": round(rr.rsi, 2)})

    return {"vwap": vwap_s, "ema50": ema50_s, "ema200": ema200_s, "rsi": rsi_s}


def _recent_zones(candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Most recent FVG zones (bull+bear) as {time, top, bottom, type}."""
    import pandas as pd
    if len(candles) < 30:
        return []
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    from core.smc.fvg_detector import FVGDetector
    fvgs = FVGDetector()
    fvg_df = fvgs.detect(df)
    zones = []
    for direction in ("bull", "bear"):
        z = fvgs.get_last_fvg(fvg_df, direction=direction)
        if z is not None:
            zones.append({
                "time": int(pd.Timestamp(z["confirm_ts"]).timestamp()),
                "top": round(z["top"], 4),
                "bottom": round(z["bottom"], 4),
                "type": direction,
            })
    return zones


# ------------------------------------------------------------------ #
# Routes                                                              #
# ------------------------------------------------------------------ #

@app.get("/api/meta")
def meta() -> JSONResponse:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT DISTINCT symbol, timeframe FROM candles ORDER BY symbol, timeframe"
        ).fetchall()
    finally:
        conn.close()
    symbols: Dict[str, List[str]] = {}
    for r in rows:
        symbols.setdefault(r["symbol"], []).append(r["timeframe"])
    return JSONResponse({"symbols": symbols})


@app.get("/api/chart")
def chart(
    symbol: str = Query("XAUUSDT"),
    tf: str = Query("15m"),
    limit: int = Query(1000, ge=50, le=5000),
    live: int = Query(1, description="1 = fetch latest candles before serving"),
) -> JSONResponse:
    fetched = _refresh_latest(symbol, tf) if live else 0

    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT timestamp,open,high,low,close,volume FROM candles "
            "WHERE symbol=? AND timeframe=? ORDER BY timestamp DESC LIMIT ?",
            (symbol, tf, limit),
        ).fetchall()
        rows = list(reversed(rows))
        sig_rows = conn.execute(
            "SELECT setup_id,timestamp,direction,grade,entry,stop_loss,tp1,tp2 "
            "FROM signals WHERE symbol=? ORDER BY timestamp",
            (symbol,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return JSONResponse({"candles": [], "indicators": {}, "signals": [], "zones": []})

    candles = _candle_dicts(rows)
    chart_candles = [{
        "time": _epoch(c["timestamp"]),
        "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"],
    } for c in candles]
    volumes = [{
        "time": _epoch(c["timestamp"]),
        "value": c["volume"],
        "color": "rgba(38,166,154,0.5)" if c["close"] >= c["open"] else "rgba(239,83,80,0.5)",
    } for c in candles]

    indicators = _compute_indicators(candles)
    zones = _recent_zones(candles)

    signals = []
    for s in sig_rows:
        is_long = str(s["direction"]).upper() == "LONG"
        signals.append({
            "time": _epoch(s["timestamp"]),
            "position": "belowBar" if is_long else "aboveBar",
            "color": {"A+": "#16a34a", "A": "#22c55e", "B": "#eab308"}.get(s["grade"], "#9ca3af"),
            "shape": "arrowUp" if is_long else "arrowDown",
            "text": f'{s["grade"]} {s["direction"]} @{round(s["entry"],1)}',
            "entry": s["entry"], "sl": s["stop_loss"], "tp1": s["tp1"], "tp2": s["tp2"],
            "grade": s["grade"],
        })

    return JSONResponse({
        "candles": chart_candles,
        "volume": volumes,
        "indicators": indicators,
        "signals": signals,
        "zones": zones,
        "live_fetched": fetched,
        "latest": chart_candles[-1]["time"] if chart_candles else None,
    })


def _smc_overlay(candles: List[Dict[str, Any]], tf: str) -> Dict[str, Any]:
    """Run the SMC detectors over the loaded candles and return drawable overlays:
    swings, sweeps, BOS/CHoCH, liquidity levels, FVG zones, order blocks."""
    import pandas as pd
    if len(candles) < 40:
        return {}
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")

    from core.structure.swing_detector import SwingDetector
    from core.structure.market_structure import MarketStructure
    from core.structure.bos_detector import BOSDetector
    from core.structure.choch_detector import CHoCHDetector
    from core.smc.liquidity_detector import LiquidityDetector
    from core.smc.sweep_detector import SweepDetector
    from core.smc.fvg_detector import FVGDetector
    from core.smc.order_block_detector import OrderBlockDetector

    out: Dict[str, Any] = {"swings": [], "sweeps": [], "bos": [], "choch": [],
                           "liquidity": [], "fvgs": [], "order_blocks": []}
    try:
        sw = SwingDetector()
        sw_df = sw.detect(df, tf)
        for _, r in sw.get_recent_swings(sw_df, 12).iterrows():
            out["swings"].append({"time": _epoch(r["confirm_ts"]),
                                  "price": round(float(r["price"]), 2),
                                  "type": r["swing_type"]})

        bos_df = BOSDetector().detect(sw_df)
        for b in BOSDetector().get_all_bos(bos_df, 6):
            out["bos"].append({"time": _epoch(b["confirm_ts"]),
                               "price": round(b["level"], 2), "dir": b["direction"]})

        struct = MarketStructure().classify(sw_df)
        choch_df = CHoCHDetector().detect(struct)
        for c in CHoCHDetector().get_all_choch(choch_df, 6):
            out["choch"].append({"time": _epoch(c["confirm_ts"]),
                                 "price": round(c["level"], 2), "dir": c["direction"]})

        liq_df = LiquidityDetector().detect(sw_df)
        for col, typ in [("eqh_level", "EQH"), ("eql_level", "EQL"),
                         ("pdh", "PDH"), ("pdl", "PDL")]:
            if col in liq_df.columns:
                s = liq_df[col].dropna()
                if not s.empty:
                    out["liquidity"].append({"price": round(float(s.iloc[-1]), 2), "type": typ})

        sd = SweepDetector()
        sweep_df = sd.detect(liq_df)
        for d in ("bull", "bear"):
            s = sd.get_last_sweep(sweep_df, direction=d)
            if s:
                out["sweeps"].append({"time": _epoch(s["confirm_ts"]),
                                      "price": round(s["level"], 2), "dir": d})

        fd = FVGDetector()
        fvg_df = fd.detect(df)
        # Only UNMITIGATED FVGs (fresh/tapped/partial/deep) — the still-relevant ones.
        from core.smc.mitigation_tracker import MitigationTracker
        mt = MitigationTracker()
        mitig_df = mt.track(fvg_df)
        for z in mt.get_unmitigated_fvgs(mitig_df, 4):
            out["fvgs"].append({"time": _epoch(z["confirm_ts"]),
                                "top": round(z["top"], 2), "bottom": round(z["bottom"], 2),
                                "dir": z["fvg_type"]})

        ob_df = OrderBlockDetector().detect(fvg_df)
        for ob in OrderBlockDetector().get_order_blocks(ob_df, n=3):
            out["order_blocks"].append({"time": _epoch(ob["timestamp"]),
                                        "top": round(ob["top"], 2),
                                        "bottom": round(ob["bottom"], 2),
                                        "dir": ob["ob_type"]})
    except Exception as exc:
        out["error"] = str(exc)
    return out


@app.get("/api/smc")
def smc(
    symbol: str = Query("XAUUSD"),
    tf: str = Query("15m"),
    limit: int = Query(1000, ge=50, le=5000),
) -> JSONResponse:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT timestamp,open,high,low,close,volume FROM candles "
            "WHERE symbol=? AND timeframe=? ORDER BY timestamp DESC LIMIT ?",
            (symbol, tf, limit),
        ).fetchall()
    finally:
        conn.close()
    candles = _candle_dicts(list(reversed(rows)))
    return JSONResponse(_smc_overlay(candles, tf))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_STATIC / "index.html"))


# Serve the static dir (index.html, any assets)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


def main() -> None:
    global DB_PATH
    p = argparse.ArgumentParser(description="XAU web chart server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--db", default=DB_PATH)
    args = p.parse_args()
    DB_PATH = args.db

    import uvicorn
    print(f"XAU web chart → http://{args.host}:{args.port}  (DB: {DB_PATH})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
