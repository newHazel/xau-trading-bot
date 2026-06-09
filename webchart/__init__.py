"""Standalone web chart — FastAPI backend + TradingView Lightweight Charts frontend.

Serves an interactive candlestick chart for a chosen symbol/timeframe, reading
candles from the SQLite DB and overlaying the bot's indicators (VWAP, EMA 50/200,
RSI) plus signal markers and recent FVG/OB zones.

Run:  python -m webchart.server      (then open http://localhost:8000)
"""
