# XAU Trading Bot v1.2

Multi-Timeframe SMC/ICT analysis and alerts system for XAU/USD.

**Mode: Alerts Only — No Auto-Trading**

## Strategy Overview


| Timeframe | Role                                       |
| --------- | ------------------------------------------ |
| 4H + 1H   | HTF Bias / Main Direction                  |
| 15m       | Intermediate Structure / Areas of Interest |
| 5m        | LTF Setup — Sweep + FVG + OB               |
| 1m / 3m   | Trigger — Micro CHoCH + Confirmation       |


## Build Phases

- **Phase 0** — Infrastructure (data, config, DB, gaps)
- **Phase 1** — Market Structure (swings, BOS, CHoCH)
- **Phase 2** — SMC Core (FVG, sweep, displacement, OB)
- **Phase 3** — Filters (sessions, news, calendar, DXY)
- **Phase 4** — Rulebook + State Machine
- **Phase 5** — Risk Management
- **Phase 6** — Backtesting + Walk-Forward
- **Phase 7** — Paper Trading
- **Phase 8** — Live Small Alerts
- **Phase 9** — Operational (health, heartbeat, dedup)
- **Phase 10** — Dashboard (optional)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

## Run

```bash
python main.py --mode research
python main.py --mode backtest
python main.py --mode paper
python main.py --mode live_alerts
```

## Tests

```bash
pytest tests/ -v --cov=core
```

## Important Rules

1. `allow_auto_trading: false` — always
2. Closed candles only — never analyze in-progress candles
3. No look-ahead bias in any detector or backtest
4. All quantitative values in YAML config — never hardcoded
5. All signals AND rejections logged to SQLite

