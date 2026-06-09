"""
XAU Trading Bot v1.2 — Main Entry Point
Mode: alerts only. allow_auto_trading is always False.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XAU Trading Bot v1.2")
    parser.add_argument(
        "--mode",
        choices=["research", "backtest", "paper", "live_alerts"],
        default="research",
        help="Operating mode (default: research)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="Path to config directory (default: config/)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/database/trading_bot.sqlite"),
        help="SQLite database path",
    )
    return parser.parse_args()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def main() -> None:
    load_dotenv()
    args = parse_args()

    from core.utils.config_loader import load_config
    from core.utils.config_hash import compute_config_hash, short_hash
    from core.logging.db import get_db

    config = load_config(args.config_dir)
    setup_logging(config.settings.get("log_level", "INFO"))

    logger = logging.getLogger(__name__)

    cfg_hash = compute_config_hash(config.raw)
    logger.info("XAU Trading Bot %s | mode=%s | config=%s",
                config.strategy_version, config.mode, short_hash(config.raw))

    db = get_db(args.db_path)

    # Record this run in the experiments table
    db.execute(
        "INSERT OR IGNORE INTO experiments "
        "(experiment_name, config_hash, strategy_version, symbol, notes) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"run_{config.mode}", cfg_hash, config.strategy_version,
         config.primary_symbol, f"mode={config.mode}"),
    )

    logger.info("Infrastructure ready. Phase 0 complete.")
    logger.info("Next: Phase 1 — Market Structure.")


if __name__ == "__main__":
    main()
