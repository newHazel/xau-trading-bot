"""
Config Loader — Phase 0.8.

Loads all YAML files from the config/ directory into a single
unified Config object. Every quantitative threshold lives in YAML —
nothing is hardcoded in the system.

Responsibilities:
  - Load each YAML file by name.
  - Validate that mandatory keys are present.
  - Enforce allow_auto_trading: false (hard stop if misconfigured).
  - Expose typed accessors for every config section.
  - Support override via environment variables for secrets.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# All YAML files that must exist in the config directory
_REQUIRED_FILES = [
    "settings.yaml",
    "symbols.yaml",
    "timeframes.yaml",
    "sessions.yaml",
    "risk.yaml",
    "smc_rules.yaml",
    "news.yaml",
    "execution_costs.yaml",
    "market_calendar.yaml",
    "mitigation_rules.yaml",
    "displacement_rules.yaml",
    "state_machine.yaml",
    "data_sources.yaml",
    "health.yaml",
]

# Mandatory keys per section — loader raises if any are missing
_REQUIRED_KEYS: Dict[str, List[str]] = {
    "settings":    ["project_name", "strategy_version", "timezone", "mode",
                    "use_closed_candles_only", "allow_auto_trading"],
    "symbols":     ["primary_symbol"],
    "timeframes":  ["base_timeframe", "analysis_timeframes"],
    "risk":        ["risk_per_trade_percent", "max_daily_losses", "max_daily_trades"],
    "smc_rules":   ["fvg_min_atr_ratio", "displacement"],
    "news":        ["tiers", "fallback"],
    "data_sources": ["primary", "fallback_1", "failover_threshold_seconds"],
    "state_machine": ["cooldown_minutes_after_signal", "setup_expiry_minutes"],
}


class Config:
    """
    Immutable-ish container for the full project config.
    Access any section with config.<section_name> (dict).
    """

    def __init__(self, data: Dict[str, Dict[str, Any]], config_dir: Path) -> None:
        self._data = data
        self._config_dir = config_dir
        self._enforce_safety()

    # ---------------------------------------------------------------- #
    # Section accessors                                                  #
    # ---------------------------------------------------------------- #

    @property
    def settings(self) -> Dict[str, Any]:
        return self._data["settings"]

    @property
    def symbols(self) -> Dict[str, Any]:
        return self._data["symbols"]

    @property
    def timeframes(self) -> Dict[str, Any]:
        return self._data["timeframes"]

    @property
    def sessions(self) -> Dict[str, Any]:
        return self._data["sessions"]

    @property
    def risk(self) -> Dict[str, Any]:
        return self._data["risk"]

    @property
    def smc_rules(self) -> Dict[str, Any]:
        return self._data["smc_rules"]

    @property
    def news(self) -> Dict[str, Any]:
        return self._data["news"]

    @property
    def execution_costs(self) -> Dict[str, Any]:
        return self._data["execution_costs"]

    @property
    def market_calendar(self) -> Dict[str, Any]:
        return self._data["market_calendar"]

    @property
    def mitigation_rules(self) -> Dict[str, Any]:
        return self._data["mitigation_rules"]

    @property
    def displacement_rules(self) -> Dict[str, Any]:
        return self._data["displacement_rules"]

    @property
    def state_machine(self) -> Dict[str, Any]:
        return self._data["state_machine"]

    @property
    def data_sources(self) -> Dict[str, Any]:
        return self._data["data_sources"]

    @property
    def health(self) -> Dict[str, Any]:
        return self._data["health"]

    # ---------------------------------------------------------------- #
    # Convenience typed properties                                       #
    # ---------------------------------------------------------------- #

    @property
    def mode(self) -> str:
        return self.settings["mode"]

    @property
    def primary_symbol(self) -> str:
        return self.symbols["primary_symbol"]

    @property
    def base_timeframe(self) -> str:
        return self.timeframes["base_timeframe"]

    @property
    def analysis_timeframes(self) -> List[str]:
        return self.timeframes["analysis_timeframes"]

    @property
    def risk_per_trade_percent(self) -> float:
        return float(self.risk["risk_per_trade_percent"])

    @property
    def allow_auto_trading(self) -> bool:
        return bool(self.settings.get("allow_auto_trading", False))

    @property
    def strategy_version(self) -> str:
        return self.settings["strategy_version"]

    @property
    def timezone(self) -> str:
        return self.settings["timezone"]

    # ---------------------------------------------------------------- #
    # Raw dict access (for passing sub-sections to other modules)       #
    # ---------------------------------------------------------------- #

    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """Flexible accessor: config.get('risk') or config.get('risk', 'tp1_r')."""
        section_data = self._data.get(section, {})
        if key is None:
            return section_data
        return section_data.get(key, default)

    @property
    def raw(self) -> Dict[str, Dict[str, Any]]:
        """Full raw dict — used for hashing."""
        return self._data

    # ---------------------------------------------------------------- #
    # Safety                                                             #
    # ---------------------------------------------------------------- #

    def _enforce_safety(self) -> None:
        """Hard-stop if allow_auto_trading is True — non-negotiable."""
        if self.settings.get("allow_auto_trading", False) is True:
            raise ValueError(
                "SAFETY VIOLATION: allow_auto_trading is set to True. "
                "This system is alerts-only. Set it to false."
            )
        logger.debug("[Config] Safety check passed: allow_auto_trading=False")

    def __repr__(self) -> str:
        return (
            f"Config(mode={self.mode!r}, symbol={self.primary_symbol!r}, "
            f"version={self.strategy_version!r})"
        )


# ------------------------------------------------------------------ #
# Loader                                                               #
# ------------------------------------------------------------------ #

class ConfigLoader:
    """Loads and validates all YAML config files from a directory."""

    def __init__(self, config_dir: str | Path = "config") -> None:
        self._config_dir = Path(config_dir)

    def load(self) -> Config:
        """
        Load all required YAML files, validate mandatory keys,
        and return a Config instance.

        Raises
        ------
        FileNotFoundError  : if a required YAML file is missing.
        KeyError           : if a mandatory key is absent.
        ValueError         : if allow_auto_trading is True.
        """
        if not self._config_dir.is_dir():
            raise FileNotFoundError(
                f"Config directory not found: {self._config_dir.resolve()}"
            )

        data: Dict[str, Dict[str, Any]] = {}

        for filename in _REQUIRED_FILES:
            path = self._config_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Required config file missing: {path.resolve()}"
                )
            section = filename.replace(".yaml", "")
            data[section] = self._load_yaml(path)
            logger.debug("[ConfigLoader] Loaded %s", filename)

        # Load optional telegram config (example file is ok to skip)
        telegram_path = self._config_dir / "telegram.yaml"
        if telegram_path.exists():
            data["telegram"] = self._load_yaml(telegram_path)
        else:
            data["telegram"] = {}
            logger.debug("[ConfigLoader] telegram.yaml not found — Telegram disabled.")

        self._validate(data)
        config = Config(data, self._config_dir)
        logger.info("[ConfigLoader] Config loaded: %r", config)
        return config

    # ---------------------------------------------------------------- #
    # Internal                                                           #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            result = yaml.safe_load(f)
        return result or {}

    @staticmethod
    def _validate(data: Dict[str, Dict[str, Any]]) -> None:
        for section, required_keys in _REQUIRED_KEYS.items():
            section_data = data.get(section, {})
            for key in required_keys:
                if key not in section_data:
                    raise KeyError(
                        f"Mandatory config key missing: [{section}] → '{key}'"
                    )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def load_config(config_dir: str | Path = "config") -> Config:
    """One-call shortcut: load_config() → Config."""
    return ConfigLoader(config_dir).load()
