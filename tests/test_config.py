"""
Tests for config_loader.py + config_hash.py — Phase 0.8.
"""

import json
import pytest
import tempfile
from pathlib import Path

import yaml

from core.utils.config_loader import Config, ConfigLoader, load_config
from core.utils.config_hash import (
    compute_config_hash,
    short_hash,
    hashes_match,
    _clean,
)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

def _write_yaml(dir_path: Path, filename: str, data: dict) -> None:
    with open(dir_path / filename, "w") as f:
        yaml.dump(data, f)


def _make_config_dir(tmp_path: Path) -> Path:
    """Write all required YAML files to tmp_path."""
    files = {
        "settings.yaml": {
            "project_name": "Test Bot",
            "strategy_version": "1.2.0",
            "timezone": "Asia/Jerusalem",
            "mode": "research",
            "use_closed_candles_only": True,
            "allow_auto_trading": False,
            "log_level": "DEBUG",
        },
        "symbols.yaml": {
            "primary_symbol": "XAUUSD",
            "alternative_symbols": ["GC=F"],
            "dxy_symbol": "DXY",
        },
        "timeframes.yaml": {
            "base_timeframe": "1m",
            "analysis_timeframes": ["5m", "15m", "1h", "4h"],
            "fractal_windows": {"1m": 3, "5m": 5},
        },
        "sessions.yaml": {"timezone": "Asia/Jerusalem", "london_kill_zone": {"start": "10:00"}},
        "risk.yaml": {
            "risk_per_trade_percent": 0.5,
            "max_daily_losses": 2,
            "max_daily_trades": 3,
        },
        "smc_rules.yaml": {
            "fvg_min_atr_ratio": 0.3,
            "displacement": {"enabled": True, "atr_period": 14},
        },
        "news.yaml": {
            "tiers": {"tier_1": {"events": ["FOMC"], "block_before_minutes": 60}},
            "fallback": {"use_manual_csv_if_api_fails": True},
        },
        "execution_costs.yaml": {"default_spread": 0.25},
        "market_calendar.yaml": {"enabled": True},
        "mitigation_rules.yaml": {"max_allowed_fill_percent_for_live": 0.5},
        "displacement_rules.yaml": {"enabled": True, "atr_period": 14},
        "state_machine.yaml": {
            "cooldown_minutes_after_signal": 60,
            "setup_expiry_minutes": 90,
        },
        "data_sources.yaml": {
            "primary": "oanda",
            "fallback_1": "bybit",
            "failover_threshold_seconds": 30,
        },
        "health.yaml": {"checks": ["data_freshness"]},
    }
    for filename, data in files.items():
        _write_yaml(tmp_path, filename, data)
    return tmp_path


# ------------------------------------------------------------------ #
# ConfigLoader                                                         #
# ------------------------------------------------------------------ #

class TestConfigLoader:
    def test_load_returns_config(self, tmp_path):
        _make_config_dir(tmp_path)
        config = load_config(tmp_path)
        assert isinstance(config, Config)

    def test_primary_symbol_correct(self, tmp_path):
        _make_config_dir(tmp_path)
        config = load_config(tmp_path)
        assert config.primary_symbol == "XAUUSD"

    def test_mode_correct(self, tmp_path):
        _make_config_dir(tmp_path)
        config = load_config(tmp_path)
        assert config.mode == "research"

    def test_strategy_version_correct(self, tmp_path):
        _make_config_dir(tmp_path)
        config = load_config(tmp_path)
        assert config.strategy_version == "1.2.0"

    def test_analysis_timeframes_list(self, tmp_path):
        _make_config_dir(tmp_path)
        config = load_config(tmp_path)
        assert "5m" in config.analysis_timeframes
        assert "4h" in config.analysis_timeframes

    def test_risk_per_trade_is_float(self, tmp_path):
        _make_config_dir(tmp_path)
        config = load_config(tmp_path)
        assert isinstance(config.risk_per_trade_percent, float)
        assert config.risk_per_trade_percent == pytest.approx(0.5)

    def test_allow_auto_trading_is_false(self, tmp_path):
        _make_config_dir(tmp_path)
        config = load_config(tmp_path)
        assert config.allow_auto_trading is False

    def test_missing_config_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config directory"):
            load_config(tmp_path / "nonexistent")

    def test_missing_file_raises(self, tmp_path):
        _make_config_dir(tmp_path)
        (tmp_path / "risk.yaml").unlink()
        with pytest.raises(FileNotFoundError, match="risk.yaml"):
            load_config(tmp_path)

    def test_missing_mandatory_key_raises(self, tmp_path):
        _make_config_dir(tmp_path)
        # Remove a mandatory key from settings
        data = yaml.safe_load((tmp_path / "settings.yaml").read_text())
        del data["allow_auto_trading"]
        _write_yaml(tmp_path, "settings.yaml", data)
        with pytest.raises(KeyError, match="allow_auto_trading"):
            load_config(tmp_path)

    def test_section_get_accessor(self, tmp_path):
        _make_config_dir(tmp_path)
        config = load_config(tmp_path)
        assert config.get("risk", "max_daily_losses") == 2

    def test_section_get_default(self, tmp_path):
        _make_config_dir(tmp_path)
        config = load_config(tmp_path)
        assert config.get("risk", "nonexistent_key", "default") == "default"

    def test_repr_contains_mode(self, tmp_path):
        _make_config_dir(tmp_path)
        config = load_config(tmp_path)
        assert "research" in repr(config)

    def test_telegram_missing_gives_empty_dict(self, tmp_path):
        _make_config_dir(tmp_path)
        config = load_config(tmp_path)
        assert config.get("telegram") == {}


# ------------------------------------------------------------------ #
# Safety enforcement                                                   #
# ------------------------------------------------------------------ --

class TestConfigSafety:
    def test_auto_trading_true_raises(self, tmp_path):
        _make_config_dir(tmp_path)
        data = yaml.safe_load((tmp_path / "settings.yaml").read_text())
        data["allow_auto_trading"] = True
        _write_yaml(tmp_path, "settings.yaml", data)
        with pytest.raises(ValueError, match="SAFETY VIOLATION"):
            load_config(tmp_path)


# ------------------------------------------------------------------ #
# Config Hash                                                          #
# ------------------------------------------------------------------ #

class TestConfigHash:
    SAMPLE = {"risk": {"rr": 2.0}, "mode": "research"}

    def test_same_config_same_hash(self):
        h1 = compute_config_hash(self.SAMPLE)
        h2 = compute_config_hash(self.SAMPLE)
        assert h1 == h2

    def test_different_config_different_hash(self):
        h1 = compute_config_hash({"a": 1})
        h2 = compute_config_hash({"a": 2})
        assert h1 != h2

    def test_key_order_irrelevant(self):
        h1 = compute_config_hash({"a": 1, "b": 2})
        h2 = compute_config_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_hash_is_64_chars(self):
        assert len(compute_config_hash(self.SAMPLE)) == 64

    def test_short_hash_is_12_chars(self):
        assert len(short_hash(self.SAMPLE)) == 12

    def test_short_hash_is_prefix_of_full(self):
        full = compute_config_hash(self.SAMPLE)
        short = short_hash(self.SAMPLE)
        assert full.startswith(short)

    def test_hashes_match_same(self):
        stored = compute_config_hash(self.SAMPLE)
        assert hashes_match(self.SAMPLE, stored) is True

    def test_hashes_match_different(self):
        stored = compute_config_hash({"x": 1})
        assert hashes_match(self.SAMPLE, stored) is False

    def test_secrets_stripped_from_hash(self):
        cfg_with_secret    = {"api_key": "abc123", "mode": "research"}
        cfg_without_secret = {"mode": "research"}
        assert compute_config_hash(cfg_with_secret) == compute_config_hash(cfg_without_secret)

    def test_float_noise_normalised(self):
        h1 = compute_config_hash({"val": 2.0000000001})
        h2 = compute_config_hash({"val": 2.0000000002})
        assert h1 == h2   # rounded to 8 dp → same

    def test_nested_secrets_stripped(self):
        cfg1 = {"telegram": {"bot_token": "SECRET", "chat_id": "12345"}, "mode": "research"}
        cfg2 = {"telegram": {"bot_token": "OTHER",  "chat_id": "99999"}, "mode": "research"}
        assert compute_config_hash(cfg1) == compute_config_hash(cfg2)

    def test_hash_from_real_config(self, tmp_path):
        _make_config_dir(tmp_path)
        config = load_config(tmp_path)
        h = compute_config_hash(config.raw)
        assert len(h) == 64

    def test_config_change_changes_hash(self, tmp_path):
        _make_config_dir(tmp_path)
        config1 = load_config(tmp_path)
        h1 = compute_config_hash(config1.raw)

        # Change a config value
        data = yaml.safe_load((tmp_path / "risk.yaml").read_text())
        data["risk_per_trade_percent"] = 1.0
        _write_yaml(tmp_path, "risk.yaml", data)

        config2 = load_config(tmp_path)
        h2 = compute_config_hash(config2.raw)
        assert h1 != h2
