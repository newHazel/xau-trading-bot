"""
Config Hash — Phase 0.8.

Computes a deterministic SHA-256 hash of the full config dict.
The hash is stored with every backtest run, signal, and experiment
so we can always reproduce the exact conditions of any result.

Design:
  - Sorts keys recursively before hashing → same config = same hash.
  - Excludes secrets (API keys, tokens) that come from .env.
  - Returns a short 12-character prefix for display + full hash for storage.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Keys to strip before hashing — secrets must never influence the config hash
_SECRET_KEYS = {
    "bot_token", "api_key", "api_secret", "account_id",
    "chat_id", "token", "password", "secret",
}


def compute_config_hash(config_data: Dict[str, Any]) -> str:
    """
    Compute a stable SHA-256 hex digest of `config_data`.

    Secrets are stripped, keys are sorted, floats are rounded to 8
    decimal places to avoid floating-point noise between platforms.

    Returns the full 64-character hex digest.
    """
    cleaned = _clean(config_data)
    serialised = json.dumps(cleaned, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    logger.debug("[ConfigHash] hash=%s (first 12: %s)", digest, digest[:12])
    return digest


def short_hash(config_data: Dict[str, Any]) -> str:
    """Return the first 12 characters — human-readable in logs and filenames."""
    return compute_config_hash(config_data)[:12]


def hashes_match(config_data: Dict[str, Any], stored_hash: str) -> bool:
    """
    Compare current config against a stored hash from a previous run.
    Returns True if configs are identical (no drift).
    """
    current = compute_config_hash(config_data)
    match = current == stored_hash
    if not match:
        logger.warning(
            "[ConfigHash] Config drift detected! current=%s stored=%s",
            current[:12], stored_hash[:12],
        )
    return match


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _clean(obj: Any) -> Any:
    """
    Recursively strip secrets and normalise values for stable serialisation.
    """
    if isinstance(obj, dict):
        return {
            k: _clean(v)
            for k, v in sorted(obj.items())
            if k.lower() not in _SECRET_KEYS
        }
    if isinstance(obj, list):
        return [_clean(item) for item in obj]
    if isinstance(obj, float):
        return round(obj, 8)
    if isinstance(obj, bool):
        return obj  # bool before int — isinstance(True, int) is True
    if isinstance(obj, int):
        return obj
    if obj is None:
        return None
    return str(obj)
