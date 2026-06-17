"""Percentage-of-price cost model — fixes cheap-coin auto-rejection (grade D).

The default absolute costs are gold-calibrated; a $0.08 coin can't share them.
Percent mode scales spread/slippage with price so every asset pays the same
fraction. Gold keeps absolute mode (must be byte-for-byte unchanged).
"""

from core.risk.rr_calculator import RRCalculator
from core.filters.spread_filter import SpreadFilter

RISK = {"rr_tiers": {"min_to_enter": 2.0}}
PCT = {"cost_model": "percent", "spread_pct": 0.0002, "slippage_pct": 0.0002}


def _net(rr, entry):
    # short: risk = 1% of price, reward = 3.5% of price (geometry scales with price)
    return rr.calculate("short", entry, entry * 1.01, entry * 0.965).net_rr


def test_percent_cost_is_price_invariant():
    rr = RRCalculator(RISK, PCT)
    # identical geometry at $1800 and $0.08 must yield the SAME net R:R
    assert abs(_net(rr, 1800.0) - _net(rr, 0.08)) < 1e-6
    assert _net(rr, 0.08) > 2.0  # cheap coin no longer collapses


def test_absolute_cost_collapses_cheap_coin_but_percent_fixes_it():
    rr_abs = RRCalculator(RISK, {"default_spread": 0.25, "default_slippage": 0.10})
    # the bug: gold-sized absolute cost destroys a cheap coin's net R:R
    res_cheap = rr_abs.calculate("short", 0.08, 0.0808, 0.0772)
    assert not res_cheap.valid
    # the fix: percent mode keeps it valid
    rr_pct = RRCalculator(RISK, PCT)
    assert rr_pct.calculate("short", 0.08, 0.0808, 0.0772).valid


def test_absolute_mode_unchanged_for_gold():
    # gold geometry under the unchanged absolute model still grades fine
    rr_abs = RRCalculator(RISK, {"default_spread": 0.25, "default_slippage": 0.10})
    res = rr_abs.calculate("short", 4300.0, 4304.0, 4286.0)
    assert res.valid and res.net_rr > 2.0


def test_spread_filter_percent_allows_cheap_coin():
    sf = SpreadFilter({"cost_model": "percent", "spread_pct": 0.0002, "max_spread_atr_ratio": 0.15})
    # DOGE-like: tiny ATR but spread scales with price → allowed
    assert sf.is_trade_allowed(spread=None, atr=0.0002, price=0.0876) is True


def test_spread_filter_absolute_blocks_cheap_coin():
    sf = SpreadFilter({"default_spread": 0.25, "max_spread_atr_ratio": 0.15})
    assert sf.is_trade_allowed(spread=None, atr=4.16) is True       # gold: ok
    assert sf.is_trade_allowed(spread=None, atr=0.0002) is False    # cheap coin: blocked (the bug)


def test_percent_mode_bad_price_does_not_revert_to_gold_default():
    # NaN/None price in percent mode must NOT fall back to the gold 0.25 default
    # (that would re-block the cheap coins this mode exists to support).
    sf = SpreadFilter({"cost_model": "percent", "spread_pct": 0.0002, "max_spread_atr_ratio": 0.15})
    assert sf.is_trade_allowed(spread=None, atr=0.0002, price=float("nan")) is True
    assert sf.is_trade_allowed(spread=None, atr=0.0002, price=None) is True
