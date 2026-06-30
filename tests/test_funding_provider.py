"""Tests for core/data/funding_provider.py — funding regime + crowded-side gate."""

import pandas as pd
from core.data.funding_provider import funding_regime, funding_blocks


def _series(rates, start="2026-03-01"):
    idx = pd.date_range(start, periods=len(rates), freq="8h", tz="UTC")
    return pd.DataFrame({"funding_rate": rates}, index=idx)


class TestFundingRegime:
    def test_crowded_long_on_positive_spike(self):
        df = _series([0.0001] * 30 + [0.0010])      # flat-low then a positive extreme
        regime, cur = funding_regime(df, df.index[-1])
        assert regime == "crowded_long"
        assert cur == 0.0010

    def test_crowded_short_on_negative_spike(self):
        df = _series([-0.0001] * 30 + [-0.0010])
        regime, cur = funding_regime(df, df.index[-1])
        assert regime == "crowded_short"
        assert cur == -0.0010

    def test_flat_positive_is_neutral_not_crowded(self):
        # the bug we fixed: a flat positive series must NOT read crowded_long every bar
        df = _series([0.0001] * 31)
        regime, _ = funding_regime(df, df.index[-1])
        assert regime == "neutral"

    def test_high_percentile_but_negative_is_neutral(self):
        # top of range but still <= 0 → not 'crowded_long' (sign guard)
        df = _series([-0.0005] * 30 + [-0.00001])   # least-negative, but still < 0
        regime, _ = funding_regime(df, df.index[-1])
        assert regime != "crowded_long"

    def test_insufficient_history_neutral(self):
        df = _series([0.0001, 0.0009])               # < 10 observations
        regime, cur = funding_regime(df, df.index[-1])
        assert regime == "neutral"
        assert cur == 0.0009

    def test_leakage_free_only_past_used(self):
        # a huge FUTURE spike must not affect the regime at an earlier ts
        df = _series([0.0001] * 20 + [0.0050])        # spike is the LAST row
        ts_before_spike = df.index[19]
        regime, cur = funding_regime(df, ts_before_spike)
        assert cur == 0.0001                          # not the future 0.0050
        assert regime == "neutral"

    def test_none_or_empty(self):
        assert funding_regime(None, pd.Timestamp("2026-03-01", tz="UTC")) == ("neutral", None)
        empty = pd.DataFrame({"funding_rate": []})
        assert funding_regime(empty, pd.Timestamp("2026-03-01", tz="UTC")) == ("neutral", None)


class TestFundingBlocks:
    def test_blocks_crowded_side(self):
        assert funding_blocks("crowded_long", "long") is True
        assert funding_blocks("crowded_short", "short") is True

    def test_allows_contrarian_side(self):
        assert funding_blocks("crowded_long", "short") is False
        assert funding_blocks("crowded_short", "long") is False

    def test_neutral_never_blocks(self):
        assert funding_blocks("neutral", "long") is False
        assert funding_blocks("neutral", "short") is False


class TestFilterHookIntegration:
    """make_filter_hook reads history['funding'] and folds the crowded-side block into
    no_blocking_filters — only when funding_filter is on (default OFF = unchanged)."""

    def _run(self, funding_filter, direction):
        from core.engine.pipeline_hooks import make_filter_hook
        from core.engine.signal_pipeline import PipelineContext
        n = 60
        idx = pd.date_range("2026-04-01", periods=n, freq="5min", tz="UTC")
        df = pd.DataFrame({"open": [100.0]*n, "high": [101.0]*n, "low": [99.0]*n,
                           "close": [100.0]*n, "volume": [1000.0]*n}, index=idx)
        # funding fully BEFORE the test ts; last row is a positive extreme → crowded_long
        f_idx = pd.date_range("2026-03-20", periods=33, freq="8h", tz="UTC")
        fdf = pd.DataFrame({"funding_rate": [0.0001]*32 + [0.0020]}, index=f_idx)
        ctx = PipelineContext(timestamp=idx[-1], bar_index=n-1, symbol="ETHUSDT")
        ctx.direction = direction
        cfg = {"funding_filter": True} if funding_filter else {}
        make_filter_hook(cfg)(ctx, {"timestamp": idx[-1]}, {"5m": df, "funding": fdf})
        return ctx

    def test_funding_off_by_default_no_regime(self):
        ctx = self._run(funding_filter=False, direction="long")
        assert "funding_regime" not in ctx.extra      # branch skipped when OFF

    def test_funding_on_blocks_crowded_long(self):
        ctx = self._run(funding_filter=True, direction="long")
        assert ctx.extra.get("funding_regime") == "crowded_long"
        assert ctx.no_blocking_filters is False        # funding_ok=False forces the AND

    def test_funding_on_allows_contrarian_short(self):
        ctx = self._run(funding_filter=True, direction="short")
        assert ctx.extra.get("funding_regime") == "crowded_long"
        # short into crowded-long is NOT blocked by funding (other filters aside)
