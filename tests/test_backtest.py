"""Unit tests for the walk-forward backtest, using synthetic prices."""

import numpy as np
import pandas as pd
import pytest

import backtest


def _synthetic_prices(days: int = 320, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    specs = {"BTC-USD": (0.0009, 0.03), "ETH-USD": (0.0008, 0.028), "AAPL": (0.0005, 0.012)}
    data = {
        t: 100 * np.exp(np.cumsum(rng.normal(d, v, days))) for t, (d, v) in specs.items()
    }
    return pd.DataFrame(data, index=dates)


def test_walk_forward_returns_all_strategies():
    curves, stats, dsr_info = backtest.walk_forward(
        _synthetic_prices(), lookback_days=150, rebalance_days=30, frequency=365
    )
    assert list(curves.columns) == backtest.STRATEGIES
    assert set(stats["Strategie"]) == set(backtest.STRATEGIES)
    assert len(curves) > 1
    # PSR column and Deflated Sharpe info.
    assert "PSR" in stats.columns
    assert all(0.0 <= p <= 1.0 for p in stats["PSR"])
    assert dsr_info["best"] in backtest.STRATEGIES
    assert 0.0 <= dsr_info["dsr"] <= 1.0


def test_walk_forward_curves_start_at_one():
    curves, _, _ = backtest.walk_forward(
        _synthetic_prices(), lookback_days=150, rebalance_days=30, frequency=365
    )
    for column in curves.columns:
        assert abs(float(curves[column].iloc[0]) - 1.0) < 1e-9


def test_buy_and_hold_btc_matches_btc_path():
    prices = _synthetic_prices()
    curves, _, _ = backtest.walk_forward(
        prices, lookback_days=150, rebalance_days=30, frequency=365
    )
    # Buy&Hold BTC should track BTC's own normalized price over the OOS window.
    btc = prices["BTC-USD"].loc[curves.index]
    btc_norm = btc / btc.iloc[0]
    assert np.allclose(curves["Buy&Hold BTC"].values, btc_norm.values, rtol=1e-6)


def test_walk_forward_raises_without_enough_history():
    with pytest.raises(ValueError):
        backtest.walk_forward(
            _synthetic_prices(days=100), lookback_days=150, rebalance_days=30
        )
