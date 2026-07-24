"""Unit tests for the Markowitz optimization, using synthetic prices."""

import numpy as np
import pandas as pd

import optimize


def _synthetic_prices(days: int = 300, seed: int = 42) -> pd.DataFrame:
    """Three assets with different drift/vol as geometric random walks."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    specs = {
        "AAA": (0.0010, 0.02),  # high drift, high vol
        "BBB": (0.0005, 0.01),  # medium drift, medium vol
        "CCC": (0.0002, 0.005),  # low drift, low vol
    }
    data = {}
    for ticker, (drift, vol) in specs.items():
        returns = rng.normal(drift, vol, days)
        data[ticker] = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(data, index=dates)


def test_weights_sum_to_one_and_are_long_only():
    prices = _synthetic_prices()
    mu, cov, freq = optimize.compute_inputs(prices, frequency=365)

    for portfolio in (optimize.max_sharpe(mu, cov), optimize.min_volatility(mu, cov)):
        total = sum(portfolio.weights.values())
        assert abs(total - 1.0) < 1e-4
        assert all(w >= -1e-9 for w in portfolio.weights.values())


def test_max_sharpe_beats_min_vol_on_sharpe():
    prices = _synthetic_prices()
    mu, cov, freq = optimize.compute_inputs(prices, frequency=365)

    ms = optimize.max_sharpe(mu, cov)
    mv = optimize.min_volatility(mu, cov)

    assert ms.volatility > 0
    assert mv.volatility > 0
    # The tangency portfolio should have the higher Sharpe ratio.
    assert ms.sharpe >= mv.sharpe - 1e-6


def test_run_with_passed_prices_populates_result():
    prices = _synthetic_prices()
    result = optimize.run(prices)

    assert result.min_volatility is not None
    assert result.frequency in (252, 365)
    assert set(result.mu.index) == {"AAA", "BBB", "CCC"}


def test_align_prices_drops_incomplete_rows():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {
            "BTC-USD": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "AAPL": [50.0, 51.0, np.nan, np.nan, 52.0, 53.0],  # gaps like weekends
        },
        index=dates,
    )
    aligned = optimize.align_prices(prices)

    assert int(aligned.isna().sum().sum()) == 0
    assert len(aligned) == 4


def test_mixed_universe_optimizes():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-01-01", periods=250, freq="D")
    btc = 100 * np.exp(np.cumsum(rng.normal(0.0010, 0.02, 250)))
    aapl = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.012, 250)))
    # Blank out weekends to mimic a stock's trading calendar.
    aapl[dates.weekday >= 5] = np.nan
    prices = pd.DataFrame({"BTC-USD": btc, "AAPL": aapl}, index=dates)

    result = optimize.run(prices)

    total = sum(result.min_volatility.weights.values())
    assert abs(total - 1.0) < 1e-4
    assert not result.mu.isna().any()
    assert int(result.cov.isna().sum().sum()) == 0


def _mixed_prices(days: int = 260, seed: int = 5) -> pd.DataFrame:
    """Two crypto + one stock, all daily (no NaN) for deterministic caps tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    specs = {"BTC-USD": (0.0009, 0.03), "ETH-USD": (0.0008, 0.028), "AAPL": (0.0004, 0.012)}
    data = {
        t: 100 * np.exp(np.cumsum(rng.normal(d, v, days))) for t, (d, v) in specs.items()
    }
    return pd.DataFrame(data, index=dates)


_MIXED_TYPES = {"BTC-USD": "crypto", "ETH-USD": "crypto", "AAPL": "stock"}


def test_max_weight_cap_is_respected():
    prices = _mixed_prices()
    result = optimize.run(prices, asset_types=_MIXED_TYPES, max_weight=0.4)

    for portfolio in (result.min_volatility, result.max_sharpe):
        if portfolio is None:
            continue
        assert max(portfolio.weights.values()) <= 0.4 + 1e-3


def test_crypto_sector_cap_is_respected():
    prices = _mixed_prices()
    result = optimize.run(prices, asset_types=_MIXED_TYPES, crypto_cap=0.5)

    crypto_weight = sum(
        w for t, w in result.min_volatility.weights.items() if _MIXED_TYPES[t] == "crypto"
    )
    assert crypto_weight <= 0.5 + 1e-3


def test_hrp_weights_are_valid():
    result = optimize.run(_synthetic_prices())
    assert result.hrp is not None
    assert abs(sum(result.hrp.weights.values()) - 1.0) < 1e-4
    assert all(w >= -1e-9 for w in result.hrp.weights.values())


def test_risk_contributions_sum_to_one():
    prices = _synthetic_prices()
    mu, cov, _ = optimize.compute_inputs(prices, frequency=365)
    mv = optimize.min_volatility(mu, cov)

    contrib = optimize.risk_contributions(mv.weights, cov)
    assert abs(sum(contrib.values()) - 1.0) < 1e-6
    assert set(contrib) == set(cov.columns)


def test_exp_cov_method_differs_from_ledoit_wolf():
    prices = _synthetic_prices()
    _, cov_lw, _ = optimize.compute_inputs(prices, frequency=365, cov_method="ledoit_wolf")
    _, cov_ew, _ = optimize.compute_inputs(prices, frequency=365, cov_method="exp_cov")
    assert not np.allclose(cov_lw.values, cov_ew.values)


def test_downside_metrics_shape_and_drawdown_sign():
    prices = _synthetic_prices()
    mu, cov, freq = optimize.compute_inputs(prices, frequency=365)
    mv = optimize.min_volatility(mu, cov)

    metrics = optimize.downside_metrics(prices, mv.weights, freq)
    assert set(metrics) == {"sortino", "max_drawdown", "var95", "cvar95"}
    assert metrics["max_drawdown"] <= 0.0
    assert metrics["cvar95"] >= metrics["var95"] - 1e-9  # tail loss >= VaR


def test_rebalance_conserves_total_and_hits_target():
    target = {"BTC-USD": 0.5, "ETH-USD": 0.3, "AAPL": 0.2}
    quantities = {"BTC-USD": 1.0, "ETH-USD": 2.0, "AAPL": 10.0}
    prices = {"BTC-USD": 60000.0, "ETH-USD": 3000.0, "AAPL": 200.0}

    rows, total = optimize.rebalance(target, quantities, prices)
    assert abs(total - (60000 + 6000 + 2000)) < 1e-6

    # After applying the deltas each asset sits on its target weight.
    for row in rows:
        new_value = row["current_value"] + row["delta_value"]
        assert abs(new_value - target[row["ticker"]] * total) < 1e-6
