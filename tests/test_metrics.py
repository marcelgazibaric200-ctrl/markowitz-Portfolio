"""Unit tests for the advanced quant metrics."""

import numpy as np
import pandas as pd

import metrics


def _prices(days: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    specs = {"BTC-USD": (0.0009, 0.03), "ETH-USD": (0.0008, 0.028), "AAPL": (0.0005, 0.012)}
    data = {
        t: 100 * np.exp(np.cumsum(rng.normal(d, v, days))) for t, (d, v) in specs.items()
    }
    return pd.DataFrame(data, index=dates)


def test_psr_in_unit_interval_and_grows_with_track_length():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, 1000)
    short = metrics.probabilistic_sharpe_ratio(returns[:60])
    long = metrics.probabilistic_sharpe_ratio(returns)
    assert 0.0 <= short <= 1.0
    assert 0.0 <= long <= 1.0
    assert long > short  # more evidence -> more confident


def test_deflated_sharpe_not_above_probabilistic():
    rng = np.random.default_rng(1)
    best = rng.normal(0.001, 0.01, 500)
    trial_sharpes = [0.02, 0.05, 0.08, 0.01, 0.03]
    psr = metrics.probabilistic_sharpe_ratio(best, 0.0)
    dsr = metrics.deflated_sharpe_ratio(trial_sharpes, best)
    assert dsr <= psr + 1e-9  # deflation only lowers confidence


def test_forward_simulation_bands_ordered_and_stats_valid():
    sim = metrics.forward_simulation(
        _prices(), {"BTC-USD": 0.4, "ETH-USD": 0.3, "AAPL": 0.3},
        horizon_days=90, n_paths=2000,
    )
    bands = sim["bands"]
    assert list(bands.columns) == ["p5", "p25", "p50", "p75", "p95"]
    assert (bands["p5"] <= bands["p50"]).all()
    assert (bands["p50"] <= bands["p95"]).all()
    assert abs(float(bands["p50"].iloc[0]) - 1.0) < 1e-9  # starts at 1
    assert 0.0 <= sim["p_loss"] <= 1.0
    assert sim["cvar5_terminal"] <= sim["var5_terminal"] + 1e-9


def test_pca_risk_explained_sums_to_one_and_effective_bets_bounded():
    pca = metrics.pca_risk(_prices())
    explained = pca["explained"]
    assert abs(float(explained.sum()) - 1.0) < 1e-6
    assert np.all(np.diff(explained) <= 1e-9)  # descending
    assert 1.0 - 1e-9 <= pca["effective_bets"] <= len(pca["labels"]) + 1e-9


def test_detect_regimes_labels_length_and_values():
    prices = _prices()
    labels = metrics.detect_regimes(prices, n_regimes=2)
    assert len(labels) > 0
    assert set(labels.unique()).issubset({"Calm", "Stress"})
