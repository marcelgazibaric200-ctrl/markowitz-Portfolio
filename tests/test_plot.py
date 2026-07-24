"""Unit tests for the plotting layer. Builds a figure, never opens a browser."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import optimize
import plot


def _synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=250, freq="D")
    data = {
        "AAA": 100 * np.exp(np.cumsum(rng.normal(0.0010, 0.02, 250))),
        "BBB": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 250))),
        "CCC": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.005, 250))),
    }
    return pd.DataFrame(data, index=dates)


def _synthetic_result() -> optimize.OptimizeResult:
    return optimize.run(_synthetic_prices())


def test_build_figure_has_traces():
    fig = plot.build_figure(_synthetic_result(), n_points=12)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_frontier_trace_has_points():
    fig = plot.build_figure(_synthetic_result(), n_points=12)
    names = [trace.name for trace in fig.data]
    assert "Efficient Frontier" in names
    frontier = next(t for t in fig.data if t.name == "Efficient Frontier")
    assert len(frontier.x) > 1


def test_current_portfolio_adds_trace():
    result = _synthetic_result()
    without = plot.build_figure(result, n_points=12)
    with_current = plot.build_figure(
        result, current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, n_points=12
    )
    assert len(with_current.data) == len(without.data) + 1


def test_frontier_montecarlo_has_cloud_and_frontier():
    fig = plot.build_frontier_montecarlo(_synthetic_result(), n_portfolios=500, n_points=12)
    names = [trace.name for trace in fig.data]
    assert "Random Portfolios" in names
    assert "Efficient Frontier" in names
    cloud = next(t for t in fig.data if t.name == "Random Portfolios")
    assert len(cloud.x) == 500


def test_correlation_heatmap_is_square():
    fig = plot.build_correlation_heatmap(_synthetic_prices())
    heatmap = fig.data[0]
    assert heatmap.type == "heatmap"
    assert len(heatmap.x) == 3
    assert len(heatmap.z) == 3


def test_normalized_prices_starts_at_100():
    fig = plot.build_normalized_prices(_synthetic_prices())
    assert len(fig.data) == 3
    for trace in fig.data:
        assert abs(trace.y[0] - 100.0) < 1e-6


def test_weights_bar_has_both_portfolios():
    result = _synthetic_result()
    fig = plot.build_weights_bar(result)
    names = [trace.name for trace in fig.data]
    assert "Min Volatility" in names
    for trace in fig.data:
        assert len(trace.x) == 3


def test_weights_bar_includes_hrp():
    result = _synthetic_result()
    fig = plot.build_weights_bar(result)
    assert "HRP" in [trace.name for trace in fig.data]


def test_frontier_montecarlo_marks_current_portfolio():
    result = _synthetic_result()
    fig = plot.build_frontier_montecarlo(
        result, n_portfolios=200, current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    )
    assert "Aktuell" in [trace.name for trace in fig.data]
    assert "HRP" in [trace.name for trace in fig.data]


def test_risk_contribution_bar_has_two_series():
    fig = plot.build_risk_contribution_bar(_synthetic_result())
    names = [trace.name for trace in fig.data]
    assert "Weight" in names
    assert "Risk Contribution" in names


def test_dendrogram_returns_finite_figure():
    fig = plot.build_dendrogram(_synthetic_prices())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.layout.height == 420  # finite so Streamlit can render it


def test_rolling_correlation_has_traces():
    fig = plot.build_rolling_correlation(_synthetic_prices(), window=30)
    # Three assets -> two lines against the base asset.
    assert len(fig.data) == 2


def test_correlation_network_has_node_trace():
    fig = plot.build_correlation_network(_synthetic_prices(), threshold=0.0)
    assert any(trace.name == "Assets" for trace in fig.data)
