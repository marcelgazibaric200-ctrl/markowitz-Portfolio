"""Efficient frontier visualization with plotly.

Draws the efficient frontier as a curve, the individual assets as points, and
marks the Max Sharpe and Min Volatility portfolios. Optionally plots the
current portfolio from config.CURRENT_PORTFOLIO.
"""

from __future__ import annotations

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from plotly.subplots import make_subplots

import config
import optimize
from optimize import OptimizeResult

OUTPUT_PATH = "efficient_frontier.html"


def _frontier_points(
    result: OptimizeResult,
    n_points: int = 40,
) -> tuple[list[float], list[float]]:
    """Trace the efficient frontier as (volatilities, returns).

    Sweeps target returns from the min-volatility return up to the highest
    single-asset return, solving min variance for each. Infeasible targets are
    skipped.
    """
    r_min = result.min_volatility.expected_return
    r_max = float(result.mu.max())
    if r_max <= r_min:
        return [], []

    vols: list[float] = []
    rets: list[float] = []
    for target in np.linspace(r_min, r_max, n_points):
        ef = optimize._build_ef(
            result.mu,
            result.cov,
            result.max_weight,
            result.sector_mapper,
            result.crypto_cap,
        )
        try:
            ef.efficient_return(target_return=float(target))
        except Exception:
            continue
        ret, vol, _ = ef.portfolio_performance(risk_free_rate=config.RISK_FREE_RATE)
        vols.append(vol)
        rets.append(ret)
    return vols, rets


def build_figure(
    result: OptimizeResult,
    current_weights: dict[str, float] | None = None,
    n_points: int = 40,
) -> go.Figure:
    """Assemble the plotly figure from an optimization result."""
    fig = go.Figure()

    # Efficient frontier curve.
    vols, rets = _frontier_points(result, n_points=n_points)
    if vols:
        fig.add_trace(
            go.Scatter(
                x=vols,
                y=rets,
                mode="lines",
                name="Efficient Frontier",
                line=dict(color="#1f77b4", width=3),
            )
        )

    # Individual assets.
    asset_vols = [float(np.sqrt(result.cov.loc[t, t])) for t in result.mu.index]
    asset_rets = [float(result.mu[t]) for t in result.mu.index]
    fig.add_trace(
        go.Scatter(
            x=asset_vols,
            y=asset_rets,
            mode="markers+text",
            name="Assets",
            text=list(result.mu.index),
            textposition="top center",
            marker=dict(color="#7f7f7f", size=9, symbol="circle-open"),
        )
    )

    # Max Sharpe portfolio.
    if result.max_sharpe is not None:
        fig.add_trace(
            go.Scatter(
                x=[result.max_sharpe.volatility],
                y=[result.max_sharpe.expected_return],
                mode="markers",
                name="Max Sharpe",
                marker=dict(color="#ffb000", size=16, symbol="star"),
            )
        )

    # Min Volatility portfolio.
    fig.add_trace(
        go.Scatter(
            x=[result.min_volatility.volatility],
            y=[result.min_volatility.expected_return],
            mode="markers",
            name="Min Volatility",
            marker=dict(color="#2ca02c", size=13, symbol="diamond"),
        )
    )

    # Current portfolio, if provided.
    if current_weights:
        ret, vol, _ = optimize.performance_of(current_weights, result.mu, result.cov)
        fig.add_trace(
            go.Scatter(
                x=[vol],
                y=[ret],
                mode="markers",
                name="Current Portfolio",
                marker=dict(color="#d62728", size=13, symbol="x"),
            )
        )

    fig.update_layout(
        title="Markowitz Efficient Frontier",
        xaxis_title="Volatility (annualized)",
        yaxis_title="Expected Return (annualized)",
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def show(
    result: OptimizeResult | None = None,
    open_browser: bool = True,
    output_path: str = OUTPUT_PATH,
) -> str:
    """Build the figure and write it to an HTML file, optionally opening it."""
    if result is None:
        result = optimize.run()
    current = config.CURRENT_PORTFOLIO or None
    fig = build_figure(result, current_weights=current)
    fig.write_html(output_path, auto_open=open_browser)
    return output_path


# --- Dashboard panels (dark, used by the Streamlit frontend) ----------------

_MAX_SHARPE_COLOR = "#ffd700"
_MIN_VOL_COLOR = "#00e5ff"
_HRP_COLOR = "#bc8cff"
_CURRENT_COLOR = "#ff7b72"
_RISK_COLOR = "#f85149"
_CVAR_COLOR = "#3fb950"
_SEMI_COLOR = "#ff9e64"
_CML_COLOR = "#8b949e"
_ERC_COLOR = "#e3b341"
_CALM_COLOR = "#3fb950"
_STRESS_COLOR = "#f85149"


def _dark_layout(fig: go.Figure, title: str, **kwargs) -> None:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=50, b=40),
        **kwargs,
    )


def build_frontier_montecarlo(
    result: OptimizeResult,
    n_portfolios: int = 4000,
    seed: int = 42,
    n_points: int = 30,
    current_weights: dict[str, float] | None = None,
) -> go.Figure:
    """Efficient frontier with a Monte Carlo cloud of random portfolios.

    Thousands of random long-only weight vectors are sampled and coloured by
    their Sharpe ratio, with the (constraint-aware) efficient frontier, the
    optimal portfolios, HRP and — if given — the current holdings drawn on top.
    """
    mu = result.mu
    cov = result.cov.values
    rf = config.RISK_FREE_RATE
    n_assets = len(mu)

    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)
    rets = weights @ mu.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    sharpes = np.divide(rets - rf, vols, out=np.zeros_like(rets), where=vols > 0)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=vols,
            y=rets,
            mode="markers",
            name="Random Portfolios",
            hoverinfo="skip",
            marker=dict(
                size=4,
                color=sharpes,
                colorscale="Viridis",
                opacity=0.55,
                showscale=True,
                colorbar=dict(title="Sharpe"),
            ),
        )
    )

    frontier_vols, frontier_rets = _frontier_points(result, n_points=n_points)
    if frontier_vols:
        fig.add_trace(
            go.Scatter(
                x=frontier_vols,
                y=frontier_rets,
                mode="lines",
                name="Efficient Frontier",
                line=dict(color="#f85149", width=3),
            )
        )

    if result.max_sharpe is not None:
        fig.add_trace(
            go.Scatter(
                x=[result.max_sharpe.volatility],
                y=[result.max_sharpe.expected_return],
                mode="markers",
                name="Max Sharpe",
                marker=dict(color=_MAX_SHARPE_COLOR, size=16, symbol="star"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[result.min_volatility.volatility],
            y=[result.min_volatility.expected_return],
            mode="markers",
            name="Min Volatility",
            marker=dict(color=_MIN_VOL_COLOR, size=13, symbol="diamond"),
        )
    )

    if result.hrp is not None:
        fig.add_trace(
            go.Scatter(
                x=[result.hrp.volatility],
                y=[result.hrp.expected_return],
                mode="markers",
                name="HRP",
                marker=dict(color=_HRP_COLOR, size=14, symbol="triangle-up"),
            )
        )

    for portfolio, color, symbol in (
        (result.min_cvar, _CVAR_COLOR, "x-thin"),
        (result.min_semivariance, _SEMI_COLOR, "square"),
        (result.erc, _ERC_COLOR, "pentagon"),
    ):
        if portfolio is not None:
            fig.add_trace(
                go.Scatter(
                    x=[portfolio.volatility],
                    y=[portfolio.expected_return],
                    mode="markers",
                    name=portfolio.name,
                    marker=dict(color=color, size=13, symbol=symbol),
                )
            )

    # Capital Market Line: from the risk-free point through the tangency portfolio.
    if result.max_sharpe is not None and result.max_sharpe.volatility > 0:
        rf = config.RISK_FREE_RATE
        slope = (result.max_sharpe.expected_return - rf) / result.max_sharpe.volatility
        x_end = result.max_sharpe.volatility * 1.4
        fig.add_trace(
            go.Scatter(
                x=[0.0, x_end],
                y=[rf, rf + slope * x_end],
                mode="lines",
                name="Capital Market Line",
                line=dict(color=_CML_COLOR, width=1.5, dash="dash"),
            )
        )

    if current_weights:
        ret, vol, _ = optimize.performance_of(current_weights, result.mu, result.cov)
        fig.add_trace(
            go.Scatter(
                x=[vol],
                y=[ret],
                mode="markers",
                name="Aktuell",
                marker=dict(color=_CURRENT_COLOR, size=16, symbol="x"),
            )
        )

    _dark_layout(
        fig,
        "Efficient Frontier (Monte Carlo)",
        xaxis_title="Volatility (annualized)",
        yaxis_title="Return (annualized)",
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".0%"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def build_correlation_heatmap(prices) -> go.Figure:
    """Heatmap of the correlation matrix of the assets' daily returns."""
    aligned = optimize.align_prices(prices)
    corr = aligned.pct_change().dropna().corr()
    labels = list(corr.columns)
    text = [[f"{value:.2f}" for value in row] for row in corr.values]

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="Viridis",
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="ρ"),
        )
    )
    fig.update_yaxes(autorange="reversed")
    _dark_layout(fig, "Asset Correlation Matrix")
    return fig


def build_normalized_prices(prices) -> go.Figure:
    """Line chart of every asset rebased to 100 at the start of the window."""
    aligned = optimize.align_prices(prices)
    normalized = aligned / aligned.iloc[0] * 100

    fig = go.Figure()
    for column in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index, y=normalized[column], mode="lines", name=column
            )
        )
    _dark_layout(
        fig,
        "Normalized Price History (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Index (start = 100)",
    )
    return fig


def build_weights_bar(result: OptimizeResult) -> go.Figure:
    """Grouped bar chart of the Max Sharpe, Min Volatility and HRP weights."""
    tickers = list(result.mu.index)

    fig = go.Figure()
    if result.max_sharpe is not None:
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=[result.max_sharpe.weights.get(t, 0.0) * 100 for t in tickers],
                name="Max Sharpe",
                marker_color=_MAX_SHARPE_COLOR,
            )
        )
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=[result.min_volatility.weights.get(t, 0.0) * 100 for t in tickers],
            name="Min Volatility",
            marker_color=_MIN_VOL_COLOR,
        )
    )
    for portfolio, color in (
        (result.hrp, _HRP_COLOR),
        (result.min_cvar, _CVAR_COLOR),
        (result.min_semivariance, _SEMI_COLOR),
        (result.erc, _ERC_COLOR),
    ):
        if portfolio is not None:
            fig.add_trace(
                go.Bar(
                    x=tickers,
                    y=[portfolio.weights.get(t, 0.0) * 100 for t in tickers],
                    name=portfolio.name,
                    marker_color=color,
                )
            )
    _dark_layout(fig, "Optimal Weights", barmode="group", yaxis_title="Weight (%)")
    return fig


def build_risk_contribution_bar(
    result: OptimizeResult,
    portfolio=None,
) -> go.Figure:
    """Weight vs. risk contribution per asset for one portfolio.

    Bars sitting higher on risk than on weight are the hidden risk drivers.
    Defaults to Max Sharpe (falling back to Min Volatility).
    """
    if portfolio is None:
        portfolio = result.max_sharpe or result.min_volatility
    tickers = list(result.mu.index)
    contrib = optimize.risk_contributions(portfolio.weights, result.cov)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=[portfolio.weights.get(t, 0.0) * 100 for t in tickers],
            name="Weight",
            marker_color=_MIN_VOL_COLOR,
        )
    )
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=[contrib.get(t, 0.0) * 100 for t in tickers],
            name="Risk Contribution",
            marker_color=_RISK_COLOR,
        )
    )
    _dark_layout(
        fig,
        f"Weight vs. Risk Contribution ({portfolio.name})",
        barmode="group",
        yaxis_title="%",
    )
    return fig


def build_dendrogram(prices) -> go.Figure:
    """Hierarchical clustering tree of the assets by correlation distance.

    Uses the same distance (sqrt((1-rho)/2)) and single linkage that HRP uses
    internally, so this shows exactly the cluster structure HRP allocates over.
    """
    aligned = optimize.align_prices(prices)
    corr = aligned.pct_change().dropna().corr()
    labels = list(corr.columns)
    if len(labels) < 2:
        fig = go.Figure()
        _dark_layout(fig, "Correlation Dendrogram (need >= 2 assets)")
        return fig

    def distfun(matrix):
        return ssd.squareform(np.sqrt(np.clip((1.0 - matrix) / 2.0, 0.0, 1.0)), checks=False)

    def linkagefun(dist):
        return sch.linkage(dist, method="single")

    fig = ff.create_dendrogram(
        corr.values, labels=labels, distfun=distfun, linkagefun=linkagefun
    )
    _dark_layout(fig, "Correlation Dendrogram (HRP clustering)")
    # create_dendrogram sets width/height to inf; make it finite and stretchable.
    fig.update_layout(width=None, height=420, autosize=True)
    fig.update_yaxes(title="Distance  sqrt((1-rho)/2)")
    return fig


def build_rolling_correlation(prices, window: int = 90, base: str | None = None) -> go.Figure:
    """Rolling correlation of every asset to a base asset (BTC by default)."""
    aligned = optimize.align_prices(prices)
    returns = aligned.pct_change().dropna()
    cols = list(returns.columns)
    if base is None or base not in cols:
        base = "BTC-USD" if "BTC-USD" in cols else (cols[0] if cols else None)

    win = max(5, min(window, len(returns) // 2)) if len(returns) else window
    fig = go.Figure()
    if base is not None:
        for col in cols:
            if col == base:
                continue
            rolling = returns[col].rolling(win).corr(returns[base])
            fig.add_trace(
                go.Scatter(x=rolling.index, y=rolling, mode="lines", name=f"{col} vs {base}")
            )
    _dark_layout(
        fig,
        f"Rolling {win}d Correlation to {base}",
        xaxis_title="Date",
        yaxis_title="rho",
        yaxis=dict(range=[-1, 1]),
    )
    return fig


def build_correlation_network(prices, threshold: float = 0.5) -> go.Figure:
    """Assets as nodes on a circle, edges where |correlation| >= threshold.

    Red edges = positive correlation, blue = negative, width scales with |rho|.
    """
    aligned = optimize.align_prices(prices)
    corr = aligned.pct_change().dropna().corr()
    labels = list(corr.columns)
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {labels[i]: (float(np.cos(angles[i])), float(np.sin(angles[i]))) for i in range(n)}

    fig = go.Figure()
    for i in range(n):
        for j in range(i + 1, n):
            c = float(corr.iloc[i, j])
            if abs(c) >= threshold:
                x0, y0 = pos[labels[i]]
                x1, y1 = pos[labels[j]]
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(width=1 + 5 * abs(c), color=_RISK_COLOR if c >= 0 else "#58a6ff"),
                        hoverinfo="text",
                        text=f"{labels[i]}-{labels[j]}: {c:.2f}",
                        showlegend=False,
                    )
                )
    fig.add_trace(
        go.Scatter(
            x=[pos[l][0] for l in labels],
            y=[pos[l][1] for l in labels],
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=20, color="#f0883e"),
            name="Assets",
            showlegend=False,
        )
    )
    _dark_layout(
        fig,
        f"Correlation Network (|rho| >= {threshold:.2f})",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def build_backtest_curves(curves) -> go.Figure:
    """Line chart of out-of-sample growth of $1 per strategy (from backtest)."""
    fig = go.Figure()
    for column in curves.columns:
        fig.add_trace(
            go.Scatter(x=curves.index, y=curves[column], mode="lines", name=column)
        )
    _dark_layout(
        fig,
        "Out-of-Sample Backtest (Growth of $1)",
        xaxis_title="Date",
        yaxis_title="Portfolio value (start = 1)",
    )
    return fig


def build_forward_fanchart(bands) -> go.Figure:
    """Fan chart of simulated forward portfolio paths (percentile bands)."""
    x = list(bands.index)
    fig = go.Figure()
    # Outer band 5-95, inner band 25-75, both as filled areas, plus the median.
    for lo, hi, color in (("p5", "p95", "rgba(88,166,255,0.15)"), ("p25", "p75", "rgba(88,166,255,0.30)")):
        fig.add_trace(go.Scatter(x=x, y=bands[hi], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(
            go.Scatter(
                x=x, y=bands[lo], mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor=color, name=f"{lo[1:]}-{hi[1:]}%", hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(x=x, y=bands["p50"], mode="lines", name="Median", line=dict(color="#58a6ff", width=2))
    )
    _dark_layout(
        fig,
        "Forward Simulation (Growth of $1)",
        xaxis_title="Days ahead",
        yaxis_title="Portfolio value (start = 1)",
    )
    return fig


def build_regime_correlations(corrs: dict) -> go.Figure:
    """Side-by-side correlation heatmaps per market regime (e.g. Calm vs Stress)."""
    labels_order = [k for k in ("Calm", "Stress") if k in corrs] or list(corrs)
    fig = make_subplots(rows=1, cols=len(labels_order), subplot_titles=labels_order)
    for i, label in enumerate(labels_order, start=1):
        corr = corrs[label]
        fig.add_trace(
            go.Heatmap(
                z=corr.values, x=list(corr.columns), y=list(corr.columns),
                zmin=-1, zmax=1, colorscale="Viridis", showscale=(i == len(labels_order)),
                colorbar=dict(title="rho"),
            ),
            row=1, col=i,
        )
        fig.update_yaxes(autorange="reversed", row=1, col=i)
    _dark_layout(fig, "Correlation by Regime")
    return fig


def build_regime_timeline(prices, labels) -> go.Figure:
    """Market-proxy price over time with points coloured by volatility regime."""
    aligned = optimize.align_prices(prices)
    if "BTC-USD" in aligned.columns:
        proxy = aligned["BTC-USD"]
    else:
        proxy = (aligned / aligned.iloc[0]).mean(axis=1)
    common = proxy.index.intersection(labels.index)
    proxy = proxy.loc[common]
    proxy = proxy / proxy.iloc[0] * 100
    regimes = labels.loc[common]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=proxy.index, y=proxy.values, mode="lines", name="Market proxy",
                   line=dict(color="#8b949e", width=1))
    )
    for label, color in (("Calm", _CALM_COLOR), ("Stress", _STRESS_COLOR)):
        mask = regimes == label
        if mask.any():
            fig.add_trace(
                go.Scatter(x=proxy.index[mask], y=proxy.values[mask], mode="markers",
                           name=label, marker=dict(color=color, size=5))
            )
    _dark_layout(fig, "Volatility Regimes", xaxis_title="Date", yaxis_title="Proxy (start = 100)")
    return fig


def build_pca_scree(pca: dict) -> go.Figure:
    """Explained variance per principal component with a cumulative line."""
    n = len(pca["explained"])
    x = [f"PC{i + 1}" for i in range(n)]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=x, y=pca["explained"] * 100, name="Explained %", marker_color=_MIN_VOL_COLOR)
    )
    fig.add_trace(
        go.Scatter(x=x, y=pca["cumulative"] * 100, name="Cumulative %", mode="lines+markers",
                   line=dict(color="#f0883e", width=2))
    )
    _dark_layout(
        fig,
        f"PCA Scree  (effective bets: {pca['effective_bets']:.2f})",
        yaxis_title="Variance explained (%)",
    )
    return fig
