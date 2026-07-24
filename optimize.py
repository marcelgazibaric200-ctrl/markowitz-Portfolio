"""Markowitz optimization using PyPortfolioOpt with Ledoit-Wolf shrinkage.

Computes annualized expected returns and a shrunk covariance matrix from the
stored prices, then solves for two portfolios: maximum Sharpe ratio and
minimum volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as _sch
from scipy.optimize import minimize
from pypfopt import (
    EfficientCVaR,
    EfficientFrontier,
    EfficientSemivariance,
    HRPOpt,
    expected_returns,
    risk_models,
)
from pypfopt.exceptions import OptimizationError
from pypfopt.risk_models import CovarianceShrinkage

import config
import db

# PyPortfolioOpt 1.6.0 validates the HRP linkage method against
# scipy.cluster.hierarchy._LINKAGE_METHODS, which scipy >= 1.15 dropped from the
# public namespace. Restore it so HRPOpt.optimize keeps working; scipy's
# linkage() still accepts these method names.
if not hasattr(_sch, "_LINKAGE_METHODS"):
    _sch._LINKAGE_METHODS = {
        "single": 0, "complete": 1, "average": 2, "weighted": 3,
        "centroid": 4, "median": 5, "ward": 6,
    }


@dataclass
class PortfolioResult:
    """One optimized portfolio: weights plus annualized performance.

    `tail_metric` optionally carries the native tail risk of the method (annual
    CVaR for Min CVaR, semi-deviation for Min Semivariance); None otherwise.
    """

    name: str
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float
    tail_metric: float | None = None


@dataclass
class OptimizeResult:
    """Full optimization output, reused by the plotting layer."""

    mu: pd.Series
    cov: pd.DataFrame
    frequency: int
    min_volatility: PortfolioResult
    max_sharpe: PortfolioResult | None = None
    max_sharpe_error: str | None = None
    hrp: PortfolioResult | None = None
    hrp_error: str | None = None
    min_cvar: PortfolioResult | None = None
    min_cvar_error: str | None = None
    min_semivariance: PortfolioResult | None = None
    min_semivariance_error: str | None = None
    erc: PortfolioResult | None = None
    erc_error: str | None = None
    cov_method: str = "ledoit_wolf"
    return_method: str = "mean_historical"
    denoise: bool = False
    max_weight: float = 1.0
    crypto_cap: float = 1.0
    sector_mapper: dict[str, str] | None = None


def align_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep only dates where every asset has a price.

    Crypto trades daily, stocks only on business days. Dropping incomplete rows
    aligns everything to the common trading calendar, so returns and covariances
    are computed on matching dates. For a crypto-only universe this is a no-op.
    """
    return prices.dropna(how="any")


def _estimate_returns(prices: pd.DataFrame, frequency: int, method: str) -> pd.Series:
    """Annualized expected returns via the chosen estimator."""
    if method == "ema":
        return expected_returns.ema_historical_return(prices, frequency=frequency)
    if method == "capm":
        return expected_returns.capm_return(
            prices, frequency=frequency, risk_free_rate=config.RISK_FREE_RATE
        )
    return expected_returns.mean_historical_return(prices, frequency=frequency)


def denoise_covariance(cov: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Filter noise eigenvalues out of a covariance matrix (RMT / Lopez de Prado).

    Converts to correlation, replaces every eigenvalue below the Marchenko-Pastur
    upper bound lambda+ = (1 + sqrt(n/T))^2 with their average ("constant residual
    eigenvalue"), rebuilds the correlation matrix and scales back to covariance.
    Keeps the signal eigenvalues, denoises the rest. No-op when it cannot help.
    """
    n = cov.shape[0]
    if n < 2 or n_samples <= n:
        return cov
    stds = np.sqrt(np.diag(cov.values))
    corr = risk_models.cov_to_corr(cov).values

    eigvals, eigvecs = np.linalg.eigh(corr)
    q = n_samples / n
    lambda_plus = (1.0 + np.sqrt(1.0 / q)) ** 2

    noise = eigvals < lambda_plus
    if noise.sum() > 1:
        eigvals = eigvals.copy()
        eigvals[noise] = eigvals[noise].mean()

    denoised = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(denoised))
    denoised = denoised / np.outer(d, d)  # renormalize to unit diagonal
    cov_denoised = denoised * np.outer(stds, stds)
    return pd.DataFrame(cov_denoised, index=cov.index, columns=cov.columns)


def compute_inputs(
    prices: pd.DataFrame,
    frequency: int | None = None,
    cov_method: str = "ledoit_wolf",
    return_method: str = "mean_historical",
    denoise: bool = False,
) -> tuple[pd.Series, pd.DataFrame, int]:
    """Return (mu, cov, frequency): annualized returns and covariance.

    `cov_method` selects the covariance estimator ("ledoit_wolf" shrinkage or
    "exp_cov"), `return_method` the mu estimator ("mean_historical", "ema",
    "capm"), and `denoise` optionally applies RMT denoising to the covariance.
    """
    prices = align_prices(prices)
    if len(prices) < 2:
        raise ValueError(
            "Not enough aligned price rows to optimize. Check that the assets "
            "share overlapping dates (try `python main.py fetch`)."
        )
    if frequency is None:
        frequency = config.trading_days_per_year()
    mu = _estimate_returns(prices, frequency, return_method)
    if cov_method == "exp_cov":
        cov = risk_models.exp_cov(prices, frequency=frequency)
    else:
        cov = CovarianceShrinkage(prices, frequency=frequency).ledoit_wolf()
    if denoise:
        cov = denoise_covariance(cov, n_samples=len(prices))
    return mu, cov, frequency


def _apply_caps(
    opt,
    n_assets: int,
    max_weight: float = 1.0,
    sector_mapper: dict[str, str] | None = None,
    crypto_cap: float = 1.0,
) -> None:
    """Add the per-asset and crypto-sector caps to any convex optimizer.

    EfficientFrontier, EfficientCVaR and EfficientSemivariance all inherit
    add_constraint / add_sector_constraints, so the same caps apply everywhere.
    """
    if max_weight < 1.0:
        cap = max(max_weight, 1.0 / n_assets)  # keep feasible: weights sum to 1
        opt.add_constraint(lambda w: w <= cap)
    if sector_mapper and crypto_cap < 1.0:
        kinds = set(sector_mapper.values())
        # A crypto cap only makes sense if something non-crypto can absorb the
        # rest; otherwise the weights could never sum to 1.
        if "crypto" in kinds and len(kinds) > 1:
            opt.add_sector_constraints(sector_mapper, {}, {"crypto": crypto_cap})


def _build_ef(
    mu: pd.Series,
    cov: pd.DataFrame,
    max_weight: float = 1.0,
    sector_mapper: dict[str, str] | None = None,
    crypto_cap: float = 1.0,
) -> EfficientFrontier:
    """Fresh EfficientFrontier with the optional per-asset and crypto caps.

    PyPortfolioOpt EfficientFrontier objects are single-use, so every solve
    builds its own via this helper. Keeping it central means the Max Sharpe,
    Min Volatility and the drawn frontier all honour the same constraints.
    """
    ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    _apply_caps(ef, len(mu), max_weight, sector_mapper, crypto_cap)
    return ef


def _performance(ef: EfficientFrontier) -> tuple[float, float, float]:
    return ef.portfolio_performance(risk_free_rate=config.RISK_FREE_RATE)


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    max_weight: float = 1.0,
    sector_mapper: dict[str, str] | None = None,
    crypto_cap: float = 1.0,
) -> PortfolioResult:
    """Portfolio with the best return-per-risk (tangency portfolio)."""
    ef = _build_ef(mu, cov, max_weight, sector_mapper, crypto_cap)
    ef.max_sharpe(risk_free_rate=config.RISK_FREE_RATE)
    weights = ef.clean_weights()
    ret, vol, sharpe = _performance(ef)
    return PortfolioResult("Max Sharpe", dict(weights), ret, vol, sharpe)


def min_volatility(
    mu: pd.Series,
    cov: pd.DataFrame,
    max_weight: float = 1.0,
    sector_mapper: dict[str, str] | None = None,
    crypto_cap: float = 1.0,
) -> PortfolioResult:
    """Portfolio with the lowest possible risk."""
    ef = _build_ef(mu, cov, max_weight, sector_mapper, crypto_cap)
    ef.min_volatility()
    weights = ef.clean_weights()
    ret, vol, sharpe = _performance(ef)
    return PortfolioResult("Min Volatility", dict(weights), ret, vol, sharpe)


def hierarchical_risk_parity(
    prices: pd.DataFrame,
    mu: pd.Series,
    cov: pd.DataFrame,
) -> PortfolioResult:
    """Hierarchical Risk Parity allocation (Lopez de Prado).

    Clusters assets by their correlation, then allocates by inverse variance
    down the tree. No matrix inversion and no constraints, so it stays robust
    when assets are highly correlated (typical for crypto). Performance is
    scored with `performance_of` on the same annualized mu/cov as the other
    portfolios for a consistent comparison.
    """
    returns = align_prices(prices).pct_change().dropna()
    model = HRPOpt(returns)
    model.optimize()
    weights = dict(model.clean_weights())
    ret, vol, sharpe = performance_of(weights, mu, cov)
    return PortfolioResult("HRP", weights, ret, vol, sharpe)


def min_cvar(
    mu: pd.Series,
    returns: pd.DataFrame,
    cov: pd.DataFrame,
    max_weight: float = 1.0,
    sector_mapper: dict[str, str] | None = None,
    crypto_cap: float = 1.0,
    beta: float = 0.95,
) -> PortfolioResult:
    """Portfolio that minimizes Conditional Value at Risk (expected tail loss).

    Scored with performance_of on the same mu/cov as the others (so it plots on
    the shared axis); the native annual CVaR is kept in tail_metric.
    """
    ec = EfficientCVaR(mu, returns, beta=beta, weight_bounds=(0, 1), solver="CLARABEL")
    _apply_caps(ec, len(mu), max_weight, sector_mapper, crypto_cap)
    ec.min_cvar()
    weights = dict(ec.clean_weights())
    _, cvar = ec.portfolio_performance()
    ret, vol, sharpe = performance_of(weights, mu, cov)
    return PortfolioResult("Min CVaR", weights, ret, vol, sharpe, tail_metric=float(cvar))


def min_semivariance(
    mu: pd.Series,
    returns: pd.DataFrame,
    cov: pd.DataFrame,
    frequency: int,
    max_weight: float = 1.0,
    sector_mapper: dict[str, str] | None = None,
    crypto_cap: float = 1.0,
) -> PortfolioResult:
    """Portfolio that minimizes downside (semivariance) risk.

    Only penalizes returns below the benchmark, unlike variance. Scored with
    performance_of; native semi-deviation kept in tail_metric.
    """
    es = EfficientSemivariance(
        mu, returns, frequency=frequency, weight_bounds=(0, 1), solver="CLARABEL"
    )
    _apply_caps(es, len(mu), max_weight, sector_mapper, crypto_cap)
    es.min_semivariance()
    weights = dict(es.clean_weights())
    _, semi_dev, _ = es.portfolio_performance(risk_free_rate=config.RISK_FREE_RATE)
    ret, vol, sharpe = performance_of(weights, mu, cov)
    return PortfolioResult(
        "Min Semivariance", weights, ret, vol, sharpe, tail_metric=float(semi_dev)
    )


def equal_risk_contribution(mu: pd.Series, cov: pd.DataFrame) -> PortfolioResult:
    """Equal Risk Contribution (ERC / true risk parity) portfolio.

    Every asset contributes the same share of portfolio risk. Solves the convex
    log-barrier formulation (Maillard/Spinu) min 0.5 w'Sigma w - (1/n) sum ln(w),
    whose stationarity condition w_i (Sigma w)_i = const equalizes the risk
    contributions; the solution is then normalized to sum to 1. Unconstrained by
    design (like HRP).
    """
    sigma = cov.values
    n = sigma.shape[0]
    inv_vol = 1.0 / np.sqrt(np.diag(sigma))
    x0 = inv_vol / inv_vol.sum()

    def objective(w):
        return 0.5 * w @ sigma @ w - np.sum(np.log(w)) / n

    def gradient(w):
        return sigma @ w - 1.0 / (n * w)

    res = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=[(1e-8, None)] * n,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = res.x / res.x.sum()
    weights = {t: float(wi) for t, wi in zip(cov.columns, w)}
    ret, vol, sharpe = performance_of(weights, mu, cov)
    return PortfolioResult("ERC", weights, ret, vol, sharpe)


def run(
    prices: pd.DataFrame | None = None,
    frequency: int | None = None,
    asset_types: dict[str, str] | None = None,
    max_weight: float = 1.0,
    crypto_cap: float = 1.0,
    cov_method: str | None = None,
    return_method: str | None = None,
    denoise: bool | None = None,
) -> OptimizeResult:
    """Load prices if needed, compute inputs, and solve the portfolios.

    Pass `frequency` to override the annualization factor. `asset_types`
    ({ticker: "crypto"|"stock"}) enables the crypto sector cap. `max_weight`
    caps any single asset (e.g. BTC), `crypto_cap` caps the combined crypto
    weight, `cov_method` picks the covariance estimator.

    Min volatility always solves. Max Sharpe can be infeasible (no asset beats
    the risk-free rate, or the caps are too tight); that case is captured, not
    raised. HRP is unconstrained by design and solved separately.
    """
    if cov_method is None:
        cov_method = config.COV_METHOD
    if return_method is None:
        return_method = config.RETURN_METHOD
    if denoise is None:
        denoise = config.DENOISE_COV
    if prices is None:
        conn = db.connect()
        prices = db.load_prices(conn, tickers=config.ASSETS)
        conn.close()
    if prices.empty:
        raise ValueError("No price data found. Run `python main.py fetch` first.")

    mu, cov, frequency = compute_inputs(
        prices,
        frequency=frequency,
        cov_method=cov_method,
        return_method=return_method,
        denoise=denoise,
    )
    returns = align_prices(prices).pct_change().dropna()
    sector_mapper = asset_types or None
    result = OptimizeResult(
        mu=mu,
        cov=cov,
        frequency=frequency,
        min_volatility=min_volatility(mu, cov, max_weight, sector_mapper, crypto_cap),
        cov_method=cov_method,
        return_method=return_method,
        denoise=denoise,
        max_weight=max_weight,
        crypto_cap=crypto_cap,
        sector_mapper=sector_mapper,
    )
    try:
        result.max_sharpe = max_sharpe(mu, cov, max_weight, sector_mapper, crypto_cap)
    except (ValueError, OptimizationError) as exc:
        result.max_sharpe_error = str(exc)
    try:
        result.hrp = hierarchical_risk_parity(prices, mu, cov)
    except (ValueError, OptimizationError) as exc:
        result.hrp_error = str(exc)
    try:
        result.min_cvar = min_cvar(mu, returns, cov, max_weight, sector_mapper, crypto_cap)
    except (ValueError, OptimizationError) as exc:
        result.min_cvar_error = str(exc)
    try:
        result.min_semivariance = min_semivariance(
            mu, returns, cov, frequency, max_weight, sector_mapper, crypto_cap
        )
    except (ValueError, OptimizationError) as exc:
        result.min_semivariance_error = str(exc)
    try:
        result.erc = equal_risk_contribution(mu, cov)
    except (ValueError, OptimizationError) as exc:
        result.erc_error = str(exc)
    return result


def performance_of(
    weights: dict[str, float],
    mu: pd.Series,
    cov: pd.DataFrame,
) -> tuple[float, float, float]:
    """Return (return, volatility, sharpe) for an arbitrary weight vector."""
    w = pd.Series(weights, dtype=float).reindex(mu.index).fillna(0.0)
    ret = float(w @ mu)
    vol = float(np.sqrt(w.values @ cov.values @ w.values))
    sharpe = (ret - config.RISK_FREE_RATE) / vol if vol > 0 else float("nan")
    return ret, vol, sharpe


def risk_contributions(
    weights: dict[str, float],
    cov: pd.DataFrame,
) -> dict[str, float]:
    """Each asset's share of total portfolio variance (sums to 1).

    A weight can look small yet dominate risk (or vice versa). The contribution
    is w_i * (cov @ w)_i / (w' cov w).
    """
    tickers = list(cov.columns)
    w = pd.Series(weights, dtype=float).reindex(tickers).fillna(0.0).values
    port_var = float(w @ cov.values @ w)
    if port_var <= 0:
        return {t: 0.0 for t in tickers}
    contrib = w * (cov.values @ w) / port_var
    return {t: float(c) for t, c in zip(tickers, contrib)}


def downside_metrics(
    prices: pd.DataFrame,
    weights: dict[str, float],
    frequency: int,
    risk_free_rate: float | None = None,
) -> dict[str, float]:
    """Downside risk view of a weighted portfolio.

    Returns Sortino ratio, max drawdown, and historical 95% VaR/CVaR (reported
    as positive daily loss fractions).
    """
    rf = config.RISK_FREE_RATE if risk_free_rate is None else risk_free_rate
    returns = align_prices(prices).pct_change().dropna()
    tickers = list(returns.columns)
    w = pd.Series(weights, dtype=float).reindex(tickers).fillna(0.0).values
    port = returns.values @ w
    if len(port) == 0:
        return {"sortino": float("nan"), "max_drawdown": 0.0, "var95": 0.0, "cvar95": 0.0}

    ann_return = float(np.mean(port) * frequency)
    downside = port[port < 0]
    downside_dev = (
        float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(frequency))
        if len(downside)
        else 0.0
    )
    sortino = (ann_return - rf) / downside_dev if downside_dev > 0 else float("nan")

    cumulative = np.cumprod(1 + port)
    peak = np.maximum.accumulate(cumulative)
    max_drawdown = float((cumulative / peak - 1).min())

    cutoff = np.percentile(port, 5)
    tail = port[port <= cutoff]
    var95 = float(-cutoff)
    cvar95 = float(-np.mean(tail)) if len(tail) else var95
    return {
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "var95": var95,
        "cvar95": cvar95,
    }


def rebalance(
    target_weights: dict[str, float],
    quantities: dict[str, float],
    latest_prices: dict[str, float],
) -> tuple[list[dict], float]:
    """Trades needed to move current holdings onto the target weights.

    Keeps the total value fixed (a pure rebalance, no fresh money). Returns one
    row per asset with current value, target value and the delta in both quote
    currency and units, plus the total portfolio value.
    """
    tickers = sorted(set(target_weights) | set(quantities) | set(latest_prices))
    current_value = {
        t: (quantities.get(t) or 0.0) * latest_prices.get(t, 0.0) for t in tickers
    }
    total = sum(current_value.values())

    rows: list[dict] = []
    for t in tickers:
        price = latest_prices.get(t, 0.0)
        cur_units = quantities.get(t) or 0.0
        cur_val = current_value[t]
        target_value = target_weights.get(t, 0.0) * total
        delta_value = target_value - cur_val
        delta_units = delta_value / price if price > 0 else 0.0
        rows.append(
            {
                "ticker": t,
                "price": price,
                "current_units": cur_units,
                "current_value": cur_val,
                "current_weight": cur_val / total if total > 0 else 0.0,
                "target_weight": target_weights.get(t, 0.0),
                "target_value": target_value,
                "delta_value": delta_value,
                "delta_units": delta_units,
            }
        )
    return rows, total


def allocate_capital(
    target_weights: dict[str, float],
    latest_prices: dict[str, float],
    capital: float,
    asset_types: dict[str, str] | None = None,
) -> tuple[list[dict], float]:
    """Turn target weights + a cash budget into units to buy.

    Crypto is bought fractionally, stocks in whole shares (floored). Returns one
    row per asset (target value, units, spent) and the leftover cash.
    """
    asset_types = asset_types or {}
    rows: list[dict] = []
    spent = 0.0
    for ticker, weight in sorted(target_weights.items()):
        price = latest_prices.get(ticker, 0.0)
        target_value = weight * capital
        if price <= 0:
            units = 0.0
        elif asset_types.get(ticker) == "stock":
            units = float(int(target_value // price))  # whole shares
        else:
            units = target_value / price  # fractional crypto
        value = units * price
        spent += value
        rows.append(
            {
                "ticker": ticker,
                "price": price,
                "target_weight": weight,
                "units": units,
                "value": value,
            }
        )
    return rows, capital - spent


def describe(result: OptimizeResult) -> str:
    """Format an OptimizeResult as human-readable text for the terminal."""
    lines: list[str] = []

    def block(portfolio: PortfolioResult) -> None:
        lines.append(f"{portfolio.name} Portfolio:")
        for ticker, weight in portfolio.weights.items():
            if weight > 0:
                lines.append(f"  {ticker:<10} {weight * 100:6.1f}%")
        lines.append(f"  Expected Return: {portfolio.expected_return * 100:6.1f}% p.a.")
        lines.append(f"  Volatility:      {portfolio.volatility * 100:6.1f}%")
        lines.append(f"  Sharpe Ratio:    {portfolio.sharpe:6.2f}")

    if result.max_sharpe is not None:
        block(result.max_sharpe)
        lines.append("")
    else:
        lines.append("Max Sharpe Portfolio: not available")
        lines.append(f"  reason: {result.max_sharpe_error}")
        lines.append("")

    block(result.min_volatility)
    lines.append("")

    if result.hrp is not None:
        block(result.hrp)
    else:
        lines.append("HRP Portfolio: not available")
        lines.append(f"  reason: {result.hrp_error}")
    return "\n".join(lines)
