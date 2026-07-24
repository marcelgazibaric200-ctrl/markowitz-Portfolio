"""Walk-forward out-of-sample backtest for the portfolio strategies.

Rebalances on a fixed schedule using only a trailing window of history (no
lookahead), holds the weights until the next rebalance, and stitches the
out-of-sample daily returns into a growth curve. Compares the three optimizers
against two honest benchmarks: equal weight and buy & hold BTC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import optimize

STRATEGIES = ["Max Sharpe", "Min Vol", "HRP", "Equal-Weight", "Buy&Hold BTC"]


def _weights_for_window(
    window: pd.DataFrame,
    frequency: int,
    tickers: list[str],
) -> dict[str, dict[str, float]]:
    """Weights per strategy from a trailing price window (equal-weight fallback)."""
    equal = {t: 1.0 / len(tickers) for t in tickers}
    weights: dict[str, dict[str, float]] = {}
    try:
        mu, cov, _ = optimize.compute_inputs(window, frequency=frequency)
        weights["Min Vol"] = optimize.min_volatility(mu, cov).weights
        try:
            weights["Max Sharpe"] = optimize.max_sharpe(mu, cov).weights
        except Exception:  # noqa: BLE001 - infeasible tangency -> hold equal weight
            weights["Max Sharpe"] = equal
        try:
            weights["HRP"] = optimize.hierarchical_risk_parity(window, mu, cov).weights
        except Exception:  # noqa: BLE001
            weights["HRP"] = equal
    except Exception:  # noqa: BLE001 - degenerate window -> everything equal weight
        weights["Min Vol"] = equal
        weights["Max Sharpe"] = equal
        weights["HRP"] = equal

    weights["Equal-Weight"] = equal
    btc = "BTC-USD" if "BTC-USD" in tickers else tickers[0]
    weights["Buy&Hold BTC"] = {btc: 1.0}
    return weights


def walk_forward(
    prices: pd.DataFrame,
    lookback_days: int = 180,
    rebalance_days: int = 30,
    frequency: int | None = None,
    risk_free_rate: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the walk-forward backtest.

    Returns (curves, stats): `curves` is the growth of $1 per strategy over the
    out-of-sample period (all starting at 1.0), `stats` holds annualized return,
    volatility, Sharpe, max drawdown and the final value per strategy.
    """
    rf = config.RISK_FREE_RATE if risk_free_rate is None else risk_free_rate
    aligned = optimize.align_prices(prices)
    tickers = list(aligned.columns)
    if frequency is None:
        frequency = 365 if len(tickers) else 252
    n = len(aligned)
    if n <= lookback_days + 1 or len(tickers) == 0:
        raise ValueError(
            "Not enough history for the backtest. Reduce the lookback or fetch "
            "more price history."
        )

    daily = aligned.pct_change()
    oos_index = aligned.index[lookback_days:n]
    returns_df = pd.DataFrame(0.0, index=oos_index, columns=STRATEGIES)

    for p in range(lookback_days, n, rebalance_days):
        window = aligned.iloc[p - lookback_days : p]
        weights = _weights_for_window(window, frequency, tickers)
        block = daily.iloc[p : min(p + rebalance_days, n)]
        for strategy in STRATEGIES:
            w = pd.Series(weights[strategy]).reindex(tickers).fillna(0.0)
            returns_df.loc[block.index, strategy] = block.mul(w, axis=1).sum(axis=1).values

    # Anchor every curve at 1.0 one step before the first out-of-sample day.
    anchor = aligned.index[lookback_days - 1]
    returns_df.loc[anchor] = 0.0
    returns_df = returns_df.sort_index()
    curves = (1.0 + returns_df).cumprod()

    stats = _stats(returns_df.drop(index=anchor), curves, frequency, rf)
    return curves, stats


def _stats(
    returns_df: pd.DataFrame,
    curves: pd.DataFrame,
    frequency: int,
    rf: float,
) -> pd.DataFrame:
    """Annualized performance per strategy from the out-of-sample daily returns."""
    rows = []
    for strategy in returns_df.columns:
        r = returns_df[strategy].values
        ann_return = float(np.mean(r) * frequency) if len(r) else 0.0
        ann_vol = float(np.std(r, ddof=1) * np.sqrt(frequency)) if len(r) > 1 else 0.0
        sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else float("nan")
        cumulative = np.cumprod(1 + r)
        peak = np.maximum.accumulate(cumulative)
        max_drawdown = float((cumulative / peak - 1).min()) if len(r) else 0.0
        rows.append(
            {
                "Strategie": strategy,
                "Rendite p.a.": ann_return,
                "Vola p.a.": ann_vol,
                "Sharpe": sharpe,
                "Max Drawdown": max_drawdown,
                "Endwert": float(curves[strategy].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)
