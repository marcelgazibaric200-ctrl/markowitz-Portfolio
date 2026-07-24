"""Advanced quant metrics: Sharpe significance, forward simulation, regimes, PCA.

Pure numerical functions kept separate from the optimizer and the Streamlit layer
so they stay unit-testable and offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from sklearn.mixture import GaussianMixture

import optimize

_EULER_MASCHERONI = 0.5772156649015329


# --- Sharpe ratio significance (Lopez de Prado) ------------------------------

def probabilistic_sharpe_ratio(
    returns: np.ndarray | pd.Series,
    sr_benchmark: float = 0.0,
) -> float:
    """Probability that the true (per-period) Sharpe exceeds sr_benchmark.

    Accounts for track length and the non-normality (skew, kurtosis) of the
    returns. Returns a probability in [0, 1].
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)  # per-period Sharpe
    g3 = float(skew(r, bias=False))
    g4 = float(kurtosis(r, fisher=False, bias=False))  # Pearson (3 for normal)
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    denom = max(denom, 1e-10)
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    trial_sharpes: np.ndarray | list[float],
    returns_best: np.ndarray | pd.Series,
) -> float:
    """Probabilistic Sharpe against a selection-inflated benchmark.

    With N strategies tried, the best Sharpe is upward biased. The benchmark
    SR* is the expected maximum Sharpe under N independent trials given the
    spread of the trial Sharpes; DSR is then the PSR of the best strategy
    against SR*. Answers "is the winning backtest real or luck?".
    """
    sharpes = np.asarray(trial_sharpes, dtype=float)
    sharpes = sharpes[np.isfinite(sharpes)]
    n_trials = len(sharpes)
    if n_trials < 2:
        return probabilistic_sharpe_ratio(returns_best, sr_benchmark=0.0)
    var = float(np.var(sharpes, ddof=1))
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr_star = np.sqrt(var) * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)
    return probabilistic_sharpe_ratio(returns_best, sr_benchmark=float(sr_star))


# --- Forward Monte-Carlo simulation ------------------------------------------

def forward_simulation(
    prices: pd.DataFrame,
    weights: dict[str, float],
    horizon_days: int = 252,
    n_paths: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap future portfolio paths from historical daily returns.

    Samples horizon_days daily portfolio returns with replacement per path
    (fat-tail honest, no distribution assumption) and compounds them. Returns
    percentile bands over time plus terminal risk stats (all relative to a
    start value of 1.0).
    """
    aligned = optimize.align_prices(prices)
    tickers = list(aligned.columns)
    w = pd.Series(weights, dtype=float).reindex(tickers).fillna(0.0).values
    port = aligned.pct_change().dropna().values @ w
    if len(port) < 2:
        raise ValueError("Not enough return history to simulate.")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(port), size=(n_paths, horizon_days))
    paths = np.cumprod(1.0 + port[idx], axis=1)  # (n_paths, horizon)

    pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    bands = pd.DataFrame(
        {"p5": pct[0], "p25": pct[1], "p50": pct[2], "p75": pct[3], "p95": pct[4]},
        index=np.arange(1, horizon_days + 1),
    )
    start = pd.DataFrame({c: [1.0] for c in bands.columns}, index=[0])
    bands = pd.concat([start, bands])

    terminal = paths[:, -1]
    var5 = float(np.percentile(terminal, 5))
    tail = terminal[terminal <= var5]
    return {
        "bands": bands,
        "p_loss": float(np.mean(terminal < 1.0)),
        "median_terminal": float(np.median(terminal)),
        "var5_terminal": var5,
        "cvar5_terminal": float(np.mean(tail)) if len(tail) else var5,
    }


# --- Regime analysis ---------------------------------------------------------

def _market_proxy_returns(prices: pd.DataFrame) -> pd.Series:
    """Daily returns of BTC (if present) else the equal-weight portfolio."""
    returns = optimize.align_prices(prices).pct_change().dropna()
    if "BTC-USD" in returns.columns:
        return returns["BTC-USD"]
    return returns.mean(axis=1)


def detect_regimes(
    prices: pd.DataFrame,
    n_regimes: int = 2,
    window: int = 20,
    seed: int = 42,
) -> pd.Series:
    """Label each day by volatility regime via a Gaussian mixture.

    The feature is the rolling realized volatility of a market proxy (BTC or the
    equal-weight portfolio). For two regimes the higher-vol cluster is "Stress",
    the lower "Calm". Returns a Series of labels indexed by date.
    """
    proxy = _market_proxy_returns(prices)
    vol = proxy.rolling(window).std().dropna()
    if len(vol) < n_regimes + 1:
        raise ValueError("Not enough history for regime detection.")

    gmm = GaussianMixture(n_components=n_regimes, random_state=seed)
    clusters = gmm.fit_predict(vol.values.reshape(-1, 1))
    order = np.argsort(gmm.means_.ravel())  # ascending volatility
    if n_regimes == 2:
        names = {order[0]: "Calm", order[1]: "Stress"}
    else:
        names = {c: f"Regime {rank + 1}" for rank, c in enumerate(order)}
    return pd.Series([names[c] for c in clusters], index=vol.index, name="regime")


def regime_correlations(prices: pd.DataFrame, labels: pd.Series) -> dict[str, pd.DataFrame]:
    """Correlation matrix of asset returns within each regime."""
    returns = optimize.align_prices(prices).pct_change().dropna()
    out: dict[str, pd.DataFrame] = {}
    for label in ["Calm", "Stress"] + sorted(set(labels) - {"Calm", "Stress"}):
        if label not in set(labels):
            continue
        dates = labels.index[labels == label]
        common = returns.index.intersection(dates)
        if len(common) >= 3:
            out[label] = returns.loc[common].corr()
    return out


# --- PCA / factor risk decomposition -----------------------------------------

def pca_risk(prices: pd.DataFrame) -> dict:
    """Principal-component decomposition of the return correlation matrix.

    Returns explained variance per component, cumulative variance, the effective
    number of independent bets (participation ratio of the eigenvalues), and the
    PC1 loadings (sign-normalized so the majority is positive).
    """
    returns = optimize.align_prices(prices).pct_change().dropna()
    corr = returns.corr()
    tickers = list(corr.columns)
    eigvals, eigvecs = np.linalg.eigh(corr.values)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    explained = eigvals / eigvals.sum()
    pc1 = eigvecs[:, 0]
    if np.sum(pc1 < 0) > np.sum(pc1 > 0):
        pc1 = -pc1
    effective_bets = float(eigvals.sum() ** 2 / np.sum(eigvals ** 2))
    return {
        "explained": explained,
        "cumulative": np.cumsum(explained),
        "effective_bets": effective_bets,
        "labels": tickers,
        "pc1_loadings": {t: float(v) for t, v in zip(tickers, pc1)},
    }
