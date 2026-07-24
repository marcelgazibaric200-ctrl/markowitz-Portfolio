"""Central configuration for the Markowitz portfolio optimizer.

Everything downstream (fetch, optimize, plot) reads from this module.
To change the portfolio universe, edit CRYPTO_ASSETS / STOCK_ASSETS below.
"""

from __future__ import annotations

# --- Portfolio universe -----------------------------------------------------

# Crypto assets: canonical ticker -> CoinGecko coin id.
# The ticker ("BTC-USD") is what we store in the database and show to the user.
# The CoinGecko id ("bitcoin") is what the CoinGecko API needs.
CRYPTO_ASSETS: dict[str, str] = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
}

# Stock assets: yfinance tickers. Add any ticker here and the whole pipeline
# (fetch, optimize, plot) picks it up. Mixed crypto/stock portfolios are
# aligned to the common trading calendar automatically.
STOCK_ASSETS: list[str] = [
    # "AAPL",
    # "NVDA",
]

# Combined universe used by the optimizer, in a stable order.
ASSETS: list[str] = list(CRYPTO_ASSETS.keys()) + STOCK_ASSETS

# --- Model settings ---------------------------------------------------------

# How much daily price history to pull for each asset.
LOOKBACK_DAYS: int = 365

# Risk-free rate used in the Sharpe ratio (~4% US T-bills).
RISK_FREE_RATE: float = 0.04

# Quote currency for crypto prices and the base currency of the portfolio.
QUOTE_CURRENCY: str = "usd"

# --- Constraints ------------------------------------------------------------

# Maximum weight a single asset may take in the optimized portfolios (1.0 = no
# cap). Lowering this caps e.g. BTC so the optimizer cannot pile everything in.
MAX_WEIGHT_PER_ASSET: float = 1.0

# Maximum combined weight of all crypto assets (sector cap). Keeps a mixed
# portfolio from jumping fully into crypto. 1.0 = no cap.
MAX_CRYPTO_WEIGHT: float = 1.0

# Covariance estimator: "ledoit_wolf" (shrinkage) or "exp_cov" (exponentially
# weighted, recent data counts more).
COV_METHOD: str = "ledoit_wolf"

# Expected-returns estimator: "mean_historical", "ema" (exponentially weighted)
# or "capm" (CAPM equilibrium returns). mu is the least stable Markowitz input,
# so this is worth experimenting with.
RETURN_METHOD: str = "mean_historical"

# Filter noise eigenvalues out of the covariance via Marchenko-Pastur (RMT)
# before optimizing. Off by default.
DENOISE_COV: bool = False

# --- Storage ----------------------------------------------------------------

# Location of the local SQLite database (relative to the project root).
DB_PATH: str = "data/prices.db"

# Your current holdings as ticker -> weight (should sum to 1). Leave empty to
# skip drawing "where you are now" onto the efficient frontier.
CURRENT_PORTFOLIO: dict[str, float] = {}


# --- Helpers ----------------------------------------------------------------

def asset_type(ticker: str) -> str:
    """Return "crypto" or "stock" for a given ticker."""
    return "crypto" if ticker in CRYPTO_ASSETS else "stock"


def coingecko_id(ticker: str) -> str | None:
    """Return the CoinGecko coin id for a crypto ticker, or None for stocks."""
    return CRYPTO_ASSETS.get(ticker)


def trading_days_per_year() -> int:
    """Annualization factor for the current universe.

    Crypto trades every day (365), stocks about 252. A mixed universe is aligned
    to the common trading calendar in optimize.align_prices, which keeps only the
    stock trading days, so 252 is the correct factor there as well.
    """
    kinds = {asset_type(t) for t in ASSETS}
    if kinds == {"crypto"}:
        return 365
    if kinds == {"stock"}:
        return 252
    return 252
