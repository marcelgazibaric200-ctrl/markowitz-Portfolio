# Markowitz Portfolio Optimizer

Local tool that optimizes a crypto and stock portfolio with Markowitz mean-variance
analysis. It fetches historical prices, computes the efficient frontier with
Ledoit-Wolf shrinkage, and reports the Max Sharpe and Min Volatility portfolios.

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

## Usage

```bash
python main.py fetch             # Fetch prices into data/prices.db
python main.py optimize          # Print Max Sharpe and Min Volatility weights
python main.py plot              # Draw the efficient frontier in the browser
python main.py all               # fetch -> optimize -> plot
python main.py all --no-open     # ... without opening a browser
```

## Frontend (Streamlit)

```bash
venv\Scripts\streamlit run app.py
```

A dark dashboard where you can:

- Create portfolios and switch between them.
- Add crypto by searching CoinGecko (type a name or symbol) and stocks by ticker.
  Assets are stored in the database and their prices are loaded immediately.
- Enter your real holdings per asset (coins, USD or percent) in an editable table.
  Your current portfolio is then drawn on the frontier and drives the rebalancing.
- Cap the weight of any single asset (e.g. BTC) and the combined crypto weight,
  switch the covariance estimator (Ledoit-Wolf / exponentially weighted) and the
  return estimator (mean historical / EMA / CAPM), and optionally denoise the
  covariance with Marchenko-Pastur (RMT). All settings feed the optimizer live.
- Refresh all prices for the selected portfolio with one button.
- Explore a quantitative dashboard organised in tabs:
  - **Frontier & Gewichte**: efficient frontier with a Monte Carlo cloud coloured
    by Sharpe, the **Capital Market Line**, five allocations (Max Sharpe, Min
    Volatility, **HRP**, **Min CVaR**, **Min Semivariance**) and your current
    holdings marked on it, a weights bar and a comparison table (CSV export).
  - **Korrelation**: correlation heatmap, a **dendrogram** (the same clustering
    HRP uses), rolling correlation to BTC over time, and a correlation network.
  - **Risiko**: weight vs. risk contribution per asset, and downside metrics
    (Sortino, max drawdown, 95% VaR/CVaR).
  - **Rebalancing**: for a chosen target portfolio, how much to buy/sell per asset
    in USD and coins to reach the optimal weights (CSV export), plus a
    discrete-allocation planner for investing fresh capital.
  - **Backtest**: walk-forward out-of-sample test of the optimizers against
    equal-weight and buy & hold BTC, with growth curves and a stats table.

The UI is built for a keyboard without arrow keys: typeable number fields,
buttons, tabs and an editable table — no sliders.

Portfolios, assets and holdings live in the database (`portfolios`,
`portfolio_assets`, `assets` tables), so the frontend does not depend on the
hardcoded `config.py` universe. On first run a "Default" portfolio is seeded from
`config.CRYPTO_ASSETS`.

The quickest way to launch on Windows is to double-click `start_app.bat`, which
starts the local server and opens the app in your browser.

## Configuration

Everything is in `config.py`:

- `CRYPTO_ASSETS` / `STOCK_ASSETS` define the portfolio universe.
- `LOOKBACK_DAYS` is how much price history to use (default 365).
- `RISK_FREE_RATE` feeds the Sharpe ratio (default 0.04).
- `MAX_WEIGHT_PER_ASSET` / `MAX_CRYPTO_WEIGHT` are the default constraint caps
  (1.0 = off); the frontend exposes both as number inputs.
- `COV_METHOD` selects the covariance estimator (`ledoit_wolf` or `exp_cov`).
- `RETURN_METHOD` selects the expected-returns estimator (`mean_historical`,
  `ema` or `capm`); `DENOISE_COV` toggles RMT covariance denoising.
- `CURRENT_PORTFOLIO` optionally plots where you are today on the frontier.

Crypto prices come from a custom CoinGecko client (`fetch/coingecko.py`), stocks
from `yfinance` (`fetch/stocks.py`).

## Notes

- Returns are annualized with 365 trading days for crypto, 252 for stocks. Mixed
  portfolios are aligned to the common trading calendar (stock trading days) in
  `optimize.align_prices`, so they annualize with 252.
- Max Sharpe is only defined when at least one asset's expected return beats the
  risk-free rate (or the caps are too tight). When that fails the tool reports it
  and still shows Min Volatility.
- HRP (Hierarchical Risk Parity) is solved separately and is unconstrained by
  design; the per-asset and crypto caps apply to Max Sharpe / Min Volatility and
  the drawn frontier.
- Min CVaR and Min Semivariance use CVXPY's CLARABEL solver and are scored on the
  same mean/variance axis as the others for comparison; their native tail metric
  (annual CVaR / semi-deviation) is shown in the comparison table.
- The backtest is walk-forward (rolling re-optimization on a trailing window, held
  out of sample) — no lookahead. Buy & hold BTC and equal weight are the honest
  benchmarks.

## Tests

```bash
python -m pytest
```

Tests are offline: the CoinGecko and yfinance clients are mocked, the database
uses a temp file.
