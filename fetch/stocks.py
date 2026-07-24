"""Stock price fetching via yfinance.

Returns the same (date, close) shape as the CoinGecko client so both plug into
the database identically. Uses auto-adjusted closes (splits and dividends).
"""

from __future__ import annotations

import yfinance as yf


class StockFetchError(RuntimeError):
    """Raised when yfinance returns no usable data for a ticker."""


def fetch_daily_prices(
    ticker: str,
    lookback_days: int = 365,
) -> list[tuple[str, float]]:
    """Return daily close prices for a stock as a sorted list of (date, close)."""
    df = yf.download(
        ticker,
        period=f"{lookback_days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise StockFetchError(f"No price data for ticker '{ticker}'")

    close = df["Close"]
    # For a single ticker yfinance may return a one-column DataFrame.
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]

    rows: list[tuple[str, float]] = []
    for ts, price in close.items():
        rows.append((ts.strftime("%Y-%m-%d"), float(price)))
    return rows


def fetch_market_metrics(ticker: str) -> dict:
    """Return current market metrics for a stock in the common dict schema.

    change_24h is left None on purpose: yfinance's percent field mixes fractions
    and percentages across tickers, so showing it risks wrong numbers.
    """
    info = yf.Ticker(ticker).info or {}
    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "symbol": ticker,
        "image": None,
        "price": info.get("regularMarketPrice") or info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "fdv": None,
        "volume_24h": info.get("regularMarketVolume") or info.get("volume"),
        "change_24h": None,
    }
