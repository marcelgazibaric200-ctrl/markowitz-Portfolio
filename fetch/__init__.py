"""Fetch orchestration: pull prices and market metrics into the DB.

The individual clients (`coingecko`, `stocks`) stay pure and only return data.
This module wires them to the database based on each asset's type.
"""

from __future__ import annotations

import sqlite3

import config
import db

from . import coingecko, stocks


def fetch_all(conn: sqlite3.Connection) -> dict[str, int]:
    """Fetch and store prices for every asset in config.ASSETS (CLI path).

    Returns a dict of ticker -> number of price rows stored.
    """
    stored: dict[str, int] = {}
    for ticker in config.ASSETS:
        kind = config.asset_type(ticker)
        cg_id = config.coingecko_id(ticker)
        if kind == "crypto":
            rows = coingecko.fetch_daily_prices(
                cg_id, vs_currency=config.QUOTE_CURRENCY, days=config.LOOKBACK_DAYS
            )
        else:
            rows = stocks.fetch_daily_prices(ticker, lookback_days=config.LOOKBACK_DAYS)

        db.upsert_asset(conn, ticker, name=ticker, asset_type=kind, coingecko_id=cg_id)
        stored[ticker] = db.save_prices(conn, ticker, rows)
    return stored


def fetch_for_tickers(
    conn: sqlite3.Connection,
    tickers: list[str],
    lookback_days: int = config.LOOKBACK_DAYS,
) -> dict[str, int]:
    """Fetch and store prices for specific tickers, using DB asset metadata.

    Each ticker must already exist in the assets table (type and CoinGecko id
    are read from there). Returns a dict of ticker -> rows stored.
    """
    stored: dict[str, int] = {}
    for ticker in tickers:
        asset = db.get_asset(conn, ticker)
        if asset is None:
            continue
        if asset["asset_type"] == "crypto":
            rows = coingecko.fetch_daily_prices(
                asset["coingecko_id"],
                vs_currency=config.QUOTE_CURRENCY,
                days=lookback_days,
            )
        else:
            rows = stocks.fetch_daily_prices(ticker, lookback_days=lookback_days)
        stored[ticker] = db.save_prices(conn, ticker, rows)
    return stored


def market_metrics(asset: dict) -> dict:
    """Return current market metrics for an asset dict (from db.get_asset)."""
    if asset["asset_type"] == "crypto":
        metrics = coingecko.fetch_market_metrics(
            asset["coingecko_id"], vs_currency=config.QUOTE_CURRENCY
        )
    else:
        metrics = stocks.fetch_market_metrics(asset["ticker"])
    # Fall back to the stored name if the provider did not return one.
    metrics["name"] = metrics.get("name") or asset.get("name") or asset["ticker"]
    return metrics
