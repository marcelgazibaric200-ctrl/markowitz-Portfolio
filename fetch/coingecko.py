"""Minimal CoinGecko client for daily historical prices.

Built from scratch on top of `requests`. Implements only what the optimizer
needs: daily close prices for one coin over a lookback window.

Uses the free public API. For a `days` value above 90 CoinGecko returns daily
granularity automatically, so we do not send the `interval` parameter (it is a
paid-plan feature and errors on the free tier).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import requests

BASE_URL = "https://api.coingecko.com/api/v3"

# Politeness / retry settings for the rate-limited free tier.
_MAX_RETRIES = 4
_RETRY_BACKOFF_SECONDS = 5
_TIMEOUT_SECONDS = 30


class CoinGeckoError(RuntimeError):
    """Raised when CoinGecko cannot be reached or returns unusable data."""


def _get(url: str, params: dict) -> dict:
    """GET a JSON endpoint with retries on rate limiting."""
    last_error = "unknown error"
    for attempt in range(_MAX_RETRIES):
        resp = requests.get(url, params=params, timeout=_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
            last_error = "429 rate limited"
            continue
        raise CoinGeckoError(
            f"CoinGecko returned {resp.status_code}: {resp.text[:200]}"
        )
    raise CoinGeckoError(f"CoinGecko request failed after retries: {last_error}")


def fetch_daily_prices(
    coin_id: str,
    vs_currency: str = "usd",
    days: int = 365,
) -> list[tuple[str, float]]:
    """Return daily close prices for a coin as a sorted list of (date, close).

    date is an ISO string "YYYY-MM-DD" (UTC), close is a float in vs_currency.
    """
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    data = _get(url, params)

    prices = data.get("prices")
    if not prices:
        raise CoinGeckoError(f"No price data for coin '{coin_id}'")

    # market_chart returns [[timestamp_ms, price], ...]. Daily data has one
    # point per day; we keep the last point seen per date defensively.
    by_date: dict[str, float] = {}
    for ts_ms, price in prices:
        date = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        by_date[date] = float(price)

    return sorted(by_date.items())


def search_coin(query: str, limit: int = 8) -> list[dict]:
    """Search CoinGecko for coins matching a name or symbol.

    Returns a list of {id, symbol, name, thumb, market_cap_rank} for the UI.
    """
    data = _get(f"{BASE_URL}/search", {"query": query})
    coins = data.get("coins") or []
    return [
        {
            "id": c.get("id"),
            "symbol": (c.get("symbol") or "").upper(),
            "name": c.get("name"),
            "thumb": c.get("thumb"),
            "market_cap_rank": c.get("market_cap_rank"),
        }
        for c in coins[:limit]
    ]


def fetch_market_metrics(coin_id: str, vs_currency: str = "usd") -> dict:
    """Return current market metrics for a coin in a common dict schema."""
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }
    data = _get(f"{BASE_URL}/coins/{coin_id}", params)
    md = data.get("market_data") or {}

    def in_currency(section: str) -> float | None:
        value = md.get(section)
        return value.get(vs_currency) if isinstance(value, dict) else None

    return {
        "name": data.get("name"),
        "symbol": (data.get("symbol") or "").upper(),
        "image": (data.get("image") or {}).get("small"),
        "price": in_currency("current_price"),
        "market_cap": in_currency("market_cap"),
        "fdv": in_currency("fully_diluted_valuation"),
        "volume_24h": in_currency("total_volume"),
        "change_24h": md.get("price_change_percentage_24h"),
    }
