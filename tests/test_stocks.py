"""Unit tests for the stocks client. No network: yfinance.download is mocked."""

import pandas as pd
import pytest

import fetch.stocks as stocks


def test_fetch_daily_prices_parses_close(monkeypatch):
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"Close": [185.0, 186.5]}, index=idx)
    monkeypatch.setattr(stocks.yf, "download", lambda *a, **k: df)

    rows = stocks.fetch_daily_prices("AAPL", lookback_days=5)

    assert rows == [("2024-01-02", 185.0), ("2024-01-03", 186.5)]


def test_fetch_daily_prices_raises_on_empty(monkeypatch):
    monkeypatch.setattr(stocks.yf, "download", lambda *a, **k: pd.DataFrame())

    with pytest.raises(stocks.StockFetchError):
        stocks.fetch_daily_prices("AAPL", lookback_days=5)


class _FakeTicker:
    def __init__(self, info):
        self.info = info


def test_fetch_market_metrics_maps_info(monkeypatch):
    info = {
        "longName": "Apple Inc.",
        "regularMarketPrice": 185.0,
        "marketCap": 2_900_000_000_000,
        "regularMarketVolume": 55_000_000,
    }
    monkeypatch.setattr(stocks.yf, "Ticker", lambda t: _FakeTicker(info))

    m = stocks.fetch_market_metrics("AAPL")

    assert m["name"] == "Apple Inc."
    assert m["price"] == 185.0
    assert m["market_cap"] == 2_900_000_000_000
    assert m["volume_24h"] == 55_000_000
    assert m["fdv"] is None
    assert m["change_24h"] is None
