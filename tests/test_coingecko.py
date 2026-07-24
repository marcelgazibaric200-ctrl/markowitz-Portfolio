"""Unit tests for the CoinGecko client. No network: requests.get is mocked."""

import pytest

import fetch.coingecko as cg


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_fetch_daily_prices_parses_timestamps(monkeypatch):
    # Two daily points at 2024-01-01 and 2024-01-02 (UTC, ms timestamps).
    payload = {
        "prices": [
            [1704067200000, 42000.0],  # 2024-01-01 00:00 UTC
            [1704153600000, 43000.0],  # 2024-01-02 00:00 UTC
        ]
    }
    monkeypatch.setattr(cg.requests, "get", lambda *a, **k: _FakeResponse(200, payload))

    rows = cg.fetch_daily_prices("bitcoin", days=2)

    assert rows == [("2024-01-01", 42000.0), ("2024-01-02", 43000.0)]


def test_fetch_daily_prices_raises_on_empty(monkeypatch):
    monkeypatch.setattr(
        cg.requests, "get", lambda *a, **k: _FakeResponse(200, {"prices": []})
    )
    with pytest.raises(cg.CoinGeckoError):
        cg.fetch_daily_prices("bitcoin", days=2)


def test_fetch_daily_prices_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(
        cg.requests, "get", lambda *a, **k: _FakeResponse(404, {"error": "not found"})
    )
    with pytest.raises(cg.CoinGeckoError):
        cg.fetch_daily_prices("not-a-coin", days=2)


def test_search_coin_maps_results(monkeypatch):
    payload = {
        "coins": [
            {"id": "solana", "symbol": "sol", "name": "Solana",
             "thumb": "http://x/sol.png", "market_cap_rank": 5},
            {"id": "wrapped-solana", "symbol": "wsol", "name": "Wrapped SOL",
             "thumb": "http://x/wsol.png", "market_cap_rank": 500},
        ]
    }
    monkeypatch.setattr(cg.requests, "get", lambda *a, **k: _FakeResponse(200, payload))

    results = cg.search_coin("sol")

    assert results[0] == {
        "id": "solana", "symbol": "SOL", "name": "Solana",
        "thumb": "http://x/sol.png", "market_cap_rank": 5,
    }
    assert len(results) == 2


def test_fetch_market_metrics_extracts_usd(monkeypatch):
    payload = {
        "name": "Bitcoin",
        "symbol": "btc",
        "image": {"small": "http://x/btc.png"},
        "market_data": {
            "current_price": {"usd": 62195.0},
            "market_cap": {"usd": 1.248e12},
            "fully_diluted_valuation": {"usd": 1.30e12},
            "total_volume": {"usd": 7.8e6},
            "price_change_percentage_24h": 1.23,
        },
    }
    monkeypatch.setattr(cg.requests, "get", lambda *a, **k: _FakeResponse(200, payload))

    m = cg.fetch_market_metrics("bitcoin")

    assert m["price"] == 62195.0
    assert m["market_cap"] == 1.248e12
    assert m["fdv"] == 1.30e12
    assert m["volume_24h"] == 7.8e6
    assert m["change_24h"] == 1.23
    assert m["image"] == "http://x/btc.png"
    assert m["symbol"] == "BTC"
