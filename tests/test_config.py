"""Unit tests for config helpers, mainly the annualization factor."""

import config


def test_trading_days_crypto_only():
    # The default universe is crypto-only.
    assert config.trading_days_per_year() == 365


def test_trading_days_mixed(monkeypatch):
    monkeypatch.setattr(config, "ASSETS", ["BTC-USD", "AAPL"])
    assert config.trading_days_per_year() == 252


def test_trading_days_stock_only(monkeypatch):
    monkeypatch.setattr(config, "ASSETS", ["AAPL", "NVDA"])
    assert config.trading_days_per_year() == 252
