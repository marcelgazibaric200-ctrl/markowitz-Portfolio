"""Hermetic smoke test: the Streamlit app renders without raising.

Points config.DB_PATH at a temp database seeded with synthetic prices, so the
full analysis path (optimization + all figures + tabs) executes offline. Guards
against import/render regressions in app.py.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
import db

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")


def _seed(db_path: str) -> None:
    conn = db.connect(db_path)
    pf = db.create_portfolio(conn, "T")
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    for ticker, asset_type in [("BTC-USD", "crypto"), ("ETH-USD", "crypto"), ("AAPL", "stock")]:
        db.upsert_asset(conn, ticker, ticker, asset_type)
        db.add_asset_to_portfolio(conn, pf["id"], ticker)
        series = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.02, 300)))
        db.save_prices(conn, ticker, list(zip(dates.strftime("%Y-%m-%d"), series)))
        db.set_asset_quantity(conn, pf["id"], ticker, 1.0)
    conn.close()


def test_app_renders_without_exception(tmp_path, monkeypatch):
    db_path = str(tmp_path / "app.db")
    monkeypatch.setattr(config, "DB_PATH", db_path)
    _seed(db_path)

    st.cache_resource.clear()
    st.cache_data.clear()

    at = AppTest.from_file(APP_PATH, default_timeout=120).run()

    assert not at.exception
    assert len(at.tabs) == 5
