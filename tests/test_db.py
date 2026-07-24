"""Unit tests for the SQLite storage layer, using a temp database."""

import db


def test_save_and_load_roundtrip(tmp_path):
    conn = db.connect(str(tmp_path / "test.db"))
    db.upsert_asset(conn, "BTC-USD", "Bitcoin", "crypto")
    written = db.save_prices(
        conn, "BTC-USD", [("2024-01-01", 42000.0), ("2024-01-02", 43000.0)]
    )
    assert written == 2

    wide = db.load_prices(conn)
    assert list(wide.columns) == ["BTC-USD"]
    assert len(wide) == 2
    assert wide["BTC-USD"].iloc[0] == 42000.0
    conn.close()


def test_save_prices_upserts_same_date(tmp_path):
    conn = db.connect(str(tmp_path / "test.db"))
    db.save_prices(conn, "BTC-USD", [("2024-01-01", 42000.0)])
    db.save_prices(conn, "BTC-USD", [("2024-01-01", 99999.0)])  # same date -> update

    wide = db.load_prices(conn)
    assert len(wide) == 1
    assert wide["BTC-USD"].iloc[0] == 99999.0
    conn.close()


def test_load_prices_empty_returns_empty_frame(tmp_path):
    conn = db.connect(str(tmp_path / "test.db"))
    assert db.load_prices(conn).empty
    conn.close()


def test_upsert_asset_stores_coingecko_id(tmp_path):
    conn = db.connect(str(tmp_path / "test.db"))
    db.upsert_asset(conn, "BTC-USD", "Bitcoin", "crypto", coingecko_id="bitcoin")

    asset = db.get_asset(conn, "BTC-USD")
    assert asset["coingecko_id"] == "bitcoin"
    assert asset["asset_type"] == "crypto"
    conn.close()


def test_upsert_asset_keeps_id_when_update_passes_none(tmp_path):
    conn = db.connect(str(tmp_path / "test.db"))
    db.upsert_asset(conn, "BTC-USD", "Bitcoin", "crypto", coingecko_id="bitcoin")
    db.upsert_asset(conn, "BTC-USD", "Bitcoin", "crypto")  # no id passed

    assert db.get_asset(conn, "BTC-USD")["coingecko_id"] == "bitcoin"
    conn.close()


def test_portfolio_crud_and_composition(tmp_path):
    conn = db.connect(str(tmp_path / "test.db"))
    pf = db.create_portfolio(conn, "Crypto Core")
    assert pf["id"] > 0
    assert db.create_portfolio(conn, "Crypto Core")["id"] == pf["id"]  # idempotent

    db.upsert_asset(conn, "BTC-USD", "Bitcoin", "crypto", coingecko_id="bitcoin")
    db.upsert_asset(conn, "ETH-USD", "Ethereum", "crypto", coingecko_id="ethereum")
    db.add_asset_to_portfolio(conn, pf["id"], "BTC-USD", weight=0.6)
    db.add_asset_to_portfolio(conn, pf["id"], "ETH-USD", weight=0.4)

    assets = db.list_portfolio_assets(conn, pf["id"])
    assert [a["ticker"] for a in assets] == ["BTC-USD", "ETH-USD"]
    assert assets[0]["weight"] == 0.6

    db.remove_asset_from_portfolio(conn, pf["id"], "ETH-USD")
    assert [a["ticker"] for a in db.list_portfolio_assets(conn, pf["id"])] == ["BTC-USD"]
    conn.close()


def test_set_and_read_asset_quantity(tmp_path):
    conn = db.connect(str(tmp_path / "test.db"))
    pf = db.create_portfolio(conn, "Holdings")
    db.upsert_asset(conn, "BTC-USD", "Bitcoin", "crypto")
    db.add_asset_to_portfolio(conn, pf["id"], "BTC-USD")

    # Default quantity is NULL until set.
    assert db.list_portfolio_assets(conn, pf["id"])[0]["quantity"] is None

    db.set_asset_quantity(conn, pf["id"], "BTC-USD", 0.75)
    assert db.list_portfolio_assets(conn, pf["id"])[0]["quantity"] == 0.75
    conn.close()


def test_latest_prices_returns_most_recent_close(tmp_path):
    conn = db.connect(str(tmp_path / "test.db"))
    db.upsert_asset(conn, "BTC-USD", "Bitcoin", "crypto")
    db.upsert_asset(conn, "ETH-USD", "Ethereum", "crypto")
    db.save_prices(conn, "BTC-USD", [("2024-01-01", 42000.0), ("2024-01-03", 45000.0)])
    db.save_prices(conn, "ETH-USD", [("2024-01-02", 2500.0)])

    prices = db.latest_prices(conn, ["BTC-USD", "ETH-USD"])
    assert prices == {"BTC-USD": 45000.0, "ETH-USD": 2500.0}

    assert db.latest_prices(conn, ["BTC-USD"]) == {"BTC-USD": 45000.0}
    conn.close()


def test_delete_portfolio_cascades(tmp_path):
    conn = db.connect(str(tmp_path / "test.db"))
    pf = db.create_portfolio(conn, "Temp")
    db.upsert_asset(conn, "BTC-USD", "Bitcoin", "crypto", coingecko_id="bitcoin")
    db.add_asset_to_portfolio(conn, pf["id"], "BTC-USD")

    db.delete_portfolio(conn, pf["id"])
    assert db.get_portfolio(conn, portfolio_id=pf["id"]) is None
    assert db.list_portfolio_assets(conn, pf["id"]) == []
    conn.close()
