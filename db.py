"""SQLite storage for prices, assets and portfolios.

Tables:
  assets            metadata per instrument (ticker, type, CoinGecko id)
  prices            daily closes per ticker
  portfolios        named portfolios
  portfolio_assets  which assets belong to a portfolio (with optional weight)

Plain stdlib sqlite3, no ORM. See the project note for the schema rationale.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone

import pandas as pd

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    ticker        TEXT PRIMARY KEY,
    name          TEXT,
    asset_type    TEXT,
    coingecko_id  TEXT,
    active        INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS prices (
    id      INTEGER PRIMARY KEY,
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,
    close   REAL NOT NULL,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS portfolios (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_assets (
    portfolio_id  INTEGER NOT NULL,
    ticker        TEXT NOT NULL,
    weight        REAL,
    PRIMARY KEY (portfolio_id, ticker),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);
"""

ASSET_COLS = "ticker, name, asset_type, coingecko_id, active"


def connect(
    db_path: str | None = None,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Open a connection, creating the folder, schema and migrations if needed.

    Reads config.DB_PATH at call time when db_path is None (so tests can point
    it elsewhere). Pass check_same_thread=False for Streamlit, which may touch
    the connection from different threads across reruns.
    """
    if db_path is None:
        db_path = config.DB_PATH
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first schema version."""
    asset_cols = {row[1] for row in conn.execute("PRAGMA table_info(assets)")}
    if "coingecko_id" not in asset_cols:
        conn.execute("ALTER TABLE assets ADD COLUMN coingecko_id TEXT")
        conn.commit()

    # quantity = physical units held (coins/shares) per portfolio position.
    holding_cols = {row[1] for row in conn.execute("PRAGMA table_info(portfolio_assets)")}
    if "quantity" not in holding_cols:
        conn.execute("ALTER TABLE portfolio_assets ADD COLUMN quantity REAL")
        conn.commit()


# --- Assets -----------------------------------------------------------------

def _asset_row_to_dict(row: tuple) -> dict:
    return {
        "ticker": row[0],
        "name": row[1],
        "asset_type": row[2],
        "coingecko_id": row[3],
        "active": bool(row[4]),
    }


def upsert_asset(
    conn: sqlite3.Connection,
    ticker: str,
    name: str,
    asset_type: str,
    coingecko_id: str | None = None,
    active: bool = True,
) -> None:
    """Insert or update a row in the assets table.

    A NULL coingecko_id in the update keeps any existing id (COALESCE), so
    re-fetching a stock never wipes a stored crypto id.
    """
    conn.execute(
        "INSERT INTO assets (ticker, name, asset_type, coingecko_id, active)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(ticker) DO UPDATE SET"
        " name = excluded.name,"
        " asset_type = excluded.asset_type,"
        " coingecko_id = COALESCE(excluded.coingecko_id, assets.coingecko_id),"
        " active = excluded.active",
        (ticker, name, asset_type, coingecko_id, int(active)),
    )
    conn.commit()


def get_asset(conn: sqlite3.Connection, ticker: str) -> dict | None:
    row = conn.execute(
        f"SELECT {ASSET_COLS} FROM assets WHERE ticker = ?", (ticker,)
    ).fetchone()
    return _asset_row_to_dict(row) if row else None


def list_assets(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(f"SELECT {ASSET_COLS} FROM assets ORDER BY ticker").fetchall()
    return [_asset_row_to_dict(r) for r in rows]


# --- Prices -----------------------------------------------------------------

def save_prices(
    conn: sqlite3.Connection,
    ticker: str,
    rows: Iterable[tuple[str, float]],
) -> int:
    """Store (date, close) rows for a ticker. Existing dates are updated.

    Returns the number of rows written.
    """
    data = [(ticker, date, float(close)) for date, close in rows]
    conn.executemany(
        "INSERT INTO prices (ticker, date, close) VALUES (?, ?, ?)"
        " ON CONFLICT(ticker, date) DO UPDATE SET close = excluded.close",
        data,
    )
    conn.commit()
    return len(data)


def load_prices(
    conn: sqlite3.Connection,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Load prices into a wide DataFrame.

    index = date (datetime), columns = ticker, values = close.
    Pass `tickers` to select and order a subset of columns.
    """
    cur = conn.execute("SELECT ticker, date, close FROM prices ORDER BY date")
    records = cur.fetchall()
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records, columns=["ticker", "date", "close"])
    wide = df.pivot(index="date", columns="ticker", values="close")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()

    if tickers is not None:
        cols = [t for t in tickers if t in wide.columns]
        wide = wide[cols]
    return wide


def latest_prices(
    conn: sqlite3.Connection,
    tickers: list[str] | None = None,
) -> dict[str, float]:
    """Return the most recent stored close per ticker as {ticker: price}."""
    rows = conn.execute(
        "SELECT p.ticker, p.close FROM prices p"
        " JOIN (SELECT ticker, MAX(date) AS d FROM prices GROUP BY ticker) last"
        " ON p.ticker = last.ticker AND p.date = last.d"
    ).fetchall()
    prices = {ticker: float(close) for ticker, close in rows}
    if tickers is not None:
        prices = {t: prices[t] for t in tickers if t in prices}
    return prices


# --- Portfolios -------------------------------------------------------------

def create_portfolio(conn: sqlite3.Connection, name: str) -> dict:
    """Create a portfolio (idempotent by name) and return it."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO portfolios (name, created_at) VALUES (?, ?)"
        " ON CONFLICT(name) DO NOTHING",
        (name, now),
    )
    conn.commit()
    return get_portfolio(conn, name=name)


def get_portfolio(
    conn: sqlite3.Connection,
    portfolio_id: int | None = None,
    name: str | None = None,
) -> dict | None:
    if portfolio_id is not None:
        row = conn.execute(
            "SELECT id, name, created_at FROM portfolios WHERE id = ?", (portfolio_id,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, name, created_at FROM portfolios WHERE name = ?", (name,)
        ).fetchone()
    return {"id": row[0], "name": row[1], "created_at": row[2]} if row else None


def list_portfolios(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, name, created_at FROM portfolios ORDER BY name"
    ).fetchall()
    return [{"id": r[0], "name": r[1], "created_at": r[2]} for r in rows]


def delete_portfolio(conn: sqlite3.Connection, portfolio_id: int) -> None:
    conn.execute("DELETE FROM portfolios WHERE id = ?", (portfolio_id,))
    conn.commit()


# --- Portfolio composition --------------------------------------------------

def add_asset_to_portfolio(
    conn: sqlite3.Connection,
    portfolio_id: int,
    ticker: str,
    weight: float | None = None,
) -> None:
    conn.execute(
        "INSERT INTO portfolio_assets (portfolio_id, ticker, weight) VALUES (?, ?, ?)"
        " ON CONFLICT(portfolio_id, ticker) DO UPDATE SET weight = excluded.weight",
        (portfolio_id, ticker, weight),
    )
    conn.commit()


def set_asset_quantity(
    conn: sqlite3.Connection,
    portfolio_id: int,
    ticker: str,
    quantity: float | None,
) -> None:
    """Store the held quantity (coins/shares) for a portfolio position."""
    conn.execute(
        "UPDATE portfolio_assets SET quantity = ?"
        " WHERE portfolio_id = ? AND ticker = ?",
        (quantity, portfolio_id, ticker),
    )
    conn.commit()


def remove_asset_from_portfolio(
    conn: sqlite3.Connection, portfolio_id: int, ticker: str
) -> None:
    conn.execute(
        "DELETE FROM portfolio_assets WHERE portfolio_id = ? AND ticker = ?",
        (portfolio_id, ticker),
    )
    conn.commit()


def list_portfolio_assets(conn: sqlite3.Connection, portfolio_id: int) -> list[dict]:
    """Return the assets of a portfolio as dicts, including their weight."""
    rows = conn.execute(
        "SELECT a.ticker, a.name, a.asset_type, a.coingecko_id, a.active,"
        " pa.weight, pa.quantity"
        " FROM portfolio_assets pa"
        " JOIN assets a ON a.ticker = pa.ticker"
        " WHERE pa.portfolio_id = ?"
        " ORDER BY a.ticker",
        (portfolio_id,),
    ).fetchall()
    result = []
    for r in rows:
        asset = _asset_row_to_dict(r[:5])
        asset["weight"] = r[5]
        asset["quantity"] = r[6]
        result.append(asset)
    return result
