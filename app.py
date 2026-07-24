"""Streamlit frontend for the Markowitz portfolio optimizer.

Run with:  venv\\Scripts\\streamlit run app.py   (or double-click start_app.bat)

Lets you create portfolios, add crypto (CoinGecko search) and stock assets to
the database, enter your real holdings (coins / USD / percent), cap single-asset
and total crypto weight, and view a quantitative dashboard: efficient frontier
with a Monte Carlo cloud, Max Sharpe / Min Volatility / HRP portfolios, risk
contributions, downside metrics, four correlation views and a rebalancing plan.

The UI is built for a keyboard without arrow keys: everything is typeable number
fields, click buttons, tabs and an editable table (no sliders).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import analysis
import config
import db
import optimize
import plot
from fetch import coingecko
import fetch

st.set_page_config(page_title="Markowitz Portfolio", layout="wide")


@st.cache_resource
def get_conn():
    return db.connect(check_same_thread=False)


@st.cache_data(ttl=120, show_spinner=False)
def cached_search(query: str) -> list[dict]:
    return coingecko.search_coin(query)


@st.cache_data(show_spinner=False)
def compute_analysis(
    tickers: tuple[str, ...],
    asset_types_items: tuple[tuple[str, str], ...],
    current_weights_items: tuple[tuple[str, float], ...],
    rf: float,
    freq: int,
    max_weight: float,
    crypto_cap: float,
    cov_method: str,
    data_version: int,
):
    """Load prices, optimize with constraints, and build the dashboard figures.

    Cached by all inputs (including data_version, which bumps on any data or
    holdings change), so reruns are cheap.
    """
    conn = db.connect(check_same_thread=False)
    prices = db.load_prices(conn, tickers=list(tickers))
    conn.close()
    if prices.empty:
        return None

    config.RISK_FREE_RATE = rf
    asset_types = dict(asset_types_items)
    current_weights = dict(current_weights_items) or None
    result = optimize.run(
        prices,
        frequency=freq,
        asset_types=asset_types,
        max_weight=max_weight,
        crypto_cap=crypto_cap,
        cov_method=cov_method,
    )
    figures = {
        "frontier": plot.build_frontier_montecarlo(result, current_weights=current_weights),
        "weights": plot.build_weights_bar(result),
        "risk": plot.build_risk_contribution_bar(result),
        "heatmap": plot.build_correlation_heatmap(prices),
        "dendrogram": plot.build_dendrogram(prices),
        "rolling": plot.build_rolling_correlation(prices),
        "network": plot.build_correlation_network(prices),
        "prices": plot.build_normalized_prices(prices),
    }
    primary = result.max_sharpe or result.min_volatility
    downside = optimize.downside_metrics(prices, primary.weights, freq, rf)
    return result, figures, downside


def bump_data_version() -> None:
    st.session_state.data_version = st.session_state.get("data_version", 0) + 1


def ensure_seed(conn) -> None:
    """Seed a Default portfolio from config on first run so the UI is not empty."""
    if db.list_portfolios(conn):
        return
    pf = db.create_portfolio(conn, "Default")
    for ticker, coingecko_id in config.CRYPTO_ASSETS.items():
        db.upsert_asset(
            conn, ticker, name=ticker, asset_type="crypto", coingecko_id=coingecko_id
        )
        db.add_asset_to_portfolio(conn, pf["id"], ticker)


def portfolio_comparison(result) -> pd.DataFrame:
    """Return a tidy comparison table of the solved portfolios."""
    rows = []
    for portfolio in (result.max_sharpe, result.min_volatility, result.hrp):
        if portfolio is None:
            continue
        rows.append(
            {
                "Portfolio": portfolio.name,
                "Rendite p.a.": analysis.fmt_pct(portfolio.expected_return, signed=True),
                "Volatilitaet p.a.": analysis.fmt_pct(portfolio.volatility),
                "Sharpe": analysis._de(portfolio.sharpe, 2),
            }
        )
    return pd.DataFrame(rows)


# --- App --------------------------------------------------------------------

conn = get_conn()
ensure_seed(conn)
st.session_state.setdefault("data_version", 0)

# Sidebar: portfolio selection, settings, constraints, refresh.
with st.sidebar:
    st.header("Portfolios")
    portfolios = db.list_portfolios(conn)
    names = [p["name"] for p in portfolios]
    selected_name = st.selectbox("Portfolio", names, key="selected_portfolio")
    selected = next(p for p in portfolios if p["name"] == selected_name)

    with st.form("new_portfolio", clear_on_submit=True):
        new_name = st.text_input("Neues Portfolio")
        if st.form_submit_button("Erstellen") and new_name.strip():
            db.create_portfolio(conn, new_name.strip())
            st.session_state.selected_portfolio = new_name.strip()
            st.rerun()

    st.divider()
    st.subheader("Einstellungen")
    rf = st.number_input(
        "Risk-free Rate (%)", value=config.RISK_FREE_RATE * 100, step=0.5
    ) / 100
    lookback = int(
        st.number_input("History (Tage)", value=config.LOOKBACK_DAYS, step=30, min_value=30)
    )

    st.subheader("Constraints")
    max_weight = st.number_input(
        "Max Gewicht pro Asset (%)", value=100.0, min_value=1.0, max_value=100.0, step=5.0,
        help="Deckelt jedes einzelne Asset (z.B. BTC). 100 = keine Deckelung.",
    ) / 100
    crypto_cap = st.number_input(
        "Max Krypto-Anteil (%)", value=100.0, min_value=0.0, max_value=100.0, step=5.0,
        help="Deckelt die Summe aller Krypto-Assets. Wirkt nur mit mind. einer Aktie.",
    ) / 100
    cov_label = st.radio(
        "Kovarianz-Schaetzer",
        ["Ledoit-Wolf", "Exponentiell gewichtet"],
        help="Ledoit-Wolf = robuste Schrumpfung. EW = juengere Daten zaehlen mehr.",
    )
    cov_method = "exp_cov" if cov_label.startswith("Exp") else "ledoit_wolf"

    st.divider()
    if st.button("Aktualisieren", width="stretch", type="primary"):
        tickers = [a["ticker"] for a in db.list_portfolio_assets(conn, selected["id"])]
        if tickers:
            with st.spinner("Lade Preise ..."):
                fetch.fetch_for_tickers(conn, tickers, lookback_days=lookback)
            bump_data_version()
            st.success("Preise aktualisiert")
        else:
            st.info("Keine Assets im Portfolio")

    if len(portfolios) > 1 and st.button("Portfolio loeschen", width="stretch"):
        db.delete_portfolio(conn, selected["id"])
        del st.session_state["selected_portfolio"]
        st.rerun()

st.markdown(f"## {selected['name']}")

# --- Asset management -------------------------------------------------------
st.subheader("Assets")
col_add, col_list = st.columns(2)

with col_add:
    st.markdown("**Krypto hinzufuegen**")
    query = st.text_input("Suche (Name oder Symbol)", key="crypto_query")
    if query.strip():
        try:
            results = cached_search(query.strip())
        except Exception as exc:  # noqa: BLE001 - surface any API error in UI
            results = []
            st.error(f"Suche fehlgeschlagen: {exc}")
        for coin in results[:6]:
            label = f"{coin['name']} ({coin['symbol']})"
            if st.button(f"+ {label}", key=f"add_{coin['id']}"):
                ticker = f"{coin['symbol']}-USD"
                db.upsert_asset(
                    conn, ticker, name=coin["name"], asset_type="crypto",
                    coingecko_id=coin["id"],
                )
                db.add_asset_to_portfolio(conn, selected["id"], ticker)
                with st.spinner(f"Lade {ticker} ..."):
                    try:
                        fetch.fetch_for_tickers(conn, [ticker], lookback_days=lookback)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Laden fehlgeschlagen: {exc}")
                bump_data_version()
                st.rerun()

    st.markdown("**Aktie hinzufuegen**")
    with st.form("add_stock", clear_on_submit=True):
        stock = st.text_input("Ticker (z.B. AAPL)")
        if st.form_submit_button("Hinzufuegen") and stock.strip():
            ticker = stock.strip().upper()
            db.upsert_asset(conn, ticker, name=ticker, asset_type="stock")
            db.add_asset_to_portfolio(conn, selected["id"], ticker)
            with st.spinner(f"Lade {ticker} ..."):
                try:
                    fetch.fetch_for_tickers(conn, [ticker], lookback_days=lookback)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Laden fehlgeschlagen: {exc}")
            bump_data_version()
            st.rerun()

with col_list:
    st.markdown("**Im Portfolio**")
    portfolio_assets = db.list_portfolio_assets(conn, selected["id"])
    if not portfolio_assets:
        st.caption("Noch keine Assets. Fuege links welche hinzu.")
    for asset in portfolio_assets:
        row_label, row_button = st.columns([4, 1])
        row_label.write(f"{asset['ticker']}  ·  {asset['asset_type']}")
        if row_button.button("x", key=f"rm_{asset['ticker']}"):
            db.remove_asset_from_portfolio(conn, selected["id"], asset["ticker"])
            bump_data_version()
            st.rerun()

# --- Holdings (real positions) ---------------------------------------------
portfolio_assets = db.list_portfolio_assets(conn, selected["id"])
tickers = [a["ticker"] for a in portfolio_assets]
asset_types = {a["ticker"]: a["asset_type"] for a in portfolio_assets}
latest = db.latest_prices(conn, tickers)

if tickers:
    st.subheader("Meine Holdings")
    st.caption(
        "Trage ein, wie viel du haelst. Zellen anklicken, Wert tippen, dann speichern."
    )
    mode = st.radio("Eingabe-Modus", ["Coins", "USD", "Prozent"], horizontal=True)

    holdings = []
    for a in portfolio_assets:
        qty = a["quantity"] or 0.0
        price = latest.get(a["ticker"], 0.0)
        holdings.append(
            {"Ticker": a["ticker"], "Coins": round(qty, 8), "USD": round(qty * price, 2)}
        )
    df = pd.DataFrame(holdings)
    total_value = float(df["USD"].sum())
    df["Prozent"] = (df["USD"] / total_value * 100).round(2) if total_value > 0 else 0.0

    target_total = total_value
    if mode == "Prozent":
        target_total = st.number_input(
            "Portfoliowert gesamt ($) fuer Prozent-Eingabe",
            value=float(total_value) if total_value > 0 else 10000.0,
            min_value=0.0, step=100.0,
        )

    edit_col = mode
    disabled = ["Ticker"] + [c for c in ("Coins", "USD", "Prozent") if c != edit_col]
    with st.form("holdings_form"):
        edited = st.data_editor(
            df, disabled=disabled, hide_index=True, width="stretch", key="holdings_editor"
        )
        if st.form_submit_button("Holdings speichern"):
            for _, row in edited.iterrows():
                ticker = row["Ticker"]
                price = latest.get(ticker, 0.0)
                if mode == "Coins":
                    qty = float(row["Coins"])
                elif mode == "USD":
                    qty = float(row["USD"]) / price if price > 0 else 0.0
                else:  # Prozent of target_total
                    qty = (
                        float(row["Prozent"]) / 100 * target_total / price
                        if price > 0 else 0.0
                    )
                db.set_asset_quantity(conn, selected["id"], ticker, qty)
            bump_data_version()
            st.success("Holdings gespeichert")
            st.rerun()

# Current weights from holdings (for the frontier point and rebalancing).
quantities = {a["ticker"]: (a["quantity"] or 0.0) for a in portfolio_assets}
values = {t: quantities[t] * latest.get(t, 0.0) for t in tickers}
total_holdings = sum(values.values())
current_weights = (
    {t: values[t] / total_holdings for t in tickers} if total_holdings > 0 else {}
)

# --- Analysis dashboard -----------------------------------------------------
st.subheader("Analyse")

if not tickers:
    st.info("Fuege Assets hinzu und klick Aktualisieren.")
else:
    freq = 365 if set(asset_types.values()) == {"crypto"} else 252
    config.RISK_FREE_RATE = rf
    outcome = compute_analysis(
        tuple(tickers),
        tuple(sorted(asset_types.items())),
        tuple(sorted(current_weights.items())),
        rf,
        freq,
        max_weight,
        crypto_cap,
        cov_method,
        st.session_state.data_version,
    )

    if outcome is None:
        st.info("Noch keine Preisdaten. Klick Aktualisieren in der Seitenleiste.")
    else:
        result, figures, downside = outcome

        if result.max_sharpe is None:
            st.caption(
                "Max Sharpe nicht loesbar (kein Asset ueber der Risk-free Rate oder "
                "Caps zu eng). Kennzahlen unten beziehen sich auf Min Volatility."
            )

        tab_front, tab_corr, tab_risk, tab_rebal = st.tabs(
            ["Frontier & Gewichte", "Korrelation", "Risiko", "Rebalancing"]
        )

        with tab_front:
            st.dataframe(portfolio_comparison(result), hide_index=True, width="stretch")
            left, right = st.columns(2)
            left.plotly_chart(figures["frontier"], width="stretch")
            right.plotly_chart(figures["weights"], width="stretch")
            st.plotly_chart(figures["prices"], width="stretch")

        with tab_corr:
            c1, c2 = st.columns(2)
            c1.plotly_chart(figures["heatmap"], width="stretch")
            c2.plotly_chart(figures["dendrogram"], width="stretch")
            c3, c4 = st.columns(2)
            c3.plotly_chart(figures["rolling"], width="stretch")
            c4.plotly_chart(figures["network"], width="stretch")

        with tab_risk:
            st.plotly_chart(figures["risk"], width="stretch")
            st.markdown("**Downside-Metriken** (Portfolio Max Sharpe / Min Vol)")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Sortino", analysis._de(downside["sortino"], 2))
            d2.metric("Max Drawdown", analysis.fmt_pct(downside["max_drawdown"]))
            d3.metric("VaR 95% (Tag)", analysis.fmt_pct(downside["var95"]))
            d4.metric("CVaR 95% (Tag)", analysis.fmt_pct(downside["cvar95"]))

        with tab_rebal:
            options = {
                p.name: p
                for p in (result.max_sharpe, result.min_volatility, result.hrp)
                if p is not None
            }
            choice = st.radio("Ziel-Portfolio", list(options.keys()), horizontal=True)
            target = options[choice]
            if total_holdings <= 0:
                st.info("Trage oben deine Holdings ein, um Rebalancing zu berechnen.")
            else:
                rows, total = optimize.rebalance(target.weights, quantities, latest)
                table = pd.DataFrame(
                    [
                        {
                            "Ticker": r["ticker"],
                            "Aktuell": analysis.fmt_usd(r["current_value"]),
                            "Ist %": analysis.fmt_pct(r["current_weight"]),
                            "Ziel %": analysis.fmt_pct(r["target_weight"]),
                            "Ziel $": analysis.fmt_usd(r["target_value"]),
                            "Kauf/Verkauf $": analysis.fmt_usd(r["delta_value"]),
                            "Kauf/Verkauf Coins": analysis._de(r["delta_units"], 4),
                        }
                        for r in rows
                    ]
                )
                st.caption(f"Gesamtwert: {analysis.fmt_usd(total)} (bleibt konstant)")
                st.dataframe(table, hide_index=True, width="stretch")
