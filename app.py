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
import backtest
import config
import db
import metrics
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
    return_method: str,
    denoise: bool,
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
        return_method=return_method,
        denoise=denoise,
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

    # Regime analysis (needs enough history) and PCA factor decomposition.
    extras: dict = {"pca": None, "regime_counts": None}
    figures["regime_timeline"] = None
    figures["regime_corr"] = None
    try:
        labels = metrics.detect_regimes(prices)
        regime_corr = metrics.regime_correlations(prices, labels)
        figures["regime_timeline"] = plot.build_regime_timeline(prices, labels)
        if regime_corr:
            figures["regime_corr"] = plot.build_regime_correlations(regime_corr)
        extras["regime_counts"] = dict(labels.value_counts())
    except Exception:  # noqa: BLE001 - regime detection is best-effort
        pass
    if len(tickers) >= 2:
        pca = metrics.pca_risk(prices)
        figures["pca"] = plot.build_pca_scree(pca)
        extras["pca"] = pca
    else:
        figures["pca"] = None
    return result, figures, downside, extras


@st.cache_data(show_spinner=False)
def run_forward_sim(
    tickers: tuple[str, ...],
    weights_items: tuple[tuple[str, float], ...],
    horizon_days: int,
    n_paths: int,
    data_version: int,
):
    """Forward Monte-Carlo simulation of a weight vector, cached by inputs."""
    conn = db.connect(check_same_thread=False)
    prices = db.load_prices(conn, tickers=list(tickers))
    conn.close()
    if prices.empty:
        return None
    return metrics.forward_simulation(prices, dict(weights_items), horizon_days, n_paths)


@st.cache_data(show_spinner=False)
def run_backtest(
    tickers: tuple[str, ...],
    freq: int,
    rf: float,
    lookback_days: int,
    rebalance_days: int,
    data_version: int,
):
    """Walk-forward backtest, cached by inputs (re-optimization is expensive)."""
    conn = db.connect(check_same_thread=False)
    prices = db.load_prices(conn, tickers=list(tickers))
    conn.close()
    if prices.empty:
        return None
    return backtest.walk_forward(
        prices, lookback_days, rebalance_days, frequency=freq, risk_free_rate=rf
    )


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
    for portfolio in (
        result.max_sharpe,
        result.min_volatility,
        result.hrp,
        result.min_cvar,
        result.min_semivariance,
        result.erc,
    ):
        if portfolio is None:
            continue
        rows.append(
            {
                "Portfolio": portfolio.name,
                "Rendite p.a.": analysis.fmt_pct(portfolio.expected_return, signed=True),
                "Volatilitaet p.a.": analysis.fmt_pct(portfolio.volatility),
                "Sharpe": analysis._de(portfolio.sharpe, 2),
                "Tail (CVaR/Semi)": "-"
                if portfolio.tail_metric is None
                else analysis._de(portfolio.tail_metric, 4),
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

    return_label = st.radio(
        "Renditeschaetzer",
        ["Mean Historical", "EMA", "CAPM"],
        help="mu ist die instabilste Markowitz-Zutat. EMA gewichtet juengere Daten, "
        "CAPM nutzt Gleichgewichtsrenditen.",
    )
    return_method = {"Mean Historical": "mean_historical", "EMA": "ema", "CAPM": "capm"}[
        return_label
    ]
    denoise = st.checkbox(
        "Kovarianz denoisen (RMT)",
        value=False,
        help="Filtert Rausch-Eigenwerte der Korrelationsmatrix (Marchenko-Pastur).",
    )

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
        return_method,
        denoise,
        st.session_state.data_version,
    )

    if outcome is None:
        st.info("Noch keine Preisdaten. Klick Aktualisieren in der Seitenleiste.")
    else:
        result, figures, downside, extras = outcome

        if result.max_sharpe is None:
            st.caption(
                "Max Sharpe nicht loesbar (kein Asset ueber der Risk-free Rate oder "
                "Caps zu eng). Kennzahlen unten beziehen sich auf Min Volatility."
            )

        tab_front, tab_corr, tab_risk, tab_rebal, tab_bt, tab_sim = st.tabs(
            ["Frontier & Gewichte", "Korrelation", "Risiko", "Rebalancing",
             "Backtest", "Simulation"]
        )

        with tab_front:
            comparison = portfolio_comparison(result)
            st.dataframe(comparison, hide_index=True, width="stretch")
            st.download_button(
                "Vergleich als CSV",
                comparison.to_csv(index=False).encode("utf-8"),
                file_name="portfolios.csv",
                mime="text/csv",
            )
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

            if figures.get("regime_timeline") is not None:
                st.markdown("**Volatilitaets-Regime** (via Gaussian Mixture)")
                if extras.get("regime_counts"):
                    counts = ", ".join(f"{k}: {v} Tage" for k, v in extras["regime_counts"].items())
                    st.caption(counts + " — Korrelationen steigen typisch im Stress.")
                st.plotly_chart(figures["regime_timeline"], width="stretch")
                if figures.get("regime_corr") is not None:
                    st.plotly_chart(figures["regime_corr"], width="stretch")

        with tab_risk:
            st.plotly_chart(figures["risk"], width="stretch")
            st.markdown("**Downside-Metriken** (Portfolio Max Sharpe / Min Vol)")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Sortino", analysis._de(downside["sortino"], 2))
            d2.metric("Max Drawdown", analysis.fmt_pct(downside["max_drawdown"]))
            d3.metric("VaR 95% (Tag)", analysis.fmt_pct(downside["var95"]))
            d4.metric("CVaR 95% (Tag)", analysis.fmt_pct(downside["cvar95"]))

            if figures.get("pca") is not None:
                st.divider()
                st.markdown("**Faktor-Risikozerlegung (PCA)**")
                pca = extras["pca"]
                p1, p2 = st.columns([1, 3])
                p1.metric("Effektive Wetten", analysis._de(pca["effective_bets"], 2))
                p1.caption(
                    f"von {len(pca['labels'])} Assets — wie viele wirklich "
                    "unabhaengige Risikoquellen im Portfolio stecken."
                )
                p2.plotly_chart(figures["pca"], width="stretch")

        with tab_rebal:
            options = {
                p.name: p
                for p in (
                    result.max_sharpe,
                    result.min_volatility,
                    result.hrp,
                    result.min_cvar,
                    result.min_semivariance,
                    result.erc,
                )
                if p is not None
            }
            choice = st.radio("Ziel-Portfolio", list(options.keys()), horizontal=True)
            target = options[choice]

            st.markdown("**Rebalancing bestehender Holdings**")
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
                st.download_button(
                    "Rebalancing als CSV",
                    table.to_csv(index=False).encode("utf-8"),
                    file_name="rebalancing.csv",
                    mime="text/csv",
                )

            st.divider()
            st.markdown("**Neues Kapital investieren**")
            capital = st.number_input(
                "Neues Kapital ($)", value=1000.0, min_value=0.0, step=100.0,
                key="new_capital",
            )
            if capital > 0:
                alloc_rows, leftover = optimize.allocate_capital(
                    target.weights, latest, capital, asset_types
                )
                alloc_table = pd.DataFrame(
                    [
                        {
                            "Ticker": r["ticker"],
                            "Ziel %": analysis.fmt_pct(r["target_weight"]),
                            "Kaufen (Stueck/Coins)": analysis._de(r["units"], 4),
                            "Wert": analysis.fmt_usd(r["value"]),
                        }
                        for r in alloc_rows
                    ]
                )
                st.dataframe(alloc_table, hide_index=True, width="stretch")
                st.caption(f"Nicht investierter Rest: {analysis.fmt_usd(leftover)}")

        with tab_bt:
            st.markdown(
                "Walk-forward: rollierend auf einem Fenster optimieren, out-of-sample "
                "halten, gegen Equal-Weight und Buy&Hold-BTC vergleichen."
            )
            bt_c1, bt_c2 = st.columns(2)
            bt_lookback = int(
                bt_c1.number_input(
                    "Lookback (Tage)", value=180, min_value=30, step=30, key="bt_lookback"
                )
            )
            bt_rebalance = int(
                bt_c2.number_input(
                    "Rebalance alle (Tage)", value=30, min_value=5, step=5, key="bt_rebalance"
                )
            )
            if st.button("Backtest laufen", type="primary"):
                try:
                    with st.spinner("Backtest laeuft ..."):
                        st.session_state.backtest_result = run_backtest(
                            tuple(tickers), freq, rf, bt_lookback, bt_rebalance,
                            st.session_state.data_version,
                        )
                except ValueError as exc:
                    st.session_state.backtest_result = None
                    st.error(str(exc))

            bt_outcome = st.session_state.get("backtest_result")
            if bt_outcome is not None:
                curves, stats, dsr_info = bt_outcome
                st.plotly_chart(plot.build_backtest_curves(curves), width="stretch")
                disp = pd.DataFrame(
                    {
                        "Strategie": stats["Strategie"],
                        "Rendite p.a.": stats["Rendite p.a."].map(
                            lambda v: analysis.fmt_pct(v, signed=True)
                        ),
                        "Vola p.a.": stats["Vola p.a."].map(analysis.fmt_pct),
                        "Sharpe": stats["Sharpe"].map(lambda v: analysis._de(v, 2)),
                        "Max Drawdown": stats["Max Drawdown"].map(analysis.fmt_pct),
                        "Endwert": stats["Endwert"].map(lambda v: analysis._de(v, 2)),
                        "PSR": stats["PSR"].map(analysis.fmt_pct),
                    }
                )
                st.dataframe(disp, hide_index=True, width="stretch")
                st.caption(
                    f"PSR = Wahrscheinlichkeit, dass der wahre Sharpe > 0 ist. "
                    f"Beste Strategie **{dsr_info['best']}**: Deflated Sharpe "
                    f"{analysis.fmt_pct(dsr_info['dsr'])} (nach Korrektur dafuer, dass "
                    f"{len(backtest.STRATEGIES)} Strategien getestet wurden). "
                    "Nahe 100% = wahrscheinlich echt, nicht Glueck."
                )

        with tab_sim:
            st.markdown(
                "Forward Monte-Carlo: bootstrappt die historischen Tagesrenditen des "
                "gewaehlten Portfolios in die Zukunft (Faecher = Perzentil-Baender)."
            )
            sim_options = {
                p.name: p
                for p in (
                    result.max_sharpe,
                    result.min_volatility,
                    result.hrp,
                    result.min_cvar,
                    result.min_semivariance,
                    result.erc,
                )
                if p is not None
            }
            if current_weights:
                sim_options = {"Aktuell": None, **sim_options}
            sim_choice = st.radio(
                "Portfolio", list(sim_options.keys()), horizontal=True, key="sim_choice"
            )
            sim_weights = (
                current_weights
                if sim_choice == "Aktuell"
                else sim_options[sim_choice].weights
            )
            s1, s2 = st.columns(2)
            horizon = int(
                s1.number_input("Horizont (Tage)", value=252, min_value=10, step=10, key="sim_h")
            )
            n_paths = int(
                s2.number_input("Pfade", value=5000, min_value=500, step=500, key="sim_n")
            )
            if st.button("Simulieren", type="primary"):
                st.session_state.sim_result = run_forward_sim(
                    tuple(tickers), tuple(sorted(sim_weights.items())), horizon, n_paths,
                    st.session_state.data_version,
                )

            sim = st.session_state.get("sim_result")
            if sim is not None:
                st.plotly_chart(plot.build_forward_fanchart(sim["bands"]), width="stretch")
                q1, q2, q3 = st.columns(3)
                q1.metric("P(Verlust)", analysis.fmt_pct(sim["p_loss"]))
                q2.metric("Median-Endwert", f"{analysis._de(sim['median_terminal'], 2)}x")
                q3.metric("CVaR 5% (Endwert)", f"{analysis._de(sim['cvar5_terminal'], 2)}x")
