import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import os
import time
from functools import wraps

from styles import inject_styles
from data_service import load_stock_list, fetch_prices, _extract_series, fetch_live_quotes, fetch_market_indices, fetch_intraday, format_currency, format_volume, retry_yf
from risk_engine import compute_returns, parametric_var, historical_var, monte_carlo_var, portfolio_var, distribution_stats, backtest_var
from charts import COLORS, plot_intraday, plot_distribution, plot_price_with_var, plot_correlation_heatmap, plot_mc_histogram, plot_backtest, plot_weights_pie

warnings.filterwarnings("ignore")

def main_page():
    """
    Value at Risk (VaR) Calculator
    Portfolio Risk Analytics Tool | Built with Streamlit + yfinance
    """
    # ─────────────────────────────────────────────
    # PAGE CONFIG
    # ─────────────────────────────────────────────
    st.set_page_config(
        page_title="VaRify",
        page_icon="📉",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    inject_styles()
    
    with st.sidebar:
        st.markdown("## 📉 VaR Calculator")
        st.markdown("---")
    
        # Country
        country = st.selectbox(
            "Market",
            ["India (NSE)", "US (S&P 500)"],
            index=0,
        )
    
        stock_df = load_stock_list(country)
        suffix = ".NS" if country == "India (NSE)" else ""
    
        # Build search options: "SYMBOL — Name (Sector)"
        stock_df["display"] = (
            stock_df["symbol"] + " — " + stock_df["name"] + " (" + stock_df["sector"] + ")"
        )
        display_to_symbol = dict(zip(stock_df["display"], stock_df["symbol"]))
    
        st.markdown("#### Stock selection")
        selected_displays = st.multiselect(
            "Search and select stocks",
            options=stock_df["display"].tolist(),
            default=[stock_df["display"].iloc[0]],
            help="Type to search by name, symbol, or sector",
        )
    
        if not selected_displays:
            st.warning("Select at least one stock.")
            st.stop()
    
        selected_symbols = [display_to_symbol[d] for d in selected_displays]
        tickers_with_suffix = [s + suffix for s in selected_symbols]
        n_stocks = len(selected_symbols)
    
        # Portfolio weights (only for multi-stock)
        weights = []
        if n_stocks > 1:
            st.markdown("#### Portfolio weights (%)")
            equal = round(100 / n_stocks, 1)
            total_w = 0
            for sym in selected_symbols:
                w = st.number_input(
                    sym,
                    min_value=0.0,
                    max_value=100.0,
                    value=equal,
                    step=0.5,
                    format="%.1f",
                    key=f"w_{sym}",
                )
                weights.append(w)
                total_w += w
            if abs(total_w - 100) > 0.5:
                st.error(f"Weights sum to {total_w:.1f}% — must equal 100%.")
                st.stop()
            weights = [w / 100 for w in weights]
        else:
            weights = [1.0]
    
        st.markdown("#### Parameters")
        position_size = st.number_input(
            "Position size (₹)" if country == "India (NSE)" else "Position size ($)",
            min_value=10_000,
            max_value=100_000_000,
            value=1_000_000,
            step=50_000,
            format="%d",
        )
    
        conf_opts = {"90%": 0.90, "95%": 0.95, "99%": 0.99, "99.9%": 0.999}
        conf_label = st.selectbox("Confidence level", list(conf_opts.keys()), index=1)
        conf = conf_opts[conf_label]
    
        horizon_opts = {
            "1 day": 1, "5 days (1W)": 5, "10 days (2W)": 10,
            "21 days (1M)": 21, "63 days (3M)": 63,
            "126 days (6M)": 126, "189 days (9M)": 189, "252 days (1Y)": 252,
        }
        horizon_label = st.selectbox("Holding period", list(horizon_opts.keys()), index=0)
        horizon = horizon_opts[horizon_label]
    
        lookback_opts = {"1 Year": 252, "2 Years": 504, "3 Years": 756, "5 Years": 1260}
        lookback_label = st.selectbox("Historical lookback", list(lookback_opts.keys()), index=0)
        lookback_days = lookback_opts[lookback_label]
    
        rf_default = 6.5 if country == "India (NSE)" else 4.0
        risk_free_rate = st.number_input(
            "Risk-free rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=rf_default,
            step=0.1,
            help="Used to compute risk-adjusted performance metrics like Sharpe Ratio."
        ) / 100.0
    
        method = st.radio(
            "VaR method",
            ["Parametric", "Historical", "Monte Carlo"],
            index=0,
            help="Parametric: assumes normality | Historical: uses actual returns | Monte Carlo: simulates 10k paths",
        )
    
        st.markdown("---")
        run_btn = st.button("▶  Calculate VaR", type="primary", use_container_width=True)
        if run_btn:
            st.session_state.calculated = True
    
    # ─────────────────────────────────────────────
    # MAIN PANEL
    # ─────────────────────────────────────────────
    st.markdown("# 📉 VaRify: Value at Risk Calculator")
    st.markdown(
        f"**{country}** &nbsp;|&nbsp; "
        f"**{', '.join(selected_symbols)}** &nbsp;|&nbsp; "
        f"Conf: **{conf_label}** &nbsp;|&nbsp; "
        f"Horizon: **{horizon_label}** &nbsp;|&nbsp; "
        f"Method: **{method}**"
    )
    
    # First run splash
    if "first_run" not in st.session_state:
        st.session_state.first_run = True
        st.info("👋 **Welcome to VaRify!** Start by picking a stock you already own in the sidebar, enter how much you've invested, and click **Calculate VaR**. We'll show you how much you could lose on a bad day — and what tools exist to manage that risk.")
    
    with st.expander("🎓 **Learn about VaR (Value at Risk)**"):
        st.markdown("""
        **What is VaR?**
        VaR answers one question: *What's the most I could lose on a really bad day (or week, or month)?*
        Think of it like checking the weather forecast. It doesn't tell you for certain it will rain, but if there's a 95% chance it won't rain, VaR tells you how badly you'll get soaked in that unlucky 5%.
        
        **What is CVaR / Expected Shortfall?**
        If VaR is the door to the danger zone, CVaR tells you the *average damage* once you walk through it.
        
        **Which method should I pick?**
        * **Parametric:** Uses a standard bell curve. Fast, but might underestimate extreme crashes (fat tails).
        * **Historical:** Uses actual past daily returns. No bell curve assumed, so it captures real-world weirdness.
        * **Monte Carlo:** Simulates 10,000 possible future paths. Great for stress testing.
        """)
    
    st.markdown("---")
    
    # ─────────────────────────────────────────────
    # LIVE MARKET DATA PANEL
    # ─────────────────────────────────────────────
    currency = "₹" if country == "India (NSE)" else "$"
    
    with st.spinner("Fetching live market data…"):
        _indices = fetch_market_indices(country)
        _quotes = fetch_live_quotes(
            tuple(tickers_with_suffix), tuple(selected_symbols),
        )
    
    # --- Market index ticker bar ---
    if _indices:
        st.markdown(
            '<div class="section-header">📡 Live market overview</div>',
            unsafe_allow_html=True,
        )
        idx_cols = st.columns(len(_indices))
        for _col, (_name, _d) in zip(idx_cols, _indices.items()):
            _arrow = "▲" if _d["change"] >= 0 else "▼"
            _clr = "#2ca02c" if _d["change"] >= 0 else "#d62728"
            _sign = "+" if _d["change"] >= 0 else ""
            _col.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:0.8rem;color:#adb5bd;">{_name}</div>
                <div style="font-size:1.3rem;font-weight:700;">{_d['price']:,.1f}</div>
                <div style="color:{_clr};font-size:0.85rem;font-weight:600;">
                    {_arrow} {_sign}{_d['change']:.1f} ({_sign}{_d['change_pct']:.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # --- Live stock quote cards ---
    if _quotes:
        st.markdown(
            '<div class="section-header">💹 Live stock data'
            ' <span style="font-size:0.75rem;color:#adb5bd;">'
            '(auto-refreshes every 5 min)</span></div>',
            unsafe_allow_html=True,
        )
        _items = list(_quotes.items())
        for _row_start in range(0, len(_items), 4):
            _row = _items[_row_start : _row_start + 4]
            _card_cols = st.columns(len(_row))
            for _cc, (_sym, _q) in zip(_card_cols, _row):
                _arrow = "▲" if _q["change"] >= 0 else "▼"
                _clr = "#2ca02c" if _q["change"] >= 0 else "#d62728"
                _sign = "+" if _q["change"] >= 0 else ""
                _cc.markdown(f"""
                <div style="background:rgba(255,255,255,0.05);
                            border:1px solid rgba(255,255,255,0.1);
                            border-radius:10px;padding:1rem;margin-bottom:0.5rem;">
                    <div style="font-size:0.85rem;color:#adb5bd;">{_sym}</div>
                    <div style="font-size:1.5rem;font-weight:700;">
                        {currency}{_q['price']:,.2f}
                    </div>
                    <div style="color:{_clr};font-size:0.9rem;font-weight:600;">
                        {_arrow} {_sign}{_q['change']:.2f} ({_sign}{_q['change_pct']:.2f}%)
                    </div>
                    <div style="font-size:0.75rem;color:#6c757d;margin-top:4px;">
                        H: {currency}{_q['day_high']:,.2f} &nbsp;|&nbsp;
                        L: {currency}{_q['day_low']:,.2f} &nbsp;|&nbsp;
                        Vol: {format_volume(_q['volume'])}
                    </div>
                    <div style="font-size:0.7rem;color:#6c757d;">
                        Prev close: {currency}{_q['prev_close']:,.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
        # Intraday chart in expander
        with st.expander("📈 Intraday price chart (15-min intervals)", expanded=False):
            _intraday = fetch_intraday(
                tuple(tickers_with_suffix), tuple(selected_symbols),
            )
            if not _intraday.empty:
                st.plotly_chart(plot_intraday(_intraday), use_container_width=True)
            else:
                st.info("Intraday data not available — the market may be closed.")
    
    # Refresh button + timestamp
    _ref1, _ref2, _ = st.columns([1, 2, 4])
    with _ref1:
        if st.button("🔄 Refresh data", use_container_width=True):
            fetch_live_quotes.clear()
            fetch_market_indices.clear()
            fetch_intraday.clear()
            st.rerun()
    with _ref2:
        st.markdown(
            f'<span style="font-size:0.75rem;color:#adb5bd;line-height:2.5rem;">'
            f'Last updated: {datetime.now().strftime("%H:%M:%S")}</span>',
            unsafe_allow_html=True,
        )
    
    st.markdown("---")
    
    if not st.session_state.get("calculated", False):
        st.markdown("""
        <div class="info-box">
            ℹ️ Configure parameters in the sidebar and click <strong>Calculate VaR</strong> to run the analysis.
            <br><br>
            <strong>What this tool computes:</strong><br>
            • <strong>Parametric VaR</strong> — uses mean and standard deviation assuming a normal distribution<br>
            • <strong>Historical VaR</strong> — uses the actual historical return distribution (no normality assumption)<br>
            • <strong>Monte Carlo VaR</strong> — simulates 10,000 price paths using GBM<br>
            • <strong>CVaR / Expected Shortfall</strong> — expected loss when VaR is breached<br>
            • <strong>Breach backtesting</strong> — Kupiec test to validate your VaR model<br>
            • <strong>Correlation matrix</strong> — shows diversification effect for portfolios
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    
    # ─────────────────────────────────────────────
    # DATA FETCH
    # ─────────────────────────────────────────────
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=int(lookback_days * 1.4))  # buffer for weekends/holidays
    
    with st.spinner("Fetching price data from Yahoo Finance…"):
        benchmark_ticker = "^NSEI" if country == "India (NSE)" else "^GSPC"
        tickers_to_fetch = list(tickers_with_suffix)
        benchmark_added = False
        if benchmark_ticker not in tickers_to_fetch:
            tickers_to_fetch.append(benchmark_ticker)
            benchmark_added = True
    
        prices_raw = fetch_prices(
            tuple(tickers_to_fetch),
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )
    
    # Handle single vs multi ticker column structure
    if prices_raw is None or prices_raw.empty:
        st.error("No data returned. Check ticker symbols or try a different date range.")
        st.stop()
    
    if isinstance(prices_raw.columns, pd.MultiIndex):
        # In case it wasn't flattened, try to find the level with tickers
        level = 1 if len(set(tickers_to_fetch).intersection(prices_raw.columns.get_level_values(1))) > 0 else 0
        prices_raw.columns = prices_raw.columns.get_level_values(level)
    
    if benchmark_ticker in prices_raw.columns:
        benchmark_prices = prices_raw[benchmark_ticker].copy()
        if benchmark_added:
            prices_raw = prices_raw.drop(columns=[benchmark_ticker])
    else:
        benchmark_prices = pd.Series(dtype=float)
    
    # Clean up columns for single ticker
    if n_stocks == 1:
        prices_raw.columns = [selected_symbols[0]]
    else:
        # Always strip the suffix (e.g., '.NS') so columns match selected_symbols
        prices_raw.columns = [str(c).replace(suffix, "") if suffix else str(c) for c in prices_raw.columns]
    
    # Keep only last `lookback_days` trading days
    prices_raw = prices_raw.dropna(how="all").tail(lookback_days)
    
    if prices_raw.empty or len(prices_raw) < 30:
        st.error(f"Insufficient data ({len(prices_raw)} rows). Try a longer lookback or check the ticker.")
        st.stop()
    
    returns_df = compute_returns(prices_raw)
    
    # ─────────────────────────────────────────────
    # COMPUTE VaR
    # ─────────────────────────────────────────────
    currency = "₹" if country == "India (NSE)" else "$"
    
    # Short-history check for Historical method
    if method == "Historical" and len(returns_df) < 252:
        st.warning("⚠️ Insufficient History for Historical Risk Modeling (< 252 days). Falling back to Parametric VaR.")
        method = "Parametric"
    
    if n_stocks == 1:
        sym = selected_symbols[0]
        rets = returns_df[sym].dropna() if sym in returns_df.columns else returns_df.iloc[:, 0].dropna()
        prices_s = prices_raw[sym] if sym in prices_raw.columns else prices_raw.iloc[:, 0]
    
        if method == "Parametric":
            result = parametric_var(rets, conf, horizon)
            daily_var_pct = parametric_var(rets, conf, 1)["var_pct"]
        elif method == "Historical":
            result = historical_var(rets, conf, horizon)
            daily_var_pct = historical_var(rets, conf, 1)["var_pct"]
        else:
            result = monte_carlo_var(rets, conf, horizon)
            daily_var_pct = monte_carlo_var(rets, conf, 1)["var_pct"]
    
        var_pct  = result["var_pct"]
        cvar_pct = result["cvar_pct"]
        var_amt  = var_pct  * position_size
        cvar_amt = cvar_pct * position_size
        cutoff_price = position_size * (1 - var_pct)
        ann_vol  = rets.std() * np.sqrt(252)
        ann_ret  = rets.mean() * 252
        dist     = distribution_stats(rets)
        bt       = backtest_var(rets, daily_var_pct, conf)
    
    else:
        weights_arr = np.array(weights)
        rets_aligned = returns_df[selected_symbols].dropna()
        port_ret_series = rets_aligned @ weights_arr
        
        if method == "Parametric":
            result = portfolio_var(rets_aligned, weights_arr, conf, horizon)
            daily_var_pct = portfolio_var(rets_aligned, weights_arr, conf, 1)["var_pct"]
        elif method == "Historical":
            result = historical_var(port_ret_series, conf, horizon)
            daily_var_pct = historical_var(port_ret_series, conf, 1)["var_pct"]
        else:
            result = monte_carlo_var(port_ret_series, conf, horizon)
            daily_var_pct = monte_carlo_var(port_ret_series, conf, 1)["var_pct"]
        
        # Compute Naive VaR generically for diversification benefit
        if method != "Parametric":
            naive_vars = []
            for i, sym in enumerate(selected_symbols):
                r = rets_aligned[sym]
                if method == "Historical":
                    v = historical_var(r, conf, horizon)["var_pct"]
                else:
                    v = monte_carlo_var(r, conf, horizon)["var_pct"]
                naive_vars.append(weights_arr[i] * v)
            result["naive_var_pct"] = sum(naive_vars)
            result["div_benefit_pct"] = result["naive_var_pct"] - result["var_pct"]
        
        var_pct  = result["var_pct"]
        cvar_pct = result["cvar_pct"]
        var_amt  = var_pct  * position_size
        cvar_amt = cvar_pct * position_size
        cutoff_price = position_size * (1 - var_pct)
        ann_vol  = port_ret_series.std() * np.sqrt(252)
        ann_ret  = port_ret_series.mean() * 252
        dist     = distribution_stats(port_ret_series)
        bt       = backtest_var(port_ret_series, daily_var_pct, conf)
    
    final_rets = rets if n_stocks == 1 else port_ret_series
    if not benchmark_prices.empty:
        benchmark_rets = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()
        
        # Align dates
        aligned = pd.concat([final_rets, benchmark_rets], axis=1).dropna()
        if len(aligned) > 2:
            cov_mat = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
            beta = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] != 0 else 1.0
        else:
            beta = 1.0
    
        if method == "Parametric":
            bm_var_pct = parametric_var(benchmark_rets, conf, horizon)["var_pct"]
        elif method == "Historical":
            bm_var_pct = historical_var(benchmark_rets, conf, horizon)["var_pct"]
        else:
            bm_var_pct = monte_carlo_var(benchmark_rets, conf, horizon)["var_pct"]
    else:
        beta = 1.0
        bm_var_pct = var_pct  # fallback
    
    # ─────────────────────────────────────────────
    # METRIC CARDS
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Risk summary</div>', unsafe_allow_html=True)
    
    # Traffic Light VaR Logic
    if bm_var_pct > 0:
        var_ratio = var_pct / bm_var_pct
    else:
        var_ratio = 1.0
    
    if var_ratio < 1.0:
        traffic_light = "🟢 Low Risk"
    elif var_ratio <= 1.5:
        traffic_light = "🟡 Moderate Risk"
    else:
        traffic_light = "🔴 High Risk"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=f"Value at Risk (95%)",
            value=format_currency(var_amt, currency),
            delta=f"{traffic_light} vs. Benchmark",
            delta_color="off"
        )
    with col2:
        st.metric(
            label=f"Expected Shortfall (CVaR)",
            value=format_currency(cvar_amt, currency),
            delta=f"{cvar_pct*100:.2f}% of position",
            delta_color="inverse"
        )
    with col3:
        bm_name = "Nifty 50" if country == "India (NSE)" else "S&P 500"
        st.metric(
            label=f"Beta vs. {bm_name}",
            value=f"{beta:.2f}",
            delta="1.0 = Market Volatility",
            delta_color="off"
        )
    
    # --- Natural Language Summary ---
    summary_name = "your portfolio" if n_stocks > 1 else selected_symbols[0]
    st.markdown(f"""
    <div style="font-size:1.05rem; padding:1.2rem; margin-top:1rem; border-left: 4px solid #0c5460; background: #1a1a2e;">
        💡 <strong>What this means for you:</strong> With {conf_label} confidence, your {format_currency(position_size, currency)} position in {summary_name} 
        could lose up to <strong>{format_currency(var_amt, currency)}</strong> over the next <strong>{horizon} trading day(s)</strong>. 
        On the absolute worst {100-conf*100:.1f}% of periods, you'd lose an average of <strong>{format_currency(cvar_amt, currency)}</strong>.<br><br>
        <small style="color: #adb5bd;"><em>Note: VaR is an estimate under normal market conditions. Black Swan events can exceed these estimates.</em></small>
    </div>
    """, unsafe_allow_html=True)
    
    
    # ─────────────────────────────────────────────
    # DISTRIBUTION STATS (single stock)
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Distribution diagnostics</div>', unsafe_allow_html=True)
    
    ds_col1, ds_col2, ds_col3, ds_col4, ds_col5 = st.columns(5)
    
    skew_color  = "badge-yellow" if abs(dist["skewness"]) > 0.5 else "badge-green"
    kurt_color  = "badge-red"    if dist["excess_kurtosis"] > 3  else "badge-green"
    norm_color  = "badge-green"  if dist["is_normal"] else "badge-red"
    norm_label  = "Normal ✓"     if dist["is_normal"] else "Non-normal ✗"
    
    with ds_col1:
        skew_icon = "🟡" if abs(dist["skewness"]) > 0.5 else "🟢"
        st.markdown(f"""
        <div style="text-align:center" title="Skewness measures asymmetry. High negative skew means bigger crashes.">
            <div style="font-size:0.8rem;color:#6c757d;">Skewness</div>
            <div style="font-size:1.5rem;font-weight:700">{dist['skewness']:.3f}</div>
            <span class="badge {skew_color}">{skew_icon} {'Negative skew' if dist['skewness'] < -0.5 else 'Positive skew' if dist['skewness'] > 0.5 else 'Approx. symmetric'}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with ds_col2:
        kurt_icon = "🔴" if dist["excess_kurtosis"] > 3 else "🟢"
        st.markdown(f"""
        <div style="text-align:center" title="Fat tails = extreme days happen more often than a bell curve predicts.">
            <div style="font-size:0.8rem;color:#6c757d;">Excess kurtosis</div>
            <div style="font-size:1.5rem;font-weight:700">{dist['excess_kurtosis']:.3f}</div>
            <span class="badge {kurt_color}">{kurt_icon} {'Fat tails ⚠' if dist['excess_kurtosis'] > 3 else 'Normal tails'}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with ds_col3:
        norm_icon = "🟢" if dist["is_normal"] else "🔴"
        st.markdown(f"""
        <div style="text-align:center" title="Tests if returns follow a bell curve. If not, Parametric VaR may undercount risk.">
            <div style="font-size:0.8rem;color:#6c757d;">Jarque-Bera stat</div>
            <div style="font-size:1.5rem;font-weight:700">{dist['jb_stat']:.1f}</div>
            <span class="badge {norm_color}">{norm_icon} {norm_label}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with ds_col4:
        breach_color = "badge-green" if bt["breach_rate"] <= bt["expected_rate"] * 1.5 else "badge-red"
        breach_icon = "🟢" if bt["breach_rate"] <= bt["expected_rate"] * 1.5 else "🔴"
        st.markdown(f"""
        <div style="text-align:center" title="How many times did the actual loss exceed the VaR estimate in the past?">
            <div style="font-size:0.8rem;color:#6c757d;">VaR breaches (hist.)</div>
            <div style="font-size:1.5rem;font-weight:700">{bt['breaches']} / {bt['total']}</div>
            <span class="badge {breach_color}">{breach_icon} {bt['breach_rate']*100:.1f}% vs {bt['expected_rate']*100:.0f}% expected</span>
        </div>
        """, unsafe_allow_html=True)
    
    with ds_col5:
        if not np.isnan(bt["pof_pvalue"]):
            pof_color = "badge-green" if bt["pof_pvalue"] > 0.05 else "badge-red"
            pof_label = "Model valid ✓" if bt["pof_pvalue"] > 0.05 else "Model rejected ✗"
            pof_icon = "🟢" if bt["pof_pvalue"] > 0.05 else "🔴"
        else:
            pof_color = "badge-yellow"
            pof_label = "Insufficient data"
            pof_icon = "🟡"
        pof_display = f"{bt['pof_pvalue']:.3f}" if not np.isnan(bt['pof_pvalue']) else "N/A"
        st.markdown(f"""
        <div style="text-align:center" title="A statistical check: does this VaR model actually work? ✓ = yes, ✗ = back to the drawing board">
            <div style="font-size:0.8rem;color:#6c757d;">Kupiec test (p-value)</div>
            <div style="font-size:1.5rem;font-weight:700">{pof_display}</div>
            <span class="badge {pof_color}">{pof_icon} {pof_label}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Normality warning
    if not dist["is_normal"]:
        st.markdown("""
        <div class="warn-box">
            ⚠️ <strong>Normality rejected (Jarque-Bera p &lt; 0.05)</strong> — The return distribution has fat tails or significant skewness.
            Parametric VaR may <strong>underestimate</strong> actual tail risk. Consider using Historical or Monte Carlo VaR instead.
        </div>
        """, unsafe_allow_html=True)
    
    
    # ─────────────────────────────────────────────
    # CHARTS — Row 1
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Distribution & price charts</div>', unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns([1.1, 1])
    
    with chart_col1:
        label = " + ".join(selected_symbols) if n_stocks > 1 else selected_symbols[0]
        if method == "Parametric":
            dist_ret = rets if n_stocks == 1 else port_ret_series
            fig_dist = plot_distribution(dist_ret, var_pct, cvar_pct, conf, method, label)
        elif method == "Historical":
            dist_ret = rets if n_stocks == 1 else port_ret_series
            fig_dist = plot_distribution(dist_ret, var_pct, cvar_pct, conf, method, label)
        else:
            dist_ret = rets if n_stocks == 1 else port_ret_series
            fig_dist = plot_mc_histogram(result["sim"], var_pct, cvar_pct, conf)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with chart_col2:
        if n_stocks == 1:
            fig_price = plot_price_with_var(prices_s, rets, var_pct, position_size, selected_symbols[0])
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            # Portfolio weights pie
            st.plotly_chart(plot_weights_pie(weights, selected_symbols), use_container_width=True)
    
    
    # ─────────────────────────────────────────────
    # PORTFOLIO: Correlation + Diversification
    # ─────────────────────────────────────────────
    if n_stocks > 1:
        st.markdown('<div class="section-header">Portfolio analytics</div>', unsafe_allow_html=True)
        corr_col, div_col = st.columns([1.4, 1])
    
        with corr_col:
            st.plotly_chart(plot_correlation_heatmap(rets_aligned), use_container_width=True)
    
        with div_col:
            st.markdown("#### Diversification benefit")
            naive  = result["naive_var_pct"]
            actual = result["var_pct"]
            benefit_pct = result["div_benefit_pct"]
            benefit_amt = benefit_pct * position_size
    
            st.metric("Naive VaR (no correlation)",
                      format_currency(naive * position_size, currency),
                      f"{naive*100:.2f}%")
            st.metric("Correlation-adjusted VaR",
                      format_currency(actual * position_size, currency),
                      f"{actual*100:.2f}%",
                      delta_color="inverse")
            st.metric("Diversification benefit",
                      format_currency(benefit_amt, currency),
                      f"{benefit_pct*100:.2f}% saved by diversification",
                      delta_color="normal")
    
            st.markdown("""
            <div class="info-box">
                <strong>How to read this:</strong> Naive VaR assumes zero correlation between stocks.
                The actual VaR accounts for real-world co-movement. The difference is your
                <em>diversification benefit</em> — how much risk you've reduced by not holding a single stock.
            </div>
            """, unsafe_allow_html=True)
    
            # Per-stock individual VaR
            st.markdown("#### Individual stock VaR")
            rows = []
            for i, sym in enumerate(selected_symbols):
                r = rets_aligned[sym]
                pv = parametric_var(r, conf, horizon)
                alloc = weights[i] * position_size
                rows.append({
                    "Ticker": sym,
                    "Weight": f"{weights[i]*100:.1f}%",
                    "Allocation": format_currency(alloc, currency),
                    "Ind. VaR %": f"{pv['var_pct']*100:.2f}%",
                    "Ind. VaR ₹/$": format_currency(pv['var_pct']*alloc, currency),
                    "Ann. Vol": f"{r.std()*np.sqrt(252)*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    
    # ─────────────────────────────────────────────
    # BACKTEST CHART
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-header">VaR breach backtesting</div>', unsafe_allow_html=True)
    
    bt_ret = rets if n_stocks == 1 else port_ret_series
    bt_sym = selected_symbols[0] if n_stocks == 1 else "Portfolio"
    
    fig_bt = plot_backtest(bt_ret, daily_var_pct, bt_sym)
    st.plotly_chart(fig_bt, use_container_width=True)
    
    with st.expander("Audit & Backtest Metrics"):
        breach_summary = backtest_var(bt_ret, daily_var_pct, conf)
        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            st.metric("Total breach days", breach_summary["breaches"])
        with b_col2:
            st.metric("Observed breach rate", f"{breach_summary['breach_rate']*100:.2f}%",
                      delta=f"Expected: {breach_summary['expected_rate']*100:.1f}%",
                      delta_color="off")
        with b_col3:
            pv = breach_summary["pof_pvalue"]
            validity = "✓ Valid" if (not np.isnan(pv) and pv > 0.05) else "✗ Rejected"
            st.metric("Kupiec test", validity,
                      delta=f"p = {pv:.3f}" if not np.isnan(pv) else "N/A",
                      delta_color="off")
    
    
    # ─────────────────────────────────────────────
    # ALL-METHODS COMPARISON TABLE
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Method comparison</div>', unsafe_allow_html=True)
    
    compare_rows = []
    ret_for_compare = rets if n_stocks == 1 else port_ret_series
    
    for m_name, m_func in [
        ("Parametric", lambda r, c, h: parametric_var(r, c, h)),
        ("Historical", lambda r, c, h: historical_var(r, c, h)),
        ("Monte Carlo", lambda r, c, h: monte_carlo_var(r, c, h)),
    ]:
        res = m_func(ret_for_compare, conf, horizon)
        vp = res["var_pct"]
        cp = res["cvar_pct"]
        compare_rows.append({
            "Method": m_name,
            f"VaR % ({conf_label})": f"{vp*100:.3f}%",
            f"VaR {currency}": format_currency(vp*position_size, currency),
            f"CVaR % ({conf_label})": f"{cp*100:.3f}%",
            f"CVaR {currency}": format_currency(cp*position_size, currency),
            "Assumes normality": "Yes" if m_name == "Parametric" else "No",
        })
    
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)
    
    
    # ─────────────────────────────────────────────
    # ROLLING VaR CHART (for multi-stock: portfolio returns)
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Rolling 30-day VaR over time</div>', unsafe_allow_html=True)
    
    roll_ret = rets if n_stocks == 1 else port_ret_series
    window = 30
    roll_mu_s    = roll_ret.rolling(window).mean()
    roll_sig_s   = roll_ret.rolling(window).std()
    z_conf       = stats.norm.ppf(1 - conf)
    rolling_var  = -(roll_mu_s * horizon + z_conf * roll_sig_s * np.sqrt(horizon))
    rolling_var_amt = rolling_var * position_size
    
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=rolling_var_amt.index,
        y=rolling_var_amt.values,
        mode="lines",
        name=f"Rolling VaR ({conf_label})",
        line=dict(color=COLORS["danger"], width=2),
        fill="tozeroy",
        fillcolor="rgba(214,39,40,0.08)",
    ))
    fig_roll.add_hline(y=var_amt, line_dash="dash", line_color=COLORS["primary"],
                       annotation_text=f"  Full-period VaR: {format_currency(var_amt, currency)}",
                       annotation_font_color=COLORS["primary"])
    fig_roll.update_layout(
        title=dict(text=f"Rolling {window}-day parametric VaR — {bt_sym}", font_size=14),
        xaxis_title="Date",
        yaxis_title=f"VaR ({currency})",
        height=340,
        margin=dict(t=60, b=40, l=70, r=30),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0", tickformat=",.0f"),
    )
    st.plotly_chart(fig_roll, use_container_width=True)
    
    
    # ─────────────────────────────────────────────
    # DATA TABLE + CSV EXPORT
    # ─────────────────────────────────────────────
    with st.expander("📋 Raw price & returns data"):
        tab1, tab2 = st.tabs(["Prices", "Log returns"])
        with tab1:
            st.dataframe(prices_raw.tail(50).style.format("{:.2f}"), use_container_width=True)
        with tab2:
            st.dataframe(returns_df.tail(50).style.format("{:.4f}"), use_container_width=True)
    
    st.markdown('<div class="section-header">Export results</div>', unsafe_allow_html=True)
    
    # Build summary export dataframe
    export_rows = []
    ret_for_export = rets if n_stocks == 1 else port_ret_series
    for m_name, m_func in [
        ("Parametric", lambda r, c, h: parametric_var(r, c, h)),
        ("Historical", lambda r, c, h: historical_var(r, c, h)),
        ("Monte Carlo", lambda r, c, h: monte_carlo_var(r, c, h)),
    ]:
        res = m_func(ret_for_export, conf, horizon)
        export_rows.append({
            "Ticker(s)": " + ".join(selected_symbols),
            "Method": m_name,
            "Confidence": conf_label,
            "Horizon (days)": horizon,
            "Position Size": position_size,
            "VaR (%)": round(res["var_pct"] * 100, 4),
            f"VaR ({currency})": round(res["var_pct"] * position_size, 2),
            "CVaR (%)": round(res["cvar_pct"] * 100, 4),
            f"CVaR ({currency})": round(res["cvar_pct"] * position_size, 2),
            "Ann. Volatility (%)": round(ret_for_export.std() * np.sqrt(252) * 100, 4),
            "Ann. Return (%)": round(ret_for_export.mean() * 252 * 100, 4),
            "Skewness": round(dist["skewness"], 4),
            "Excess Kurtosis": round(dist["excess_kurtosis"], 4),
            "JB p-value": round(dist["jb_pvalue"], 4),
            "Normal Distribution": "Yes" if dist["is_normal"] else "No",
            "VaR Breaches": bt["breaches"],
            "Breach Rate (%)": round(bt["breach_rate"] * 100, 2),
            "Kupiec p-value": round(bt["pof_pvalue"], 4) if not np.isnan(bt["pof_pvalue"]) else "N/A",
            "Lookback": lookback_label,
            "Market": country,
            "Run Date": datetime.today().strftime("%Y-%m-%d"),
        })
    
    export_df = pd.DataFrame(export_rows)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.download_button(
            label="⬇️ Download VaR summary (CSV)",
            data=csv_bytes,
            file_name=f"var_summary_{'_'.join(selected_symbols)}_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_col2:
        returns_csv = returns_df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download returns data (CSV)",
            data=returns_csv,
            file_name=f"returns_{'_'.join(selected_symbols)}_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    
    # ─────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        VaR Calculator · Built with Streamlit + yfinance + Plotly<br>
        For educational and portfolio analysis purposes only. Not financial advice.
    </div>
    """, unsafe_allow_html=True)
    

def guide_page():
    st.title("📖 VaRify User Guide")
    st.markdown("#### Institutional-grade Value at Risk (VaR) & Portfolio Risk Analytics")
    
    st.info("👋 **Welcome!** VaRify helps you quantify downside risk in your portfolio using industry-standard statistical models. Read below to understand the metrics and methodologies before diving into the Risk Terminal.")
    
    st.markdown("---")
    
    st.subheader("📊 1. Core Risk Metrics Explained")
    tab1, tab2, tab3 = st.tabs(["Value at Risk (VaR)", "Expected Shortfall (CVaR)", "Portfolio Beta"])
    
    with tab1:
        st.markdown("**Value at Risk (VaR)** answers a simple question:")
        st.markdown("> *“What is the most I can expect to lose in a day with X% confidence?”*")
        st.markdown("""
        If your 1-day 95% VaR is **₹50,000**, it means that under normal market conditions, there is only a **5% chance** (or about 1 day out of every 20) that your portfolio will lose more than ₹50,000.
        """)
        
    with tab2:
        st.markdown("**Expected Shortfall (CVaR / Conditional VaR)** answers the follow-up question:")
        st.markdown("> *“If the worst-case scenario happens, how bad will it be?”*")
        st.markdown("""
        VaR tells you the threshold, but CVaR tells you the *average loss* beyond that threshold. It is a more robust metric for capturing extreme "tail risks" (Black Swan events).
        """)
        
    with tab3:
        st.markdown("**Beta** measures your portfolio's volatility relative to the broader market (e.g., Nifty 50 or S&P 500).")
        st.markdown("""
        - **Beta = 1.0**: Your portfolio moves in tandem with the market.
        - **Beta > 1.0**: Your portfolio is more volatile than the market (higher risk, higher potential reward).
        - **Beta < 1.0**: Your portfolio is less volatile than the market (defensive).
        """)

    st.markdown("---")
    
    st.subheader("🚦 2. The Traffic Light Risk System")
    st.markdown("VaRify automatically compares your portfolio's VaR against the benchmark index (Nifty 50 or S&P 500) to give you an intuitive visual risk assessment.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.error("**🔴 High Risk**\n\nYour Portfolio VaR is **> 1.5x** the Market VaR.")
    with col2:
        st.warning("**🟡 Moderate Risk**\n\nYour Portfolio VaR is **1.0x to 1.5x** the Market VaR.")
    with col3:
        st.success("**🟢 Low Risk**\n\nYour Portfolio VaR is **less than** the Market VaR.")

    st.markdown("---")
    
    st.subheader("🧮 3. VaR Methodologies")
    with st.expander("Explore the 3 Mathematical Models", expanded=False):
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown("#### 1. Parametric")
            st.markdown("Assumes returns follow a normal (bell-curve) distribution. Fast and standard, but may underestimate extreme risks if the asset has 'fat tails'.")
        with m_col2:
            st.markdown("#### 2. Historical")
            st.markdown("Uses actual historical daily returns to find the 5% worst days. Makes no assumptions about distribution shape, but relies entirely on past data.")
        with m_col3:
            st.markdown("#### 3. Monte Carlo")
            st.markdown("Simulates 10,000 possible future return paths based on historical mean and volatility. Excellent for visualizing a wide range of potential outcomes.")

    st.markdown("---")
    
    st.subheader("🛠️ 4. How to Use the Terminal")
    st.markdown("""
    1. **Navigate** to the **Risk Terminal** using the sidebar.
    2. **Select your Assets**: Choose your country (India or US) and select up to 5 tickers.
    3. **Configure Portfolio**: Enter your total position size and allocate percentage weights for each ticker (must sum to 100%).
    4. **Tune Parameters**: Adjust Confidence Level, Holding Period, and Historical Lookback.
    5. **Calculate**: Hit the `▶ Calculate VaR` button to generate your dashboard.
    """)
    
    st.info("💡 **Pro Tip**: Use the **Audit & Backtest Metrics** expander in the terminal to view the Kupiec POF test. This statistical test backtests the VaR model against historical data to tell you if the model is mathematically valid.")

pages = {
    "Start": [
        st.Page(guide_page, title="User Guide", icon="📖", default=True)
    ],
    "Tools": [
        st.Page(main_page, title="Risk Terminal", icon="📉")
    ]
}

pg = st.navigation(pages, position="sidebar")
pg.run()
