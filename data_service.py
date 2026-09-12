import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import time
from functools import wraps

@st.cache_data(ttl=3600)
def load_stock_list(country: str) -> pd.DataFrame:
    base = os.path.dirname(__file__)
    if country == "India (NSE)":
        path = os.path.join(base, "data", "nifty500.csv")
    else:
        path = os.path.join(base, "data", "sp500.csv")
    return pd.read_csv(path)

def retry_yf(max_retries=3, backoff_factor=1.5):
    """Decorator to retry yfinance network calls on failure with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            delay = 1.0
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    time.sleep(delay)
                    delay *= backoff_factor
            # If it fails completely, return empty DataFrame or dict based on hint
            # A bit hacky, but safe for Streamlit
            if "dict" in str(func.__annotations__.get("return", "")):
                return {}
            return pd.DataFrame()
        return wrapper
    return decorator

@st.cache_data(ttl=1800, show_spinner=False)
@retry_yf(max_retries=3)
def fetch_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for given tickers."""
    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()
        
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        elif "Close" in data.columns.get_level_values(1):
            close = data.xs("Close", level=1, axis=1)
        else:
            close = data
    else:
        close = data[["Close"]] if "Close" in data.columns else data
    close = close.dropna(how="all")
    return close

def format_currency(value: float, currency: str) -> str:
    """Format large numbers with Indian (Lakhs/Crores) or US numbering."""
    is_negative = value < 0
    val_abs = abs(value)
    
    if currency == "₹":
        s = f"{val_abs:,.0f}"
        if "," in s:
            parts = s.split(",")
            last_part = parts[-1]
            rest = "".join(parts[:-1])
            formatted_rest = ",".join([rest[max(0, i-2):i] for i in range(len(rest), 0, -2)][::-1]) if rest else ""
            formatted = f"{formatted_rest},{last_part}" if formatted_rest else last_part
        else:
            formatted = s
        res = f"{currency}{formatted}"
    else:
        res = f"{currency}{val_abs:,.0f}"
        
    return f"-{res}" if is_negative else res

def _extract_series(data: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    """Safely extract a column from yfinance data (handles MultiIndex)."""
    if isinstance(data.columns, pd.MultiIndex):
        try:
            return data[(field, ticker)].dropna()
        except KeyError:
            sub = data[field]
            return sub.iloc[:, 0].dropna() if isinstance(sub, pd.DataFrame) else sub.dropna()
    if field in data.columns:
        return data[field].dropna()
    return pd.Series(dtype=float)

@st.cache_data(ttl=300, show_spinner=False)
@retry_yf(max_retries=3)
def fetch_live_quotes(tickers_yf: tuple, display_symbols: tuple) -> dict:
    """Fetch latest price, change, and volume for given tickers (cached 5 min)."""
    quotes = {}
    try:
        data = yf.download(
            list(tickers_yf), period="5d", progress=False,
            auto_adjust=True, threads=True,
        )
        if data.empty:
            return quotes
        for yf_sym, disp in zip(tickers_yf, display_symbols):
            try:
                c = _extract_series(data, "Close", yf_sym)
                h = _extract_series(data, "High", yf_sym)
                lo = _extract_series(data, "Low", yf_sym)
                v = _extract_series(data, "Volume", yf_sym)
                if len(c) < 2:
                    continue
                price = float(c.iloc[-1])
                prev = float(c.iloc[-2])
                quotes[disp] = {
                    "price": price,
                    "change": price - prev,
                    "change_pct": (price - prev) / prev * 100,
                    "day_high": float(h.iloc[-1]) if len(h) > 0 else price,
                    "day_low": float(lo.iloc[-1]) if len(lo) > 0 else price,
                    "volume": int(v.iloc[-1]) if len(v) > 0 else 0,
                    "prev_close": prev,
                }
            except Exception:
                continue
    except Exception:
        pass
    return quotes

@st.cache_data(ttl=300, show_spinner=False)
@retry_yf(max_retries=3)
def fetch_market_indices(country: str) -> dict:
    """Fetch key market index levels (cached 5 min)."""
    idx_map = (
        {"^NSEI": "NIFTY 50", "^NSEBANK": "BANK NIFTY", "^BSESN": "SENSEX"}
        if country == "India (NSE)"
        else {"^GSPC": "S&P 500", "^DJI": "DOW JONES", "^IXIC": "NASDAQ"}
    )
    results = {}
    try:
        data = yf.download(
            list(idx_map.keys()), period="5d", progress=False,
            auto_adjust=True, threads=True,
        )
        if data.empty:
            return results
        for sym, name in idx_map.items():
            try:
                c = _extract_series(data, "Close", sym)
                if len(c) < 2:
                    continue
                price = float(c.iloc[-1])
                prev = float(c.iloc[-2])
                results[name] = {
                    "price": price,
                    "change": price - prev,
                    "change_pct": (price - prev) / prev * 100,
                }
            except Exception:
                continue
    except Exception:
        pass
    return results

@st.cache_data(ttl=300, show_spinner=False)
@retry_yf(max_retries=3)
def fetch_intraday(tickers_yf: tuple, display_symbols: tuple) -> pd.DataFrame:
    """Fetch intraday data — 15-min intervals, last 5 trading days (cached 5 min)."""
    try:
        data = yf.download(
            list(tickers_yf), period="5d", interval="15m",
            progress=False, auto_adjust=True, threads=True,
        )
        if data.empty:
            return pd.DataFrame()
        cols = {}
        for yf_sym, disp in zip(tickers_yf, display_symbols):
            s = _extract_series(data, "Close", yf_sym)
            if not s.empty:
                cols[disp] = s
        return pd.DataFrame(cols).dropna(how="all") if cols else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def format_volume(vol: int) -> str:
    """Format volume with K/M/B suffix."""
    if vol >= 1_000_000_000:
        return f"{vol / 1e9:.1f}B"
    if vol >= 1_000_000:
        return f"{vol / 1e6:.1f}M"
    if vol >= 1_000:
        return f"{vol / 1e3:.1f}K"
    return str(vol)
