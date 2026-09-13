import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
from data_service import (
    load_stock_list,
    retry_yf,
    fetch_prices,
    format_currency,
    _extract_series,
    fetch_live_quotes,
    fetch_market_indices,
    fetch_intraday,
    format_volume
)

def test_load_stock_list_india():
    with patch("pandas.read_csv") as mock_read:
        mock_df = pd.DataFrame({"Symbol": ["RELIANCE", "TCS"]})
        mock_read.return_value = mock_df
        df = load_stock_list("India (NSE)")
        assert not df.empty
        assert "RELIANCE" in df["Symbol"].values

def test_load_stock_list_us():
    with patch("pandas.read_csv") as mock_read:
        mock_df = pd.DataFrame({"Symbol": ["AAPL", "MSFT"]})
        mock_read.return_value = mock_df
        df = load_stock_list("US")
        assert not df.empty

def test_format_currency_inr_lakhs():
    assert format_currency(250000, "₹") == "₹2,50,000"
    
def test_format_currency_usd():
    assert format_currency(1500000, "$") == "$1,500,000"
    
def test_format_currency_negative():
    assert format_currency(-5000, "$") == "-$5,000"

def test_format_volume_thousands():
    assert format_volume(5400) == "5.4K"
    
def test_format_volume_millions():
    assert format_volume(2300000) == "2.3M"
    
def test_format_volume_billions():
    assert format_volume(1500000000) == "1.5B"
    
def test_format_volume_small():
    assert format_volume(500) == "500"

@patch("yfinance.download")
def test_fetch_prices_empty(mock_yf):
    # Use different dates to avoid streamlit caching across tests
    import streamlit as st
    st.cache_data.clear()
    mock_yf.return_value = pd.DataFrame()
    res = fetch_prices(("AAPL",), "2023-01-01", "2023-01-02")
    assert res.empty

@patch("yfinance.download")
def test_fetch_prices_single(mock_yf):
    import streamlit as st
    st.cache_data.clear()
    df = pd.DataFrame({"Close": [100, 101, 102]})
    mock_yf.return_value = df
    res = fetch_prices(("MSFT",), "2023-01-01", "2023-01-10")
    assert not res.empty
    assert res.iloc[0]["Close"] == 100

def test_extract_series_simple():
    df = pd.DataFrame({"Close": [100, 101]})
    s = _extract_series(df, "Close", "AAPL")
    assert not s.empty
    assert len(s) == 2

def test_extract_series_multiindex():
    cols = pd.MultiIndex.from_product([["Close", "Volume"], ["AAPL", "MSFT"]])
    df = pd.DataFrame([[100, 200, 1000, 2000]], columns=cols)
    s = _extract_series(df, "Close", "AAPL")
    assert s.iloc[0] == 100

@patch("yfinance.download")
def test_fetch_live_quotes(mock_yf):
    cols = pd.MultiIndex.from_product([["Close", "High", "Low", "Volume"], ["AAPL.NS"]])
    df = pd.DataFrame([
        [100, 105, 95, 1000],
        [102, 106, 100, 1500]
    ], columns=cols)
    mock_yf.return_value = df
    
    quotes = fetch_live_quotes(("AAPL.NS",), ("Apple",))
    assert "Apple" in quotes
    assert quotes["Apple"]["price"] == 102

@patch("yfinance.download")
def test_fetch_market_indices(mock_yf):
    cols = pd.MultiIndex.from_product([["Close"], ["^NSEI"]])
    df = pd.DataFrame([[10000], [10100]], columns=cols)
    mock_yf.return_value = df
    
    res = fetch_market_indices("India (NSE)")
    assert "NIFTY 50" in res

@patch("yfinance.download")
def test_fetch_intraday(mock_yf):
    cols = pd.MultiIndex.from_product([["Close"], ["AAPL"]])
    df = pd.DataFrame([[100], [101]], columns=cols)
    mock_yf.return_value = df
    
    res = fetch_intraday(("AAPL",), ("Apple",))
    assert not res.empty