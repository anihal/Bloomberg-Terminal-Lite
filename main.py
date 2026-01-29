"""
Bloomberg Terminal Lite - Streamlit UI entry point.
"""
import streamlit as st
import pandas as pd

from data_provider import AlphaVantageClient
from indicators import calculate_sma, calculate_ema, calculate_rsi
from utils import format_currency, format_large_number, format_percent


st.set_page_config(
    page_title="Bloomberg Terminal Lite",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Bloomberg Terminal Lite")


@st.cache_data(ttl=300)
def fetch_stock_data(symbol: str) -> pd.DataFrame:
    """Fetch and cache stock data."""
    client = AlphaVantageClient()
    return client.get_stock_data(symbol)


symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()

st.sidebar.markdown("---")
st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("SMA (20-day)", value=True)
show_ema = st.sidebar.checkbox("EMA (20-day)", value=False)
show_rsi = st.sidebar.checkbox("RSI (14-day)", value=False)

if st.sidebar.button("Fetch Data") or "data" not in st.session_state:
    try:
        with st.spinner(f"Fetching data for {symbol}..."):
            df = fetch_stock_data(symbol)
            st.session_state.data = df
            st.session_state.symbol = symbol
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        st.stop()

if "data" in st.session_state:
    df = st.session_state.data
    current_symbol = st.session_state.get("symbol", symbol)

    col1, col2, col3, col4 = st.columns(4)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    change = latest["close"] - prev["close"]
    change_pct = (change / prev["close"]) * 100

    col1.metric("Close", format_currency(latest["close"]), format_percent(change_pct))
    col2.metric("High", format_currency(latest["high"]))
    col3.metric("Low", format_currency(latest["low"]))
    col4.metric("Volume", format_large_number(latest["volume"]))

    st.subheader(f"{current_symbol} Price Chart")

    chart_data = df[["close"]].copy()
    chart_data.columns = ["Close"]

    if show_sma:
        chart_data["SMA 20"] = calculate_sma(df["close"], 20)
    if show_ema:
        chart_data["EMA 20"] = calculate_ema(df["close"], 20)

    st.line_chart(chart_data)

    if show_rsi:
        st.subheader("RSI (14-day)")
        rsi_data = calculate_rsi(df["close"], 14)
        st.line_chart(rsi_data)

    st.subheader("Recent Data")
    st.dataframe(df.tail(10).sort_index(ascending=False))
