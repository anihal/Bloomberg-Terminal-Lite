"""
Bloomberg Lite Terminal - Streamlit UI entry point.
"""
import streamlit as st
import pandas as pd
import os

from data_provider import AlphaVantageClient
from indicators import add_all_indicators
from utils import format_currency, format_large_number, format_percent

# Check if API key is configured
if not os.getenv("ALPHA_VANTAGE_KEY"):
    st.error("⚠️ ALPHA_VANTAGE_KEY environment variable is not set!")
    st.info("Please set it in your Render dashboard under Environment variables.")
    st.stop()


st.set_page_config(
    page_title="Bloomberg Lite Terminal",
    page_icon="📈",
    layout="wide"
)

st.title("Bloomberg Lite Terminal")

symbol = st.text_input("Enter Stock Ticker", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL")
symbol = symbol.upper().strip()


@st.cache_data(ttl=3600)  # Cache for 1 hour to reduce API calls
def fetch_and_process_data(ticker: str) -> pd.DataFrame:
    """Fetch stock data and calculate all indicators."""
    client = AlphaVantageClient()
    df = client.get_stock_data(ticker)
    df = add_all_indicators(df)
    return df


if symbol:
    try:
        with st.spinner(f"Fetching data for {symbol}..."):
            df = fetch_and_process_data(symbol)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        price_change = latest["close"] - prev["close"]
        price_change_pct = (price_change / prev["close"]) * 100

        # Current Price and Volume - Big Metrics
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Current Price",
                value=format_currency(latest["close"]),
                delta=f"{format_percent(price_change_pct)}"
            )

        with col2:
            vol_change = latest["volume"] - prev["volume"]
            vol_change_pct = (vol_change / prev["volume"]) * 100
            st.metric(
                label="Volume",
                value=format_large_number(latest["volume"]),
                delta=f"{format_percent(vol_change_pct)}"
            )

        st.markdown("---")

        # Price Information
        st.subheader("Price Information")
        price_col1, price_col2, price_col3, price_col4 = st.columns(4)
        price_col1.metric("Open", format_currency(latest["open"]))
        price_col2.metric("High", format_currency(latest["high"]))
        price_col3.metric("Low", format_currency(latest["low"]))
        price_col4.metric("Close", format_currency(latest["close"]))

        st.markdown("---")

        # Technical Indicators Dashboard
        st.subheader("Technical Indicators")

        # Moving Averages
        st.markdown("#### Moving Averages")
        ma_col1, ma_col2, ma_col3, ma_col4, ma_col5 = st.columns(5)
        ma_col1.metric("SMA 20", format_currency(latest["sma_20"]))
        ma_col2.metric("SMA 50", format_currency(latest["sma_50"]))
        ma_col3.metric("SMA 200", format_currency(latest["sma_200"]))
        ma_col4.metric("EMA 20", format_currency(latest["ema_20"]))
        ma_col5.metric("EMA 50", format_currency(latest["ema_50"]))

        # RSI and MACD
        st.markdown("#### Momentum Indicators")
        mom_col1, mom_col2, mom_col3, mom_col4 = st.columns(4)

        rsi_value = latest["rsi"]
        if rsi_value > 70:
            rsi_status = "Overbought"
        elif rsi_value < 30:
            rsi_status = "Oversold"
        else:
            rsi_status = "Neutral"
        mom_col1.metric("RSI (14)", f"{rsi_value:.2f}", rsi_status)

        mom_col2.metric("MACD", f"{latest['macd']:.4f}")
        mom_col3.metric("MACD Signal", f"{latest['macd_signal']:.4f}")
        mom_col4.metric("MACD Histogram", f"{latest['macd_histogram']:.4f}")

        # Bollinger Bands
        st.markdown("#### Bollinger Bands (20-day)")
        bb_col1, bb_col2, bb_col3, bb_col4 = st.columns(4)
        bb_col1.metric("Upper Band", format_currency(latest["bb_upper"]))
        bb_col2.metric("Middle Band", format_currency(latest["bb_middle"]))
        bb_col3.metric("Lower Band", format_currency(latest["bb_lower"]))
        bb_col4.metric("Band Width", f"{latest['bb_width']:.4f}")

        # Price Momentum
        st.markdown("#### Price Momentum")
        pm_col1, pm_col2, pm_col3, pm_col4 = st.columns(4)
        pm_col1.metric("Momentum (10)", format_currency(latest["momentum_10"]))
        pm_col2.metric("Momentum (20)", format_currency(latest["momentum_20"]))
        pm_col3.metric("ROC (10)", format_percent(latest["roc_10"]))
        pm_col4.metric("ROC (20)", format_percent(latest["roc_20"]))

        # Volume Indicators
        st.markdown("#### Volume Indicators")
        vol_col1, vol_col2, vol_col3, vol_col4 = st.columns(4)
        vol_col1.metric("Volume SMA (20)", format_large_number(latest["volume_sma_20"]))
        vol_col2.metric("Volume Ratio", f"{latest['volume_ratio']:.2f}x")
        vol_col3.metric("OBV", format_large_number(latest["obv"]))
        vol_col4.metric("VWAP", format_currency(latest["vwap"]))

        st.markdown("---")

        # Price Chart with Moving Averages
        st.subheader("Price Chart")
        chart_data = df[["close", "sma_20", "sma_50", "bb_upper", "bb_lower"]].tail(100)
        chart_data.columns = ["Close", "SMA 20", "SMA 50", "BB Upper", "BB Lower"]
        st.line_chart(chart_data)

        # RSI Chart
        st.subheader("RSI Chart")
        rsi_chart = df[["rsi"]].tail(100)
        rsi_chart.columns = ["RSI"]
        st.line_chart(rsi_chart)

        # MACD Chart
        st.subheader("MACD Chart")
        macd_chart = df[["macd", "macd_signal"]].tail(100)
        macd_chart.columns = ["MACD", "Signal"]
        st.line_chart(macd_chart)

        st.markdown("---")

        # Raw Data Table
        st.subheader("Recent Data")
        display_cols = ["open", "high", "low", "close", "volume", "sma_20", "ema_20", "rsi", "macd"]
        st.dataframe(
            df[display_cols].tail(10).sort_index(ascending=False).style.format({
                "open": "${:.2f}",
                "high": "${:.2f}",
                "low": "${:.2f}",
                "close": "${:.2f}",
                "volume": "{:,.0f}",
                "sma_20": "${:.2f}",
                "ema_20": "${:.2f}",
                "rsi": "{:.2f}",
                "macd": "{:.4f}"
            }),
            use_container_width=True
        )

    except ValueError as e:
        st.error(str(e))
        if "Rate Limit" in str(e):
            st.info("💡 Tip: Alpha Vantage free tier allows 5 requests/minute and 500/day. Please wait a minute and try again.")
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
