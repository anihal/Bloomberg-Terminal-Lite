"""
Bloomberg Lite Terminal - Streamlit UI entry point.
"""
import streamlit as st
import pandas as pd
import altair as alt
import os
import time

from data_provider import AlphaVantageClient, get_company_metadata
from indicators import add_all_indicators, calculate_1m_return, calculate_3m_return
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
    df = calculate_1m_return(df)
    df = calculate_3m_return(df)
    return df


if symbol:
    try:
        with st.spinner(f"Fetching data for {symbol}..."):
            df = fetch_and_process_data(symbol)
            time.sleep(1)  # Rate limit: 1 request per second for Alpha Vantage free tier
            metadata = get_company_metadata(symbol)

        # Company Identity Section
        st.markdown(f"### {metadata['Name']} ({metadata['Symbol']})")
        st.markdown(f"<p style='color: #888888; margin-top: -10px;'>{metadata['Sector']} | {metadata['Industry']}</p>", unsafe_allow_html=True)
        st.markdown("---")

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        price_change = latest["close"] - prev["close"]
        price_change_pct = (price_change / prev["close"]) * 100

        # Current Price and Volume - Big Metrics
        col1, col2, col3, col4 = st.columns(4)

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

        with col3:
            return_1m = latest.get("return_1m", None)
            if pd.notna(return_1m):
                color = "green" if return_1m >= 0 else "red"
                st.markdown(f"**1-Month Return**")
                st.markdown(f"<h2 style='color: {color}; margin-top: -10px;'>{return_1m:+.2f}%</h2>", unsafe_allow_html=True)
            else:
                st.metric(label="1-Month Return", value="N/A")

        with col4:
            return_3m = latest.get("return_3m", None)
            if pd.notna(return_3m):
                color = "green" if return_3m >= 0 else "red"
                st.markdown(f"**3-Month Return**")
                st.markdown(f"<h2 style='color: {color}; margin-top: -10px;'>{return_3m:+.2f}%</h2>", unsafe_allow_html=True)
            else:
                st.metric(label="3-Month Return", value="N/A")

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
        ma_col3.metric("SMA 100", format_currency(latest["sma_100"]))
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

        # Price & Volume Chart (Bloomberg-style dark theme)
        st.subheader("Price & Volume")
        price_vol_data = df[["close", "volume"]].tail(100).copy().reset_index()

        # Price chart - clean closing price line
        price_chart = alt.Chart(price_vol_data).mark_line(
            color="#FF6600",  # Bloomberg orange
            strokeWidth=2
        ).encode(
            x=alt.X("date:T", axis=alt.Axis(format="%b %d", title=None, labels=False)),
            y=alt.Y("close:Q", title="Price ($)", scale=alt.Scale(zero=False))
        ).properties(
            height=250
        )

        # Volume chart - bar chart below
        volume_chart = alt.Chart(price_vol_data).mark_bar(
            color="#FF6600",
            opacity=0.7
        ).encode(
            x=alt.X("date:T", axis=alt.Axis(format="%b %d", title="Date")),
            y=alt.Y("volume:Q", title="Volume")
        ).properties(
            height=100
        )

        # Stack price and volume charts vertically
        combined_chart = alt.vconcat(
            price_chart,
            volume_chart,
            spacing=0
        ).configure(
            background="#1a1a1a"
        ).configure_axis(
            labelColor="#cccccc",
            titleColor="#cccccc",
            gridColor="#333333",
            domainColor="#555555",
            tickColor="#555555"
        ).configure_view(
            strokeWidth=0
        )

        st.altair_chart(combined_chart, use_container_width=True)

        # RSI Chart
        st.subheader("RSI Chart")
        rsi_data = df[["rsi"]].tail(100).copy().reset_index()
        rsi_data = rsi_data.dropna(subset=["rsi"])

        rsi_line = alt.Chart(rsi_data).mark_line(color="#FF6600", strokeWidth=2).encode(
            x=alt.X("date:T", axis=alt.Axis(format="%b %d", title="Date")),
            y=alt.Y("rsi:Q", scale=alt.Scale(domain=[0, 100]), title="RSI")
        )

        overbought_line = alt.Chart(pd.DataFrame({"y": [70]})).mark_rule(
            strokeDash=[5, 5], color="#888888"
        ).encode(y="y:Q")

        oversold_line = alt.Chart(pd.DataFrame({"y": [30]})).mark_rule(
            strokeDash=[5, 5], color="#888888"
        ).encode(y="y:Q")

        rsi_chart = (rsi_line + overbought_line + oversold_line).properties(
            height=250
        ).configure(
            background="#1a1a1a"
        ).configure_axis(
            labelColor="#cccccc",
            titleColor="#cccccc",
            gridColor="#333333",
            domainColor="#555555",
            tickColor="#555555",
            labelFontSize=12,
            titleFontSize=14
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(rsi_chart, use_container_width=True)

        # MACD Chart
        st.subheader("MACD Chart")
        macd_data = df[["macd", "macd_signal"]].tail(100).copy().reset_index()
        macd_data = macd_data.dropna(subset=["macd", "macd_signal"])

        macd_melted = macd_data.melt(
            id_vars=["date"],
            value_vars=["macd", "macd_signal"],
            var_name="Indicator",
            value_name="Value"
        )
        macd_melted["Indicator"] = macd_melted["Indicator"].map({
            "macd": "MACD",
            "macd_signal": "Signal"
        })

        macd_chart = alt.Chart(macd_melted).mark_line().encode(
            x=alt.X("date:T", axis=alt.Axis(format="%b %d", title="Date")),
            y=alt.Y("Value:Q", title="MACD"),
            color=alt.Color(
                "Indicator:N",
                scale=alt.Scale(domain=["MACD", "Signal"], range=["#FF6600", "#00BFFF"]),
                legend=alt.Legend(title=None, orient="top", labelColor="#cccccc")
            )
        ).properties(
            height=250
        ).configure(
            background="#1a1a1a"
        ).configure_axis(
            labelColor="#cccccc",
            titleColor="#cccccc",
            gridColor="#333333",
            domainColor="#555555",
            tickColor="#555555",
            labelFontSize=12,
            titleFontSize=14
        ).configure_legend(
            labelFontSize=12,
            labelColor="#cccccc"
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(macd_chart, use_container_width=True)

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
