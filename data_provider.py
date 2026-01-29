"""
Alpha Vantage data provider for stock market data.
"""
import requests
import pandas as pd
from typing import Optional

from config import ALPHA_VANTAGE_KEY


class AlphaVantageClient:
    """Client for fetching stock data from Alpha Vantage API."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Alpha Vantage client.

        Args:
            api_key: Alpha Vantage API key. If not provided, uses ALPHA_VANTAGE_KEY from config.
        """
        self.api_key = api_key or ALPHA_VANTAGE_KEY
        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key not found. Please set ALPHA_VANTAGE_KEY in your .env file. "
                "Get your free API key from: https://www.alphavantage.co/support/#api-key"
            )

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch daily time series data for a stock symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')

        Returns:
            pandas DataFrame with columns: open, high, low, close, volume
            Index is the date (datetime).

        Raises:
            ValueError: If the API returns an error or no data is found.
            requests.RequestException: If the API request fails.
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol.upper(),
            "apikey": self.api_key,
            "outputsize": "full"
        }

        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")

        if "Note" in data:
            raise ValueError(f"API Rate Limit: {data['Note']}")

        time_series = data.get("Time Series (Daily)")
        if not time_series:
            raise ValueError(f"No data found for symbol: {symbol}")

        df = pd.DataFrame.from_dict(time_series, orient="index")

        df.columns = ["open", "high", "low", "close", "volume"]

        df = df.astype({
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": int
        })

        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        df = df.sort_index()

        return df
