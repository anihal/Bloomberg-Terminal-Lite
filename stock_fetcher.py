"""
Stock data fetcher using Finnhub for current prices and yfinance for company metrics.
"""
import finnhub
import yfinance as yf
from typing import Dict, Optional
from loguru import logger
from config import FINNHUB_API_KEY, MAX_RETRIES, RETRY_DELAY
import time


class StockDataFetcher:
    """Fetches stock data from Finnhub and yfinance APIs."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the stock data fetcher.

        Args:
            api_key: Finnhub API key. If not provided, uses FINNHUB_API_KEY from config.
        """
        self.api_key = api_key or FINNHUB_API_KEY
        if not self.api_key:
            raise ValueError(
                "Finnhub API key not found. Please set FINNHUB_API_KEY in your .env file. "
                "Get your free API key from: https://finnhub.io/register"
            )
        self.finnhub_client = finnhub.Client(api_key=self.api_key)
        logger.info("StockDataFetcher initialized successfully")

    def get_current_price(self, ticker: str) -> Optional[Dict]:
        """
        Fetch current price from Finnhub Quote API.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            Dictionary containing current price data or None if failed.
            Keys: current_price, high, low, open, previous_close, change, percent_change
        """
        for attempt in range(MAX_RETRIES):
            try:
                quote = self.finnhub_client.quote(ticker)

                if not quote or quote.get('c') == 0:
                    logger.warning(f"No quote data found for {ticker}")
                    return None

                return {
                    'current_price': quote.get('c'),  # Current price
                    'high': quote.get('h'),  # High price of the day
                    'low': quote.get('l'),  # Low price of the day
                    'open': quote.get('o'),  # Open price of the day
                    'previous_close': quote.get('pc'),  # Previous close price
                    'change': round(quote.get('c', 0) - quote.get('pc', 0), 2),
                    'percent_change': round(((quote.get('c', 0) - quote.get('pc', 0)) / quote.get('pc', 1)) * 100, 2)
                }
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {ticker}: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Failed to fetch quote for {ticker} after {MAX_RETRIES} attempts")
                    return None

    def get_company_metrics(self, ticker: str) -> Optional[Dict]:
        """
        Fetch company metrics from yfinance.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            Dictionary containing company metrics or None if failed.
            Keys: pe_ratio, market_cap, sector, industry, company_name, forward_pe, peg_ratio
        """
        for attempt in range(MAX_RETRIES):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                if not info or 'symbol' not in info:
                    logger.warning(f"No company info found for {ticker}")
                    return None

                return {
                    'company_name': info.get('longName', info.get('shortName', ticker)),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'peg_ratio': info.get('pegRatio'),
                    'dividend_yield': info.get('dividendYield'),
                    'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                    'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                    'average_volume': info.get('averageVolume'),
                    'shares_outstanding': info.get('sharesOutstanding')
                }
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {ticker} metrics: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Failed to fetch metrics for {ticker} after {MAX_RETRIES} attempts")
                    return None


def get_stock_data(ticker: str, api_key: Optional[str] = None) -> Optional[Dict]:
    """
    Fetch complete stock data combining Finnhub current price and yfinance company metrics.

    This function executes only when called (no background loop).

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        api_key: Optional Finnhub API key. If not provided, uses FINNHUB_API_KEY from .env

    Returns:
        Dictionary containing combined stock data or None if failed.
        Structure:
        {
            'ticker': str,
            'current_price': float,
            'high': float,
            'low': float,
            'open': float,
            'previous_close': float,
            'change': float,
            'percent_change': float,
            'company_name': str,
            'sector': str,
            'industry': str,
            'market_cap': int,
            'pe_ratio': float,
            'forward_pe': float,
            'peg_ratio': float,
            'dividend_yield': float,
            'fifty_two_week_high': float,
            'fifty_two_week_low': float,
            'average_volume': int,
            'shares_outstanding': int
        }

    Example:
        >>> data = get_stock_data('AAPL')
        >>> if data:
        >>>     print(f"{data['company_name']}: ${data['current_price']}")
        >>>     print(f"P/E Ratio: {data['pe_ratio']}")
        >>>     print(f"Market Cap: ${data['market_cap']:,}")
    """
    ticker = ticker.upper().strip()
    logger.info(f"Fetching stock data for {ticker}")

    try:
        fetcher = StockDataFetcher(api_key=api_key)

        # Fetch current price from Finnhub
        price_data = fetcher.get_current_price(ticker)
        if not price_data:
            logger.error(f"Failed to fetch price data for {ticker}")
            return None

        # Fetch company metrics from yfinance
        metrics_data = fetcher.get_company_metrics(ticker)
        if not metrics_data:
            logger.error(f"Failed to fetch company metrics for {ticker}")
            return None

        # Combine the data
        combined_data = {
            'ticker': ticker,
            **price_data,
            **metrics_data
        }

        logger.info(f"Successfully fetched complete data for {ticker}")
        return combined_data

    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
        return None
