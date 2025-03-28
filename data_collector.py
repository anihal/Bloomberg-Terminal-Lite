import yfinance as yf
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from loguru import logger
import time
from typing import Optional, List
import requests

from config import (
    YFINANCE_RATE_LIMIT,
    MAX_RETRIES,
    RETRY_DELAY,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE
)
from models import Stock, DailyData, engine

# Configure logger
logger.add("data/app.log", rotation="500 MB")

class StockDataCollector:
    def __init__(self):
        self.Session = sessionmaker(bind=engine)
        self.rate_limit = YFINANCE_RATE_LIMIT
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY
        self.session = None
        self.logger = logger

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if the symbol exists and is valid."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return bool(info.get('regularMarketPrice'))
        except Exception as e:
            self.logger.error(f"Error validating symbol {symbol}: {str(e)}")
            return False

    def fetch_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch stock data from Yahoo Finance."""
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if df.empty:
                    self.logger.warning(f"No data found for symbol {symbol}")
                    return None
                
                # Reset index to make date a column
                df = df.reset_index()
                
                # Rename columns to match our database schema
                df = df.rename(columns={
                    'Date': 'date',
                    'Open': 'open_price',
                    'High': 'high_price',
                    'Low': 'low_price',
                    'Close': 'close_price',
                    'Volume': 'volume'
                })
                
                return df
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue
        
        return None

    def save_to_database(self, symbol: str, df: pd.DataFrame) -> bool:
        """Save stock data to database."""
        session = self.Session()
        try:
            # Get or create stock record
            stock = session.query(Stock).filter_by(symbol=symbol).first()
            if not stock:
                stock = Stock(symbol=symbol)
                session.add(stock)
                session.commit()

            # Convert DataFrame to list of DailyData objects
            daily_data = []
            for _, row in df.iterrows():
                daily_data.append(DailyData(
                    stock_id=stock.id,
                    date=row['date'],
                    open_price=row['open_price'],
                    high_price=row['high_price'],
                    low_price=row['low_price'],
                    close_price=row['close_price'],
                    volume=row['volume']
                ))

            # Bulk insert daily data
            session.bulk_save_objects(daily_data)
            session.commit()
            logger.info(f"Successfully saved data for {symbol}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving data for {symbol}: {str(e)}")
            return False
        finally:
            session.close()

    def collect_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> bool:
        """Collect stock data for a given symbol and date range."""
        self.logger.info(f"Processing symbol: {symbol}")
        
        if not self.validate_symbol(symbol):
            self.logger.error(f"Invalid symbol: {symbol}")
            return False
            
        df = self.fetch_stock_data(symbol, start_date, end_date)
        if df is not None:
            return self.save_to_database(symbol, df)
        return False

if __name__ == "__main__":
    # Example usage
    collector = StockDataCollector()
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    collector.collect_stock_data(test_symbols) 