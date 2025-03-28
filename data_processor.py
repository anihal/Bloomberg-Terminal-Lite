import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from loguru import logger
from typing import List, Optional
from pathlib import Path

from models import Stock, DailyData, engine
from config import PROCESSED_DATA_DIR

class StockDataProcessor:
    def __init__(self):
        self.Session = sessionmaker(bind=engine)
        self.processed_data_dir = Path(PROCESSED_DATA_DIR)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def load_stock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Load stock data from database into DataFrame.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with stock data or None if error occurs
        """
        session = self.Session()
        try:
            # Build query
            query = session.query(DailyData, Stock).join(Stock).filter(Stock.symbol == symbol)
            
            # Add date filters if provided
            if start_date:
                query = query.filter(DailyData.date >= datetime.strptime(start_date, '%Y-%m-%d'))
            if end_date:
                query = query.filter(DailyData.date <= datetime.strptime(end_date, '%Y-%m-%d'))
            
            data = query.all()
            if not data:
                logger.error(f"No data found for {symbol} in the specified date range")
                return None

            # Convert to DataFrame
            df = pd.DataFrame([{
                'date': d[0].date,
                'open': d[0].open_price,
                'high': d[0].high_price,
                'low': d[0].low_price,
                'close': d[0].close_price,
                'volume': d[0].volume
            } for d in data])

            # Sort by date and set index
            df = df.sort_values('date')
            
            # Handle duplicate dates by keeping the latest entry
            df = df.drop_duplicates(subset=['date'], keep='last')
            
            df.set_index('date', inplace=True)
            
            # Ensure the index is sorted
            df = df.sort_index()
            
            return df

        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            return None
        finally:
            session.close()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        if df is None or df.empty:
            logger.error("No data to clean")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        df = df.copy()
        
        logger.info(f"Starting with {len(df)} data points")

        # Handle missing values using newer pandas methods
        df = df.ffill().bfill()
        
        # Remove extreme outliers using Z-score method (more lenient threshold)
        for col in ['open', 'high', 'low', 'close']:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask = z_scores < 5  # Increased threshold from 3 to 5
            if (~mask).any():
                logger.warning(f"Found {(~mask).sum()} outliers in {col}")
            df = df[mask]
            
        logger.info(f"After outlier removal: {len(df)} data points")

        # Ensure logical price relationships
        df.loc[:, 'high'] = df[['high', 'open', 'close']].max(axis=1)
        df.loc[:, 'low'] = df[['low', 'open', 'close']].min(axis=1)

        # Validate price relationships
        invalid_prices = (df['high'] < df['low']) | (df['open'] > df['high']) | (df['open'] < df['low']) | \
                        (df['close'] > df['high']) | (df['close'] < df['low'])
        
        if invalid_prices.any():
            logger.warning(f"Found {invalid_prices.sum()} rows with invalid price relationships")
            df = df[~invalid_prices]
            
        logger.info(f"After price validation: {len(df)} data points")

        # Remove any remaining missing values
        df = df.dropna()
        
        logger.info(f"After removing missing values: {len(df)} data points")

        # Ensure no duplicate indices
        if df.index.duplicated().any():
            logger.warning(f"Found {df.index.duplicated().sum()} duplicate dates")
            df = df[~df.index.duplicated(keep='last')]
            
        logger.info(f"Final data points: {len(df)}")

        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        if df is None or df.empty:
            logger.error("No data to calculate indicators")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        df = df.copy()

        # Moving averages with min_periods=1 to handle initial periods
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        df['sma_200'] = df['close'].rolling(window=200, min_periods=1).mean()

        # Exponential moving averages
        df['ema_20'] = df['close'].ewm(span=20, adjust=False, min_periods=1).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False, min_periods=1).mean()

        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        exp1 = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
        exp2 = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['close'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Price momentum
        df['momentum'] = df['close'].pct_change(periods=10)
        df['price_change'] = df['close'].pct_change()

        # Validate indicators
        if df.isnull().any().any():
            logger.warning("Missing values found in technical indicators")
            df = df.ffill().bfill()  # Fill any remaining NAs

        return df

    def process_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Process stock data through the entire pipeline."""
        logger.info(f"Processing data for {symbol}")

        # Load data
        df = self.load_stock_data(symbol)
        if df is None:
            return None

        # Clean data
        df = self.clean_data(df)
        if df.empty:
            logger.error(f"No valid data after cleaning for {symbol}")
            return None

        # Calculate indicators
        df = self.calculate_technical_indicators(df)

        # Final validation
        if df.isnull().any().any():
            logger.warning(f"Missing values found in processed data for {symbol}")
            return None

        # Save processed data
        output_file = self.processed_data_dir / f"{symbol}_processed.parquet"
        df.to_parquet(output_file)
        logger.info(f"Saved processed data for {symbol} to {output_file}")

        return df

    def process_multiple_stocks(self, symbols: List[str]) -> None:
        """Process multiple stocks."""
        for symbol in symbols:
            self.process_stock_data(symbol)

if __name__ == "__main__":
    # Example usage
    processor = StockDataProcessor()
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "ABNB", "COIN", "UBER"]
    processor.process_multiple_stocks(test_symbols) 