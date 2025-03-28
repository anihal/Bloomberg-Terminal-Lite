import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from sqlalchemy.orm import sessionmaker
from models import Stock, DailyData, engine, Base

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_database():
    """Create the database tables."""
    Base.metadata.create_all(engine)
    logger.info("Database tables created successfully")

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return None
            
        # Reset index to make date a column
        df = df.reset_index()
        
        # Rename columns to match database schema
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def save_to_database(symbol, df):
    """Save the DataFrame to SQLite database using SQLAlchemy."""
    if df is None or df.empty:
        return
        
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Create or get stock record
        stock = session.query(Stock).filter_by(symbol=symbol).first()
        if not stock:
            stock = Stock(symbol=symbol)
            session.add(stock)
            session.commit()
        
        # Add daily data
        for _, row in df.iterrows():
            daily_data = DailyData(
                stock_id=stock.id,
                date=row['date'],
                open_price=row['open_price'],
                high_price=row['high_price'],
                low_price=row['low_price'],
                close_price=row['close_price'],
                volume=row['volume']
            )
            session.add(daily_data)
        
        session.commit()
        logger.info(f"Successfully saved data for {symbol}")
        
    except Exception as e:
        logger.error(f"Error saving to database: {str(e)}")
        session.rollback()
    finally:
        session.close()

def main():
    """Main function to fetch and store stock data."""
    # List of stocks to fetch
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM']
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3 years of data
    
    # Create database and tables
    create_database()
    
    # Fetch and save data for each stock
    for symbol in stocks:
        logger.info(f"Fetching data for {symbol}...")
        df = fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        save_to_database(symbol, df)

if __name__ == "__main__":
    main() 