from data_collector import StockDataCollector
from models import Stock, DailyData, engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import pandas as pd
import time

def display_stock_summary(session, symbol):
    """Display summary of collected data for a stock."""
    stock = session.query(Stock).filter_by(symbol=symbol).first()
    if stock:
        data = session.query(DailyData).filter_by(stock_id=stock.id).all()
        if data:
            print(f"\nSummary for {symbol}:")
            print(f"Total days of data: {len(data)}")
            print(f"Date range: {data[0].date.date()} to {data[-1].date.date()}")
            
            # Calculate some basic statistics
            prices = pd.DataFrame([{
                'date': d.date,
                'close': d.close_price,
                'volume': d.volume
            } for d in data])
            
            print("\nPrice Statistics:")
            print(f"Latest price: ${data[-1].close_price:.2f}")
            print(f"Highest price: ${prices['close'].max():.2f}")
            print(f"Lowest price: ${prices['close'].min():.2f}")
            print(f"Average price: ${prices['close'].mean():.2f}")
            print(f"Average daily volume: {prices['volume'].mean():,.0f}")
            print("-" * 50)

def main():
    # Initialize the collector
    collector = StockDataCollector()
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months of data
    
    print(f"Collecting data from {start_date.date()} to {end_date.date()}")
    print("This may take a few minutes...\n")
    
    # List of stocks to collect
    symbols = [
        "AAPL",     # Apple
        "MSFT",     # Microsoft
        "GOOGL",    # Google
        "AMZN",     # Amazon
        "NVDA",     # NVIDIA
        "META",     # Meta
        "TSLA",     # Tesla
        "JPM",      # JPMorgan Chase
        "V",        # Visa
        "WMT",      # Walmart
        "PG",       # Procter & Gamble
        "JNJ",      # Johnson & Johnson
        "MA"        # Mastercard
    ]
    
    # Create a session for displaying summaries
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Collect data for each symbol
    for symbol in symbols:
        # Collect data
        success = collector.collect_stock_data(symbol, start_date, end_date)
        
        # Display summary statistics if collection was successful
        if success:
            display_stock_summary(session, symbol)
        
        # Respect rate limiting (2 requests per second)
        time.sleep(0.5)
    
    session.close()

if __name__ == "__main__":
    main() 