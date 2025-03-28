from data_collector import StockDataCollector
from models import Stock, DailyData, engine
from sqlalchemy.orm import sessionmaker
import yfinance as yf

# Initialize collector
collector = StockDataCollector()

# Try to get Block data directly
ticker = yf.Ticker("SQ")
info = ticker.info
print("Block (SQ) Stock Information:")
print(f"Name: {info.get('longName', 'N/A')}")
print(f"Current Price: ${info.get('currentPrice', 'N/A')}")
print(f"Market Cap: ${info.get('marketCap', 'N/A'):,}") 