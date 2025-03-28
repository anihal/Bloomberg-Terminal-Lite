import pytest
import pandas as pd
from datetime import datetime, timedelta
from data_collector import StockDataCollector
from models import Stock, DailyData, engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def collector():
    return StockDataCollector()

@pytest.fixture
def session():
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_validate_symbol(collector):
    # Test valid symbol
    assert collector.validate_symbol("AAPL") is True
    
    # Test invalid symbol
    assert collector.validate_symbol("INVALID_SYMBOL") is False

def test_fetch_stock_data(collector):
    # Test fetching data for a valid symbol
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    df = collector.fetch_stock_data("AAPL", start_date, end_date)
    assert df is not None
    assert not df.empty
    assert all(col in df.columns for col in ['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'])

def test_save_to_database(collector, session):
    # Create test data as DataFrame
    test_data = pd.DataFrame({
        'date': [datetime.now()],
        'open_price': [150.0],
        'high_price': [155.0],
        'low_price': [149.0],
        'close_price': [153.0],
        'volume': [1000000]
    })
    
    # Test saving data
    success = collector.save_to_database("TEST", test_data)
    assert success is True
    
    # Verify data was saved
    stock = session.query(Stock).filter_by(symbol="TEST").first()
    assert stock is not None
    
    daily_data = session.query(DailyData).filter_by(stock_id=stock.id).first()
    assert daily_data is not None
    assert daily_data.close_price == 153.0

def test_collect_stock_data(collector, session):
    # Test collecting data for multiple symbols
    symbols = ["AAPL", "MSFT"]
    collector.collect_stock_data(symbols)
    
    # Verify data was collected
    for symbol in symbols:
        stock = session.query(Stock).filter_by(symbol=symbol).first()
        assert stock is not None
        daily_data = session.query(DailyData).filter_by(stock_id=stock.id).first()
        assert daily_data is not None 