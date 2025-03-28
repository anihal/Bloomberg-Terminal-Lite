from data_processor import StockDataProcessor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def display_processing_summary(df: pd.DataFrame, symbol: str):
    """Display summary of processed data."""
    print(f"\nProcessing Summary for {symbol}:")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Total days: {len(df)}")
    
    print("\nTechnical Indicators Summary:")
    print(f"Latest RSI: {df['rsi'].iloc[-1]:.2f}")
    print(f"Latest MACD: {df['macd'].iloc[-1]:.2f}")
    print(f"Latest Momentum: {df['momentum'].iloc[-1]:.2%}")
    
    print("\nPrice Statistics:")
    print(f"Latest price: ${df['close'].iloc[-1]:.2f}")
    print(f"20-day SMA: ${df['sma_20'].iloc[-1]:.2f}")
    print(f"50-day SMA: ${df['sma_50'].iloc[-1]:.2f}")
    print(f"200-day SMA: ${df['sma_200'].iloc[-1]:.2f}")
    
    print("\nVolume Statistics:")
    print(f"Latest volume: {df['volume'].iloc[-1]:,.0f}")
    print(f"Volume ratio: {df['volume_ratio'].iloc[-1]:.2f}")
    print("-" * 50)

def verify_data_quality(df: pd.DataFrame) -> bool:
    """Verify the quality of processed data."""
    # Check for missing values
    if df.isnull().any().any():
        print("Warning: Missing values found in processed data")
        return False
    
    # Check for infinite values
    if np.isinf(df).any().any():
        print("Warning: Infinite values found in processed data")
        return False
    
    # Check for negative prices
    if (df[['open', 'high', 'low', 'close']] < 0).any().any():
        print("Warning: Negative prices found in processed data")
        return False
    
    # Check for logical price relationships
    if not (df['high'] >= df['open']).all() or not (df['high'] >= df['close']).all():
        print("Warning: High price is not highest price")
        return False
    
    if not (df['low'] <= df['open']).all() or not (df['low'] <= df['close']).all():
        print("Warning: Low price is not lowest price")
        return False
    
    return True

def main():
    """Main function to test the data processing pipeline."""
    print("Starting data processing pipeline...\n")
    
    # Initialize the processor
    processor = StockDataProcessor()
    
    # Test symbols including Berkshire Hathaway
    symbols = ["BRK-A", "BRK-B", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        df = processor.process_stock_data(symbol)
        
        if df is not None and not df.empty:
            print("Data quality check passed for", symbol)
            
            # Calculate date range
            date_range = df.index.max() - df.index.min()
            
            print("\nProcessing Summary for {}:".format(symbol))
            print("Date range: {} to {}".format(
                df.index.min().strftime('%Y-%m-%d'),
                df.index.max().strftime('%Y-%m-%d')
            ))
            print("Total days:", len(df))
            
            print("\nTechnical Indicators Summary:")
            print("Latest RSI: {:.2f}".format(df['rsi'].iloc[-1]))
            print("Latest MACD: {:.2f}".format(df['macd'].iloc[-1]))
            print("Latest Momentum: {:.2%}".format(df['momentum'].iloc[-1]))
            
            print("\nPrice Statistics:")
            print("Latest price: ${:.2f}".format(df['close'].iloc[-1]))
            print("20-day SMA: ${:.2f}".format(df['sma_20'].iloc[-1]))
            print("50-day SMA: ${:.2f}".format(df['sma_50'].iloc[-1]))
            print("200-day SMA: ${:.2f}".format(df['sma_200'].iloc[-1]))
            
            print("\nVolume Statistics:")
            print("Latest volume: {:,.0f}".format(df['volume'].iloc[-1]))
            print("Volume ratio: {:.2f}".format(df['volume_ratio'].iloc[-1]))
            
            # Additional Berkshire specific analysis
            if symbol in ['BRK-A', 'BRK-B']:
                print("\nBerkshire Specific Analysis:")
                print("YTD Return: {:.2%}".format(
                    (df['close'].iloc[-1] / df[df.index.year == df.index.year.min()]['close'].iloc[0]) - 1
                ))
                print("Annual Volatility: {:.2%}".format(
                    df['price_change'].std() * np.sqrt(252)
                ))
                print("Maximum Drawdown: {:.2%}".format(
                    ((df['close'] / df['close'].expanding().max()) - 1).min()
                ))
            
        else:
            print("Data processing failed for", symbol)
            
        print("-" * 50)

if __name__ == "__main__":
    main() 