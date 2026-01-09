"""
Test script for the new stock_fetcher module.
"""
from stock_fetcher import get_stock_data
from loguru import logger
import json


def test_get_stock_data(ticker='AAPL'):
    """
    Test the get_stock_data function.

    Args:
        ticker: Stock symbol to test (default: AAPL)
    """
    logger.info(f"Testing get_stock_data for {ticker}")

    # Fetch stock data
    data = get_stock_data(ticker)

    if data:
        logger.info(f"Successfully fetched data for {ticker}")
        print("\n" + "="*60)
        print(f"Stock Data for {ticker}")
        print("="*60)
        print(json.dumps(data, indent=2, default=str))
        print("="*60 + "\n")

        # Verify required fields
        required_fields = [
            'ticker', 'current_price', 'company_name', 'sector',
            'market_cap', 'pe_ratio'
        ]

        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            logger.warning(f"Missing fields: {missing_fields}")
        else:
            logger.success("All required fields are present")

        return True
    else:
        logger.error(f"Failed to fetch data for {ticker}")
        return False


if __name__ == "__main__":
    # Test with a few popular stocks
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']

    logger.info("Starting stock fetcher tests")
    print("\nNOTE: You must have FINNHUB_API_KEY set in your .env file")
    print("Get your free API key from: https://finnhub.io/register\n")

    success_count = 0
    for symbol in test_symbols:
        if test_get_stock_data(symbol):
            success_count += 1
        print("\n")

    print(f"\nTest Results: {success_count}/{len(test_symbols)} successful")

    if success_count == len(test_symbols):
        logger.success("All tests passed!")
    else:
        logger.warning(f"{len(test_symbols) - success_count} tests failed")
