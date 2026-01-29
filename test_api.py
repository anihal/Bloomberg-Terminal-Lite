"""Test script to debug Alpha Vantage API response."""
import requests
import json
from config import ALPHA_VANTAGE_KEY

def test_api():
    """Test the Alpha Vantage API and print the response."""
    symbol = "AAPL"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_KEY,
        "outputsize": "compact"  # Changed to compact for free tier
    }

    print(f"Testing API with key: {ALPHA_VANTAGE_KEY}")
    print(f"URL: https://www.alphavantage.co/query")
    print(f"Params: {params}\n")

    response = requests.get("https://www.alphavantage.co/query", params=params)
    print(f"Status Code: {response.status_code}")

    data = response.json()
    print(f"\nAPI Response Keys: {data.keys()}\n")

    if "Error Message" in data:
        print(f"\nAPI Error: {data['Error Message']}")

    if "Note" in data:
        print(f"\nAPI Note (Rate Limit): {data['Note']}")

    if "Information" in data:
        print(f"\nAPI Information: {data['Information']}")

    time_series = data.get("Time Series (Daily)")
    if time_series:
        print(f"\nTime Series found! Number of dates: {len(time_series)}")
        print(f"Most recent date: {list(time_series.keys())[0]}")
        print(f"Sample data: {list(time_series.values())[0]}")
    else:
        print("\nNo Time Series (Daily) data found in response!")
        print(f"Full Response:\n{json.dumps(data, indent=2)}")

if __name__ == "__main__":
    test_api()
