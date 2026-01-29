# Bloomberg Terminal Lite

A lightweight, web-based stock terminal built with Python and Streamlit. Get real-time stock data, technical indicators, and interactive charts for market analysis.

## Features

- **Real-time Stock Data**: Fetch daily stock prices using Alpha Vantage API
- **Technical Indicators**:
  - Moving Averages (SMA 20/50/200, EMA 20/50)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Price Momentum & ROC (Rate of Change)
  - Volume indicators (OBV, VWAP, Volume Ratio)
- **Interactive Charts**: Visualize price trends, RSI, and MACD
- **Clean UI**: Professional dashboard layout with metrics and indicators

## Tech Stack

- **Python 3.8+**
- **Streamlit**: Web framework for the UI
- **Pandas**: Data manipulation and analysis
- **Alpha Vantage API**: Real-time stock market data
- **python-dotenv**: Environment variable management

## Installation

### Prerequisites

- Python 3.8 or higher
- Alpha Vantage API key (free)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/anihal/Bloomberg-Terminal-Lite.git
cd Bloomberg-Terminal-Lite
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Get your free Alpha Vantage API key:
   - Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Sign up and get your API key

4. Create a `.env` file in the project root:
```bash
cp .env.example .env
```

5. Edit `.env` and add your API key:
```
ALPHA_VANTAGE_KEY=your_api_key_here
```

## Usage

Run the application locally:

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

Enter a stock ticker (e.g., AAPL, MSFT, GOOGL) to view:
- Current price and volume
- Price information (Open, High, Low, Close)
- Technical indicators
- Interactive charts
- Historical data

## Deployment on Render

1. Fork this repository to your GitHub account

2. Create a new Web Service on [Render](https://render.com):
   - Connect your GitHub repository
   - Select `Bloomberg-Terminal-Lite`

3. Configure the service:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run main.py --server.port $PORT --server.address 0.0.0.0`

4. Add environment variable:
   - Key: `ALPHA_VANTAGE_KEY`
   - Value: Your Alpha Vantage API key

5. Deploy and access your live app!

## API Rate Limits

Alpha Vantage free tier limitations:
- 5 API calls per minute
- 500 API calls per day

The app caches data for 1 hour to minimize API usage.

## Project Structure

```
Bloomberg-Terminal-Lite/
├── main.py              # Streamlit UI entry point
├── data_provider.py     # Alpha Vantage API client
├── indicators.py        # Technical indicator calculations
├── utils.py             # Utility functions for formatting
├── config.py            # Configuration and environment variables
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment file
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## License

MIT License - feel free to use this project for learning or personal use.

## Acknowledgments

- Data provided by [Alpha Vantage](https://www.alphavantage.co/)
- Built with [Streamlit](https://streamlit.io/)

## Support

For issues or questions, please open an issue on GitHub.
