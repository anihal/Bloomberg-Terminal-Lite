# Bloomberg Terminal Lite

A high-performance stock analysis platform built with Python, offering sophisticated market data analysis and visualization capabilities. This project demonstrates expertise in financial data processing, machine learning, and full-stack development.

## 🌐 Live Demo

Try it now: [Bloomberg Terminal Lite](https://bloomberg-terminal-lite.onrender.com/)

Enter any stock symbol (e.g., AAPL, MSFT, GOOGL) to see:
- Real-time stock price analysis
- Technical indicators (SMA 20, 50, 200)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Daily price changes and volume

Note: The demo is hosted on Render's free tier, so initial load might take 30-60 seconds if the service is idle.

## 🎯 Project Overview

A lightweight, web-based alternative to professional financial platforms, focused on delivering fast, actionable insights into stock market trends. Perfect for traders, analysts, and financial enthusiasts who need quick access to market data and predictions.

## ✨ Implemented Features

### 1. Real-time Stock Data Fetching
- **Multi-source Integration**: Combines Finnhub (real-time quotes) with yfinance (company metrics)
- **On-demand Execution**: `get_stock_data(ticker)` function executes only when called
- **Comprehensive Metrics**: Current price, P/E ratio, Market Cap, Sector, Industry, and more
- **Smart Rate Limiting**: Built-in protection against API rate limits with retry logic
- **Environment Configuration**: API keys externalized using .env files

### 2. Advanced Data Processing Pipeline
- **Data Cleaning**: Sophisticated outlier detection and handling
- **Technical Indicators**: Comprehensive suite including:
  - Moving Averages (SMA 20, 50, 200)
  - Exponential Moving Averages (EMA 20, 50)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Volume Indicators
  - Price Momentum

### 3. Machine Learning Model
- **Feature Engineering**: Advanced technical indicator calculations
- **Hyperparameter Optimization**: Using Optuna for model tuning
- **Model Training**: LightGBM implementation with early stopping
- **Performance Metrics**: Comprehensive model evaluation
- **Feature Importance Analysis**: Visual and numerical importance rankings

### 4. Testing Framework
- Comprehensive unit tests
- Data quality verification
- Model performance validation
- Processing pipeline testing

## 🚀 Future Enhancements

### 1. Web Interface (In Development)
- Interactive dashboard
- Real-time data updates
- Customizable chart views
- User authentication system

### 2. Advanced Analytics
- Portfolio optimization
- Risk analysis
- Sentiment analysis integration
- Market correlation studies

### 3. Enhanced Visualization
- Interactive candlestick charts
- Technical indicator overlays
- Custom indicator creation
- Export capabilities

### 4. Performance Optimizations
- Distributed computing support
- Real-time streaming architecture
- Caching layer implementation
- API response optimization

## 🛠 Tech Stack

- **Core**: Python 3.9+
- **Data Processing**: Pandas, NumPy, Polars
- **Machine Learning**: LightGBM, scikit-learn
- **Optimization**: Optuna
- **Data Storage**: SQLite, SQLAlchemy
- **API Integration**: Finnhub (real-time quotes), yfinance (company metrics & historical data)
- **Web Framework**: Flask, Bokeh (visualization)
- **Testing**: pytest
- **Configuration**: python-dotenv

## 📊 Performance Metrics

- Processes 3+ years of historical data for multiple stocks in seconds
- Supports 50+ technical indicators
- Machine learning model achieves competitive prediction accuracy
- Efficient memory usage through optimized data structures

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Finnhub API key (free at https://finnhub.io/register)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/Bloomberg-Terminal-Lite.git
cd Bloomberg-Terminal-Lite
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your Finnhub API key
# FINNHUB_API_KEY=your_actual_api_key_here
```

4. Run the web interface:
```bash
python web_interface/app.py
```

### Optional: Legacy Data Collection

For historical data collection and model training:

```bash
# Collect historical data
python data_collector.py

# Train ML models
python model_trainer.py
```

## 📈 Sample Usage

### Real-time Stock Data Fetching

```python
from stock_fetcher import get_stock_data

# Get real-time stock data (price from Finnhub + metrics from yfinance)
data = get_stock_data('AAPL')

if data:
    print(f"Company: {data['company_name']}")
    print(f"Current Price: ${data['current_price']:.2f}")
    print(f"Change: ${data['change']:.2f} ({data['percent_change']:.2f}%)")
    print(f"Market Cap: ${data['market_cap']:,}")
    print(f"P/E Ratio: {data['pe_ratio']:.2f}")
    print(f"Sector: {data['sector']}")
```

### Historical Data Collection (Legacy)

```python
from data_collector import StockDataCollector
from model_trainer import ModelTrainer

# Collect historical data
collector = StockDataCollector()
collector.collect_stock_data("AAPL", "2020-01-01", "2023-12-31")

# Train model
trainer = ModelTrainer()
model = trainer.train_for_symbol("AAPL")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

Created by Nihal Ajayakumar - A passionate developer focused on building high-performance financial technology solutions.

---
*Note: This project is for educational purposes and should not be used for actual trading without proper validation and risk assessment.* 