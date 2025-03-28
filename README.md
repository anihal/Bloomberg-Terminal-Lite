# Bloomberg Terminal Lite

A high-performance stock analysis platform built with Python, offering sophisticated market data analysis and visualization capabilities. This project demonstrates expertise in financial data processing, machine learning, and full-stack development.

## 🎯 Project Overview

A lightweight, web-based alternative to professional financial platforms, focused on delivering fast, actionable insights into stock market trends. Perfect for traders, analysts, and financial enthusiasts who need quick access to market data and predictions.

## ✨ Implemented Features

### 1. Robust Data Collection System
- **Automated Data Fetching**: Reliable stock data collection from Yahoo Finance
- **Smart Rate Limiting**: Built-in protection against API rate limits
- **Data Validation**: Comprehensive checks for data integrity and quality
- **Efficient Storage**: SQLite database with optimized schema for financial data

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
- **Data Storage**: SQLite
- **API Integration**: yfinance
- **Testing**: pytest

## 📊 Performance Metrics

- Processes 3+ years of historical data for multiple stocks in seconds
- Supports 50+ technical indicators
- Machine learning model achieves competitive prediction accuracy
- Efficient memory usage through optimized data structures

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run data collection:
```bash
python data_collector.py
```
4. Process data and train models:
```bash
python model_trainer.py
```

## 📈 Sample Usage

```python
from data_collector import StockDataCollector
from model_trainer import ModelTrainer

# Collect data
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