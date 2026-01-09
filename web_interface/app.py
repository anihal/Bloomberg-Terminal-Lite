from flask import Flask, render_template, request, jsonify
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Span
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import yfinance as yf

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import StockDataProcessor
from model_trainer import ModelTrainer
from stock_fetcher import get_stock_data

app = Flask(__name__)

# Initialize our data processor and model trainer
processor = StockDataProcessor()
model_trainer = ModelTrainer()

def fetch_stock_data(symbol, start_date=None, end_date=None):
    """Fetch stock data directly from yfinance."""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            return None
            
        # Calculate technical indicators
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def create_stock_plot(df, symbol):
    """Create a Bokeh plot for stock data."""
    # Create the main figure
    p = figure(
        title=f"{symbol} Stock Price",
        x_axis_label="Date",
        y_axis_label="Price",
        x_axis_type="datetime",
        height=400,
        width=800
    )
    
    # Add candlestick chart
    p.line(df.index, df['Close'], line_color="navy", line_width=2, legend_label="Close Price")
    
    # Add moving averages
    p.line(df.index, df['sma_20'], line_color="orange", line_width=1, legend_label="20-day SMA")
    p.line(df.index, df['sma_50'], line_color="green", line_width=1, legend_label="50-day SMA")
    p.line(df.index, df['sma_200'], line_color="red", line_width=1, legend_label="200-day SMA")
    
    # Configure legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    # Add RSI subplot
    rsi = figure(
        title="RSI",
        x_axis_label="Date",
        y_axis_label="RSI",
        x_axis_type="datetime",
        height=200,
        width=800,
        x_range=p.x_range
    )
    
    # Add RSI line
    rsi.line(df.index, df['rsi'], line_color="purple", line_width=2)
    
    # Add horizontal lines for overbought/oversold levels using Span
    overbought = Span(location=70, dimension='width', line_color='red', line_dash='dashed', line_width=1)
    oversold = Span(location=30, dimension='width', line_color='green', line_dash='dashed', line_width=1)
    rsi.add_layout(overbought)
    rsi.add_layout(oversold)
    
    # Combine plots
    layout = column(p, rsi)
    return layout

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle stock analysis request."""
    symbol = request.form.get('symbol', '').upper()

    try:
        # Fetch real-time stock data using new get_stock_data function
        stock_data = get_stock_data(symbol)
        if stock_data is None:
            return jsonify({'error': f'Unable to fetch data for symbol {symbol}. Please check if the symbol is valid.'})

        # Try to get historical data from database first for charting
        df = processor.process_stock_data(symbol)

        # If no data in database, fetch directly from yfinance for historical chart
        if df is None or df.empty:
            df = fetch_stock_data(symbol)
            if df is None:
                return jsonify({'error': f'No historical data found for symbol {symbol}'})

        # Create visualization
        plot = create_stock_plot(df, symbol)
        script, div = components(plot)

        # Format market cap
        market_cap = stock_data.get('market_cap')
        if market_cap:
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
        else:
            market_cap_str = "N/A"

        # Calculate metrics combining real-time data and technical indicators
        metrics = {
            'Company': stock_data.get('company_name', symbol),
            'Current Price': f"${stock_data.get('current_price', 0):.2f}",
            'Change': f"${stock_data.get('change', 0):.2f} ({stock_data.get('percent_change', 0):.2f}%)",
            'Day High': f"${stock_data.get('high', 0):.2f}",
            'Day Low': f"${stock_data.get('low', 0):.2f}",
            'Open': f"${stock_data.get('open', 0):.2f}",
            'Previous Close': f"${stock_data.get('previous_close', 0):.2f}",
            'Market Cap': market_cap_str,
            'P/E Ratio': f"{stock_data.get('pe_ratio', 'N/A'):.2f}" if stock_data.get('pe_ratio') else 'N/A',
            'Forward P/E': f"{stock_data.get('forward_pe', 'N/A'):.2f}" if stock_data.get('forward_pe') else 'N/A',
            'PEG Ratio': f"{stock_data.get('peg_ratio', 'N/A'):.2f}" if stock_data.get('peg_ratio') else 'N/A',
            'Sector': stock_data.get('sector', 'N/A'),
            'Industry': stock_data.get('industry', 'N/A'),
            'RSI': f"{df['rsi'].iloc[-1]:.2f}" if not pd.isna(df['rsi'].iloc[-1]) else 'N/A',
            'MACD': f"{df['macd'].iloc[-1]:.2f}" if not pd.isna(df['macd'].iloc[-1]) else 'N/A',
            'Avg Volume': f"{stock_data.get('average_volume', 0):,.0f}" if stock_data.get('average_volume') else 'N/A'
        }

        return jsonify({
            'success': True,
            'script': script,
            'div': div,
            'metrics': metrics
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 