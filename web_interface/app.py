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

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import StockDataProcessor
from model_trainer import ModelTrainer

app = Flask(__name__)

# Initialize our data processor and model trainer
processor = StockDataProcessor()
model_trainer = ModelTrainer()

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
    p.line(df.index, df['close'], line_color="navy", line_width=2, legend_label="Close Price")
    
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
        # Get stock data
        df = processor.process_stock_data(symbol)
        if df is None or df.empty:
            return jsonify({'error': f'No data found for symbol {symbol}'})
        
        # Create visualization
        plot = create_stock_plot(df, symbol)
        script, div = components(plot)
        
        # Get model predictions
        X, y = model_trainer.prepare_features(df)
        if X is not None and y is not None:
            model, _, _ = model_trainer.train_model(X, y, {})
            predictions = model.predict(X)
            
            # Calculate some basic metrics
            metrics = {
                'Latest Price': f"${df['close'].iloc[-1]:.2f}",
                'RSI': f"{df['rsi'].iloc[-1]:.2f}",
                'MACD': f"{df['macd'].iloc[-1]:.2f}",
                'Volume': f"{df['volume'].iloc[-1]:,.0f}",
                'Prediction': f"${predictions[-1]:.2f}"
            }
        else:
            metrics = {
                'Latest Price': f"${df['close'].iloc[-1]:.2f}",
                'RSI': f"{df['rsi'].iloc[-1]:.2f}",
                'MACD': f"{df['macd'].iloc[-1]:.2f}",
                'Volume': f"{df['volume'].iloc[-1]:,.0f}",
                'Prediction': "N/A"
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