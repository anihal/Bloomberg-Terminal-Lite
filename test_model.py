import pandas as pd
import numpy as np
from model_trainer import ModelTrainer
from data_processor import StockDataProcessor
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to test the stock price prediction model.
    """
    try:
        # Initialize processor and trainer
        processor = StockDataProcessor()
        model_trainer = ModelTrainer()
        
        # List of stocks to process
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM']
        
        # Date range for testing (use 2023 data)
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        # Create a summary DataFrame to store results
        results_summary = []
        
        for stock in stocks:
            print(f"\nProcessing {stock}...")
            try:
                # Load and process stock data
                data = processor.load_stock_data(stock, start_date, end_date)
                if data is None or len(data) < 100:
                    print(f"Insufficient data for {stock}")
                    continue
                
                # Prepare features and target
                X, y = model_trainer.prepare_features(data)
                
                if len(X) < 100:
                    print(f"Insufficient data after feature calculation for {stock}")
                    continue
                
                # Split data into training and validation sets (80/20)
                train_size = int(len(X) * 0.8)
                X_train = X[:train_size]
                y_train = y[:train_size]
                X_val = X[train_size:]
                y_val = y[train_size:]
                
                # Optimize hyperparameters with more trials and early stopping
                print(f"Optimizing hyperparameters for {stock}...")
                best_params = model_trainer.optimize_hyperparameters(
                    X_train, y_train,
                    n_trials=100  # Increase number of trials
                )
                
                # Train model with early stopping
                print(f"Training model for {stock}...")
                model, scaler, feature_importance = model_trainer.train_model(
                    X_train, y_train,
                    best_params
                )
                
                # Save model and related data
                model_trainer.save_model(model, scaler, feature_importance, stock)
                
                # Plot feature importance
                model_trainer.plot_feature_importance(feature_importance, stock)
                
                # Plot learning curves
                model_trainer.plot_learning_curves(model, stock)
                
                # Print feature importance
                print("\nTop 10 Most Important Features:")
                for feature, importance in feature_importance.head(10).items():
                    print(f"{feature}: {importance:.4f}")
                
                # Evaluate model on validation set
                print("\nModel Evaluation Metrics:")
                metrics = model_trainer.evaluate_model(model, scaler, data.iloc[train_size:])
                if metrics:
                    print(f"RMSE: {metrics['rmse']:.4f}")
                    print(f"MAE: {metrics['mae']:.4f}")
                    print(f"Direction Accuracy: {metrics['direction_accuracy']:.4f}")
                    print(f"Strategy Returns: {metrics['strategy_returns']:.4f}")
                    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
                    print(f"Maximum Drawdown: {metrics['max_drawdown']:.4f}")
                    print(f"Win Rate: {metrics['win_rate']:.4f}")
                    print(f"Profit Factor: {metrics['profit_factor']:.4f}")
                    
                    # Add results to summary
                    results_summary.append({
                        'Stock': stock,
                        'RMSE': metrics['rmse'],
                        'MAE': metrics['mae'],
                        'Direction Accuracy': metrics['direction_accuracy'],
                        'Strategy Returns': metrics['strategy_returns'],
                        'Sharpe Ratio': metrics['sharpe_ratio'],
                        'Max Drawdown': metrics['max_drawdown'],
                        'Win Rate': metrics['win_rate'],
                        'Profit Factor': metrics['profit_factor']
                    })
                
            except Exception as e:
                print(f"Error processing {stock}: {str(e)}")
                continue
            
        # Create and save results summary
        if results_summary:
            summary_df = pd.DataFrame(results_summary)
            summary_df.to_csv(model_trainer.models_dir / "results_summary.csv", index=False)
            
            # Plot summary metrics
            plt.figure(figsize=(15, 10))
            metrics_to_plot = ['RMSE', 'Direction Accuracy', 'Strategy Returns', 'Sharpe Ratio', 'Win Rate']
            for i, metric in enumerate(metrics_to_plot, 1):
                plt.subplot(2, 3, i)
                sns.barplot(x='Stock', y=metric, data=summary_df)
                plt.title(f'{metric} by Stock')
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(model_trainer.models_dir / "summary_metrics.png")
            plt.close()
            
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM"]
    
    # Set date range for training
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    for symbol in test_symbols:
        print(f"\nProcessing {symbol}...")
        
        try:
            # Get stock data with date range
            df = trainer.processor.load_stock_data(symbol, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                print(f"No data found for {symbol} between {start_date} and {end_date}")
                continue
                
            # Clean and calculate indicators
            df = trainer.processor.clean_data(df)
            if df.empty:
                print(f"No valid data after cleaning for {symbol}")
                continue
                
            # Prepare features
            X, y = trainer.prepare_features(df)
            if X is None or y is None:
                print(f"Insufficient data after feature calculation for {symbol}")
                continue
                
            # Optimize hyperparameters
            print(f"Training model for {symbol}...")
            best_params = trainer.optimize_hyperparameters(X, y, n_trials=100)
            
            # Train model with optimized parameters
            model, scaler, feature_importance = trainer.train_model(X, y, best_params)
            
            # Save model
            trainer.save_model(model, scaler, feature_importance, symbol)
            
            # Print feature importance
            print("\nTop 10 Most Important Features:")
            for feature, importance in feature_importance.head(10).items():
                print(f"{feature}: {importance:.4f}")
                
            # Evaluate model
            metrics = trainer.evaluate_model(X, y)
            print("\nModel Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
                
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")