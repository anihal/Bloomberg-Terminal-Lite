import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import optuna
from loguru import logger
from pathlib import Path
import joblib
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

from data_processor import StockDataProcessor
from config import PROCESSED_DATA_DIR

class ModelTrainer:
    def __init__(self):
        """Initialize the model trainer."""
        self.processor = StockDataProcessor()
        self.models_dir = Path(PROCESSED_DATA_DIR) / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model training."""
        # Create a copy of the DataFrame to avoid warnings
        df = df.copy()
        
        logger.info(f"Starting feature preparation with {len(df)} data points")
        
        # Check if we have enough data
        if len(df) < 200:  # Need at least 200 days for SMA200
            logger.warning(f"Insufficient data points ({len(df)}) for feature calculation. Need at least 200.")
            return None, None
            
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        logger.info(f"After calculating technical indicators: {len(df)} data points")
        
        # Define feature columns
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'sma_200',
            'ema_20', 'ema_50',
            'rsi', 'macd', 'macd_signal',
            'momentum', 'price_range', 'volatility', 'volume_momentum'
        ]
        
        # Check for missing columns
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return None, None
        
        # Remove rows with missing values
        before_dropna = len(df)
        df = df.dropna(subset=feature_columns + ['target'])
        after_dropna = len(df)
        logger.info(f"Removed {before_dropna - after_dropna} rows with missing values")
        
        # Check if we have enough data after removing NaN values
        if len(df) < 100:  # Minimum required samples
            logger.warning(f"Insufficient data points ({len(df)}) after removing NaN values. Need at least 100.")
            return None, None
            
        # Scale features
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(df[feature_columns]),
            columns=feature_columns,
            index=df.index
        )
        
        # Store scaler for later use
        self.scaler = scaler
        
        logger.info(f"Final feature set has {len(X)} data points")
        
        return X, df['target']
        
    def create_time_series_splits(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> List[Tuple]:
        """Create time series cross-validation splits."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return list(tscv.split(X, y))
        
    def optimize_hyperparameters(self, X_train, y_train, n_trials=50):
        """Optimize LightGBM hyperparameters using Optuna."""
        def objective(trial):
            param = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 16, 256),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True)
            }
            
            cv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X_train):
                X_train_cv = X_train.iloc[train_idx]
                y_train_cv = y_train.iloc[train_idx]
                X_val_cv = X_train.iloc[val_idx]
                y_val_cv = y_train.iloc[val_idx]
                
                model = lgb.LGBMRegressor(**param)
                model.fit(
                    X_train_cv, y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    callbacks=[lgb.early_stopping(100)]
                )
                
                pred = model.predict(X_val_cv)
                score = np.sqrt(mean_squared_error(y_val_cv, pred))  # Calculate RMSE directly
                cv_scores.append(score)
            
            return np.mean(cv_scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
        
    def train_model(self, X, y, best_params):
        """Train a LightGBM model with the given parameters.
        
        Args:
            X: Features DataFrame
            y: Target Series
            best_params: Dictionary of best hyperparameters from optimization
            
        Returns:
            model: Trained LightGBM model
            scaler: Fitted StandardScaler
            feature_importance: Series of feature importance scores
        """
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model with best parameters
        self.model = LGBMRegressor(**best_params)
        self.model.fit(X_scaled, y)
        
        # Get feature importance
        self.feature_importance = pd.Series(self.model.feature_importances_, index=X.columns)
        self.feature_importance = self.feature_importance.sort_values(ascending=False)
        
        return self.model, self.scaler, self.feature_importance
            
    def save_model(self, model, scaler, feature_importance, stock_symbol: str, version: str = None):
        """Save model, scaler, and feature importance to disk.
        
        Args:
            model: Trained LightGBM model
            scaler: Fitted StandardScaler
            feature_importance: Series of feature importance scores
            stock_symbol: Stock symbol
            version: Optional version string for model versioning
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_str = f"_{version}" if version else ""
        model_path = self.models_dir / f"{stock_symbol}_{timestamp}{version_str}"
        
        # Save model
        model.booster_.save_model(str(model_path) + ".model")
        
        # Save scaler
        joblib.dump(scaler, str(model_path) + "_scaler.joblib")
        
        # Save feature importance
        feature_importance.to_csv(str(model_path) + "_feature_importance.csv")
        
        self.logger.info(f"Saved model for {stock_symbol} to {model_path}")
        
    def load_model(self, stock_symbol: str, version: str = None) -> Tuple[LGBMRegressor, StandardScaler, pd.Series]:
        """Load the latest model for a stock.
        
        Args:
            stock_symbol: Stock symbol
            version: Optional version string for model versioning
            
        Returns:
            Tuple of (model, scaler, feature_importance)
        """
        # Find the latest model file for this stock
        version_str = f"_{version}" if version else ""
        model_files = list(self.models_dir.glob(f"{stock_symbol}_*{version_str}.model"))
        if not model_files:
            raise FileNotFoundError(f"No model found for {stock_symbol}")
            
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        base_path = latest_model.with_suffix("")
        
        # Load model
        model = LGBMRegressor()
        model.booster_ = lgb.Booster(model_file=str(latest_model))
        
        # Load scaler
        scaler = joblib.load(str(base_path) + "_scaler.joblib")
        
        # Load feature importance
        feature_importance = pd.read_csv(str(base_path) + "_feature_importance.csv", index_col=0)
        
        return model, scaler, feature_importance
        
    def calculate_additional_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Calculate additional performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            returns: Actual returns
            
        Returns:
            Dictionary of additional metrics
        """
        # Calculate strategy returns
        strategy_returns = returns * np.sign(y_pred)
        
        # Calculate Sharpe ratio (assuming daily data)
        daily_rf_rate = 0.02 / 252  # 2% annual risk-free rate
        excess_returns = strategy_returns - daily_rf_rate
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        
        # Calculate maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        # Calculate win rate
        win_rate = np.mean(strategy_returns > 0)
        
        # Calculate profit factor
        gross_profits = np.sum(strategy_returns[strategy_returns > 0])
        gross_losses = abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
        
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, stock_symbol: str):
        """Plot actual vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            stock_symbol: Stock symbol
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted Returns - {stock_symbol}')
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = self.models_dir / f"{stock_symbol}_predictions.png"
        plt.savefig(plot_path)
        plt.close()
        
    def plot_feature_importance(self, feature_importance: pd.Series, stock_symbol: str):
        """Plot feature importance.
        
        Args:
            feature_importance: Series of feature importance scores
            stock_symbol: Stock symbol
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(x=feature_importance.values, y=feature_importance.index)
        plt.title(f'Feature Importance - {stock_symbol}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        # Save plot
        plot_path = self.models_dir / f"{stock_symbol}_feature_importance.png"
        plt.savefig(plot_path)
        plt.close()
        
    def plot_learning_curves(self, model: LGBMRegressor, stock_symbol: str):
        """Plot learning curves from model training.
        
        Args:
            model: Trained LightGBM model
            stock_symbol: Stock symbol
        """
        if hasattr(model, 'evals_result'):
            results = model.evals_result()
            plt.figure(figsize=(12, 6))
            
            for metric in results['training']:
                plt.plot(results['training'][metric], label=f'Training {metric}')
                plt.plot(results['valid_0'][metric], label=f'Validation {metric}')
                
            plt.title(f'Learning Curves - {stock_symbol}')
            plt.xlabel('Iteration')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plot_path = self.models_dir / f"{stock_symbol}_learning_curves.png"
            plt.savefig(plot_path)
            plt.close()
            
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        # Create a copy of the DataFrame to avoid warnings
        X = X.copy()
        y = y.copy()
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # Calculate direction accuracy
        direction_accuracy = np.mean(np.sign(y) == np.sign(y_pred))
        
        # Calculate strategy returns
        strategy_returns = np.mean(np.sign(y_pred) * y)
        
        # Calculate Sharpe ratio
        returns = np.sign(y_pred) * y
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calculate win rate
        win_rate = np.mean(returns > 0)
        
        # Calculate profit factor
        gross_profits = np.sum(returns[returns > 0])
        gross_losses = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        return {
            'rmse': rmse,
            'mae': mae,
            'direction_accuracy': direction_accuracy,
            'strategy_returns': strategy_returns,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # Create a copy of the DataFrame to avoid warnings
        df = df.copy()
        
        logger.info(f"Starting technical indicator calculation with {len(df)} data points")
        
        # Calculate moving averages
        df.loc[:, 'sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df.loc[:, 'sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        df.loc[:, 'sma_200'] = df['close'].rolling(window=200, min_periods=1).mean()
        df.loc[:, 'ema_20'] = df['close'].ewm(span=20, adjust=False, min_periods=1).mean()
        df.loc[:, 'ema_50'] = df['close'].ewm(span=50, adjust=False, min_periods=1).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df.loc[:, 'rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
        exp2 = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
        df.loc[:, 'macd'] = exp1 - exp2
        df.loc[:, 'macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
        
        # Calculate additional features
        df.loc[:, 'daily_return'] = df['close'].pct_change()
        df.loc[:, 'volatility'] = df['daily_return'].rolling(window=20, min_periods=1).std()
        df.loc[:, 'price_range'] = (df['high'] - df['low']) / df['close']
        df.loc[:, 'volume_momentum'] = df['volume'].pct_change()
        df.loc[:, 'momentum'] = df['close'].pct_change(periods=10)
        
        # Calculate target variable
        df.loc[:, 'target'] = df['close'].pct_change().shift(-1)
        
        # Log missing values for each column
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                logger.warning(f"Column {col} has {missing} missing values")
        
        return df

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM"]
    
    for symbol in test_symbols:
        print(f"\nProcessing {symbol}...")
        
        try:
            # Get stock data
            df = trainer.processor.process_stock_data(symbol)
            if df is None or df.empty:
                print(f"No data found for {symbol}")
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