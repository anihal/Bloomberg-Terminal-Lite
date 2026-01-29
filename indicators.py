"""
Technical indicators - SMA, EMA, RSI, etc.
"""
import pandas as pd


def calculate_sma(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        series: Price series (typically close prices)
        period: Number of periods for the average

    Returns:
        pandas Series with SMA values
    """
    return series.rolling(window=period).mean()


def calculate_ema(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        series: Price series (typically close prices)
        period: Number of periods for the average

    Returns:
        pandas Series with EMA values
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Args:
        series: Price series (typically close prices)
        period: Number of periods for RSI calculation

    Returns:
        pandas Series with RSI values (0-100)
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi
