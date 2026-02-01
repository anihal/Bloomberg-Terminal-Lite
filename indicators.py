"""
Technical indicators - SMA, EMA, RSI, MACD, Bollinger Bands, Momentum, Volume.
"""
import pandas as pd


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Simple Moving Averages (20, 50, 100) and Exponential Moving Averages (20, 50).

    Args:
        df: DataFrame with 'close' column

    Returns:
        DataFrame with added columns: sma_20, sma_50, sma_100, ema_20, ema_50
    """
    df = df.copy()

    # Simple Moving Averages
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_100"] = df["close"].rolling(window=100).mean()

    # Exponential Moving Averages
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index (14-day).

    Args:
        df: DataFrame with 'close' column
        period: RSI period (default: 14)

    Returns:
        DataFrame with added column: rsi
    """
    df = df.copy()

    delta = df["close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Add MACD (Moving Average Convergence Divergence).

    Args:
        df: DataFrame with 'close' column
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        DataFrame with added columns: macd, macd_signal, macd_histogram
    """
    df = df.copy()

    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Add Bollinger Bands (20-day).

    Args:
        df: DataFrame with 'close' column
        period: Moving average period (default: 20)
        std_dev: Number of standard deviations (default: 2.0)

    Returns:
        DataFrame with added columns: bb_middle, bb_upper, bb_lower, bb_width
    """
    df = df.copy()

    df["bb_middle"] = df["close"].rolling(window=period).mean()
    rolling_std = df["close"].rolling(window=period).std()

    df["bb_upper"] = df["bb_middle"] + (rolling_std * std_dev)
    df["bb_lower"] = df["bb_middle"] - (rolling_std * std_dev)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Price Momentum indicators.

    Args:
        df: DataFrame with 'close' column

    Returns:
        DataFrame with added columns: momentum_10, momentum_20, roc_10, roc_20
    """
    df = df.copy()

    # Price Momentum (difference)
    df["momentum_10"] = df["close"] - df["close"].shift(10)
    df["momentum_20"] = df["close"] - df["close"].shift(20)

    # Rate of Change (percentage)
    df["roc_10"] = ((df["close"] - df["close"].shift(10)) / df["close"].shift(10)) * 100
    df["roc_20"] = ((df["close"] - df["close"].shift(20)) / df["close"].shift(20)) * 100

    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Volume indicators.

    Args:
        df: DataFrame with 'close' and 'volume' columns

    Returns:
        DataFrame with added columns: volume_sma_20, volume_ratio, obv, vwap
    """
    df = df.copy()

    # Volume Moving Average
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()

    # Volume Ratio (current volume / average volume)
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    # On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv

    # VWAP (Volume Weighted Average Price) - daily reset typical
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df


def calculate_1m_return(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Calculate 1-month percentage return based on closing price.

    Args:
        df: DataFrame with 'close' column and a date column
        date_col: Name of the date column (default: 'date')

    Returns:
        DataFrame with added column: return_1m (percentage return)
    """
    df = df.copy()

    # Sort by date ascending to prevent look-ahead bias
    if date_col in df.columns:
        df = df.sort_values(by=date_col, ascending=True).reset_index(drop=True)

    # 1 month ≈ 21 trading days
    trading_days_1m = 21

    if len(df) < trading_days_1m:
        # Not enough data for 1-month return; fill with NaN
        df["return_1m"] = pd.NA
    else:
        df["return_1m"] = (
            (df["close"] - df["close"].shift(trading_days_1m))
            / df["close"].shift(trading_days_1m)
        ) * 100

    return df


def calculate_3m_return(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Calculate 3-month percentage return based on closing price.

    Args:
        df: DataFrame with 'close' column and a date column
        date_col: Name of the date column (default: 'date')

    Returns:
        DataFrame with added column: return_3m (percentage return)
    """
    df = df.copy()

    # Sort by date ascending to prevent look-ahead bias
    if date_col in df.columns:
        df = df.sort_values(by=date_col, ascending=True).reset_index(drop=True)

    # 3 months ≈ 63 trading days
    trading_days_3m = 63

    if len(df) < trading_days_3m:
        # Not enough data for 3-month return; fill with NaN
        df["return_3m"] = pd.NA
    else:
        df["return_3m"] = (
            (df["close"] - df["close"].shift(trading_days_3m))
            / df["close"].shift(trading_days_3m)
        ) * 100

    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the DataFrame.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns

    Returns:
        DataFrame with all indicator columns added
    """
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_momentum(df)
    df = add_volume_indicators(df)

    return df
