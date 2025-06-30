"""
Technical Indicators Module for Stock Analysis

This module provides a comprehensive set of technical indicators for stock price
and volume analysis, optimized for vectorized pandas operations.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from scipy.signal import argrelextrema
import warnings

from ..config import settings


def simple_moving_average(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        data: Price series (typically Close prices)
        window: Period for moving average
        
    Returns:
        pd.Series: SMA values
    """
    if len(data) < window:
        return pd.Series(index=data.index, dtype=float)
    
    return data.rolling(window=window, min_periods=window).mean()


def exponential_moving_average(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        data: Price series (typically Close prices)
        window: Period for moving average
        
    Returns:
        pd.Series: EMA values
    """
    if len(data) < window:
        return pd.Series(index=data.index, dtype=float)
    
    return data.ewm(span=window, adjust=False).mean()


def relative_strength_index(data: pd.Series, window: Optional[int] = None) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data: Price series (typically Close prices)
        window: Period for RSI calculation (default: from config)
        
    Returns:
        pd.Series: RSI values (0-100)
    """
    if window is None:
        window = settings.RSI_PERIOD
    
    if len(data) < window + 1:
        return pd.Series(index=data.index, dtype=float)
    
    # Calculate price changes and ensure numeric type
    delta = data.diff().astype(float)
    
    # Separate gains and losses
    gains = delta.where(delta > 0.0, 0.0)
    losses = -delta.where(delta < 0.0, 0.0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=window, min_periods=window).mean()
    avg_losses = losses.rolling(window=window, min_periods=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data: Price series (typically Close prices)
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)
        
    Returns:
        Tuple of (MACD line, Signal line, MACD histogram)
    """
    if len(data) < slow:
        empty_series = pd.Series(index=data.index, dtype=float)
        return empty_series, empty_series, empty_series
    
    # Calculate EMAs
    ema_fast = exponential_moving_average(data, fast)
    ema_slow = exponential_moving_average(data, slow)
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = exponential_moving_average(macd_line, signal)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def bollinger_bands(data: pd.Series, window: Optional[int] = None, num_std: Optional[float] = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: Price series (typically Close prices)
        window: Period for moving average (default: from config)
        num_std: Number of standard deviations (default: from config)
        
    Returns:
        Tuple of (Upper band, Middle band/SMA, Lower band)
    """
    if window is None:
        window = settings.BOLLINGER_WINDOW
    if num_std is None:
        num_std = settings.BOLLINGER_STD_DEV
    
    if len(data) < window:
        empty_series = pd.Series(index=data.index, dtype=float)
        return empty_series, empty_series, empty_series
    
    # Calculate middle band (SMA)
    middle_band = simple_moving_average(data, window)
    
    # Calculate standard deviation
    rolling_std = data.rolling(window=window, min_periods=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    return upper_band, middle_band, lower_band


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: Optional[int] = None) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Period for ATR calculation (default: from config)
        
    Returns:
        pd.Series: ATR values
    """
    if window is None:
        window = settings.ATR_PERIOD
    
    if len(high) < window + 1:
        return pd.Series(index=high.index, dtype=float)
    
    # Calculate true range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    # True range is the maximum of the three components
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate average true range
    atr = true_range.rolling(window=window, min_periods=window).mean()
    
    return atr


def price_volatility(data: pd.Series, window: Optional[int] = None) -> pd.Series:
    """
    Calculate price volatility (standard deviation of returns).
    
    Args:
        data: Price series (typically Close prices)
        window: Period for volatility calculation (default: from config)
        
    Returns:
        pd.Series: Volatility values
    """
    if window is None:
        window = settings.VOLATILITY_WINDOW
    
    if len(data) < window + 1:
        return pd.Series(index=data.index, dtype=float)
    
    # Calculate returns
    returns = data.pct_change()
    
    # Calculate rolling standard deviation
    volatility = returns.rolling(window=window, min_periods=window).std()
    
    return volatility


def volume_average_ratio(volume: pd.Series, window: Optional[int] = None) -> pd.Series:
    """
    Calculate volume to average volume ratio.
    
    Args:
        volume: Volume series
        window: Period for average calculation (default: from config)
        
    Returns:
        pd.Series: Volume ratio values
    """
    if window is None:
        window = settings.VOLUME_AVERAGE_WINDOW
    
    if len(volume) < window:
        return pd.Series(index=volume.index, dtype=float)
    
    # Calculate rolling average volume
    avg_volume = volume.rolling(window=window, min_periods=window).mean()
    
    # Calculate ratio
    volume_ratio = volume / avg_volume
    
    return volume_ratio


def find_support_resistance_levels(data: pd.Series, window: Optional[int] = None, min_periods: Optional[int] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Find support and resistance levels using local minima and maxima.
    
    Args:
        data: Price series (typically Low for support, High for resistance)
        window: Window size for local extrema detection (default: from config)
        min_periods: Minimum periods required for calculation (default: from config)
        
    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    if window is None:
        window = settings.SUPPORT_RESISTANCE_WINDOW
    if min_periods is None:
        min_periods = settings.SUPPORT_RESISTANCE_MIN_PERIODS
    
    if len(data) < min_periods:
        empty_series = pd.Series(index=data.index, dtype=float)
        return empty_series, empty_series
    
    # Convert to numpy array for scipy
    values = data.values
    
    # Find local minima (support levels)
    local_minima_idx = argrelextrema(values, np.less, order=window)[0]
    
    # Find local maxima (resistance levels)
    local_maxima_idx = argrelextrema(values, np.greater, order=window)[0]
    
    # Create series with support and resistance levels
    support_levels = pd.Series(index=data.index, dtype=float)
    resistance_levels = pd.Series(index=data.index, dtype=float)
    
    # Fill in the levels
    for idx in local_minima_idx:
        if idx < len(data):
            support_levels.iloc[int(idx)] = float(values[idx])
    
    for idx in local_maxima_idx:
        if idx < len(data):
            resistance_levels.iloc[int(idx)] = float(values[idx])
    
    return support_levels, resistance_levels


def find_recent_support_level(low_prices: pd.Series, current_date_idx: int, lookback_days: int = 10) -> Optional[float]:
    """
    Find the most recent support level (lowest low) within a specified lookback period.
    
    Args:
        low_prices: Series of low prices
        current_date_idx: Index position of current date
        lookback_days: Number of days to look back (default: 10)
        
    Returns:
        float or None: Support level (lowest low) or None if insufficient data
    """
    # Calculate start index for lookback
    start_idx = max(0, current_date_idx - lookback_days)
    
    # Get the lookback period data
    lookback_period = low_prices.iloc[start_idx:current_date_idx]
    
    if len(lookback_period) == 0:
        return None
    
    # Return the minimum (support level)
    return float(lookback_period.min())


def calculate_linear_trend_slope(data: pd.Series, window: Optional[int] = None) -> float:
    """
    Calculate linear trend slope using least squares regression.
    
    Args:
        data: Price series
        window: Number of periods to use for trend calculation (default: from config)
        
    Returns:
        float: Trend slope (positive = uptrend, negative = downtrend)
    """
    if window is None:
        window = settings.TREND_SLOPE_WINDOW
    
    if len(data) < window:
        return 0.0
    
    # Get the last 'window' periods
    trend_data = data.iloc[-window:]
    
    # Create x values (time index)
    x = np.arange(len(trend_data))
    y = trend_data.values
    
    # Calculate slope using numpy polyfit
    try:
        slope, _ = np.polyfit(x, y.astype(float), 1)
        return float(slope)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def detect_false_support_break(prices: pd.Series, 
                             support_level: float, 
                             break_threshold: float = 0.01,
                             recovery_days: int = 3) -> Tuple[bool, int]:
    """
    Detect if there was a false break below support level with subsequent recovery.
    
    Args:
        prices: Price series (typically Close prices)
        support_level: Support level to check against
        break_threshold: Minimum percentage break below support (default: 1%)
        recovery_days: Maximum days for recovery (default: 3)
        
    Returns:
        Tuple of (is_false_break, recovery_day_index)
    """
    if len(prices) < recovery_days + 1:
        return False, -1
    
    # Calculate break level
    break_level = support_level * (1 - break_threshold)
    
    # Find if any price broke below the break level
    break_mask = prices < break_level
    
    if not break_mask.any():
        return False, -1
    
    # Find the first break
    break_idx = break_mask.idxmax()
    break_loc = prices.index.get_loc(break_idx)
    break_position = int(break_loc) if isinstance(break_loc, (int, np.integer)) else int(break_loc.start) if isinstance(break_loc, slice) else int(break_loc[0])
    
    # Check for recovery within the specified days
    recovery_end = min(break_position + recovery_days + 1, len(prices))
    recovery_period = prices.iloc[break_position:recovery_end]
    
    # Check if price recovered above support level
    recovery_mask = recovery_period >= support_level
    
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        recovery_loc = prices.index.get_loc(recovery_idx)
        recovery_position = int(recovery_loc) if isinstance(recovery_loc, (int, np.integer)) else int(recovery_loc.start) if isinstance(recovery_loc, slice) else int(recovery_loc[0])
        return True, recovery_position - break_position
    
    return False, -1


def calculate_drawdown_metrics(prices: pd.Series) -> Tuple[float, float]:
    """
    Calculate maximum drawdown and recovery return.
    
    Args:
        prices: Price series
        
    Returns:
        Tuple of (max_drawdown_pct, recovery_return_pct)
    """
    if len(prices) < 2:
        return 0.0, 0.0
    
    # Calculate running maximum (peak)
    running_max = prices.expanding().max()
    
    # Calculate drawdown
    drawdown = (prices - running_max) / running_max
    
    # Maximum drawdown (most negative value)
    max_drawdown_pct = float(drawdown.min() * 100)  # Convert to percentage
    
    # Recovery return (from lowest point to end)
    lowest_price = prices.min()
    final_price = prices.iloc[-1]
    recovery_return_pct = float(((final_price - lowest_price) / lowest_price) * 100)
    
    return max_drawdown_pct, recovery_return_pct


def calculate_candle_patterns(open_prices: pd.Series, 
                            high_prices: pd.Series, 
                            low_prices: pd.Series, 
                            close_prices: pd.Series) -> pd.Series:
    """
    Calculate basic candle pattern indicators (red/green ratio).
    
    Args:
        open_prices: Open price series
        high_prices: High price series  
        low_prices: Low price series
        close_prices: Close price series
        
    Returns:
        pd.Series: Red candle indicator (1 for red, 0 for green/doji)
    """
    # Red candle: close < open
    red_candles = (close_prices < open_prices).astype(int)
    
    return red_candles 