"""
Unit tests for technical indicators module.

Tests all technical indicator calculations with various edge cases
and validates the mathematical correctness of the implementations.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

# Handle imports for both direct execution and package usage
try:
    from src.technical_indicators import (
        simple_moving_average,
        exponential_moving_average,
        relative_strength_index,
        macd,
        bollinger_bands,
        average_true_range,
        price_volatility,
        volume_average_ratio,
        find_support_resistance_levels,
        find_recent_support_level,
        calculate_linear_trend_slope,
        detect_false_support_break,
        calculate_drawdown_metrics,
        calculate_candle_patterns,
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.technical_indicators import (
        simple_moving_average,
        exponential_moving_average,
        relative_strength_index,
        macd,
        bollinger_bands,
        average_true_range,
        price_volatility,
        volume_average_ratio,
        find_support_resistance_levels,
        find_recent_support_level,
        calculate_linear_trend_slope,
        detect_false_support_break,
        calculate_drawdown_metrics,
        calculate_candle_patterns,
    )


class TestTechnicalIndicators:
    """Test class for technical indicators."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        # Create realistic price data with trend and volatility
        base_price = 100
        trend = np.linspace(0, 10, 50)
        noise = np.random.normal(0, 2, 50)
        prices = base_price + trend + noise
        
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        
        # Generate realistic OHLCV data
        close_prices = np.random.uniform(95, 105, 30)
        high_prices = close_prices + np.random.uniform(0, 5, 30)
        low_prices = close_prices - np.random.uniform(0, 5, 30)
        open_prices = close_prices + np.random.uniform(-2, 2, 30)
        volumes = np.random.uniform(1000000, 5000000, 30)
        
        return pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)
    
    def test_simple_moving_average(self, sample_price_data):
        """Test simple moving average calculation."""
        # Test normal case
        sma_10 = simple_moving_average(sample_price_data, 10)
        assert len(sma_10) == len(sample_price_data)
        assert not sma_10.iloc[:9].notna().any()  # First 9 values should be NaN
        assert sma_10.iloc[9:].notna().all()  # Rest should be valid
        
        # Test with insufficient data
        short_data = sample_price_data.iloc[:5]
        sma_10_short = simple_moving_average(short_data, 10)
        assert sma_10_short.empty or sma_10_short.isna().all()
        
        # Test edge case with window size 1
        sma_1 = simple_moving_average(sample_price_data, 1)
        pd.testing.assert_series_equal(sma_1, sample_price_data)
    
    def test_exponential_moving_average(self, sample_price_data):
        """Test exponential moving average calculation."""
        ema_10 = exponential_moving_average(sample_price_data, 10)
        assert len(ema_10) == len(sample_price_data)
        assert ema_10.notna().all()
        
        # EMA should be smoother than raw data
        ema_volatility = ema_10.pct_change().std()
        price_volatility = sample_price_data.pct_change().std()
        assert ema_volatility < price_volatility
    
    def test_relative_strength_index(self, sample_price_data):
        """Test RSI calculation."""
        rsi = relative_strength_index(sample_price_data, 14)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
        
        # Test with insufficient data
        short_data = sample_price_data.iloc[:10]
        rsi_short = relative_strength_index(short_data, 14)
        assert rsi_short.empty or rsi_short.isna().all()
    
    def test_macd(self, sample_price_data):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = macd(sample_price_data)
        
        assert len(macd_line) == len(sample_price_data)
        assert len(signal_line) == len(sample_price_data)
        assert len(histogram) == len(sample_price_data)
        
        # Histogram should be difference between MACD and signal
        calculated_histogram = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, calculated_histogram, check_names=False)
    
    def test_bollinger_bands(self, sample_price_data):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = bollinger_bands(sample_price_data, 20, 2.0)
        
        # Upper band should be above middle, middle above lower
        valid_data = middle.dropna()
        upper_valid = upper.loc[valid_data.index]
        lower_valid = lower.loc[valid_data.index]
        
        assert (upper_valid >= valid_data).all()
        assert (valid_data >= lower_valid).all()
    
    def test_average_true_range(self, sample_ohlcv_data):
        """Test ATR calculation."""
        atr = average_true_range(
            sample_ohlcv_data['High'],
            sample_ohlcv_data['Low'],
            sample_ohlcv_data['Close'],
            14
        )
        
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()
    
    def test_price_volatility(self, sample_price_data):
        """Test price volatility calculation."""
        volatility = price_volatility(sample_price_data, 20)
        
        # Volatility should be positive
        valid_vol = volatility.dropna()
        assert (valid_vol >= 0).all()
    
    def test_volume_average_ratio(self, sample_ohlcv_data):
        """Test volume average ratio calculation."""
        vol_ratio = volume_average_ratio(sample_ohlcv_data['Volume'], 10)
        
        # Ratio should be positive
        valid_ratio = vol_ratio.dropna()
        assert (valid_ratio > 0).all()
    
    def test_find_recent_support_level(self, sample_ohlcv_data):
        """Test recent support level detection."""
        low_prices = sample_ohlcv_data['Low']
        
        # Test normal case
        support = find_recent_support_level(low_prices, 20, 10)
        assert support is not None
        assert isinstance(support, float)
        
        # Test edge cases
        support_start = find_recent_support_level(low_prices, 5, 10)
        assert support_start is not None
        
        support_insufficient = find_recent_support_level(low_prices, 0, 10)
        assert support_insufficient is None
    
    def test_calculate_linear_trend_slope(self, sample_price_data):
        """Test linear trend slope calculation."""
        # Test with upward trending data
        upward_trend = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        slope_up = calculate_linear_trend_slope(upward_trend, 10)
        assert slope_up > 0
        
        # Test with downward trending data
        downward_trend = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        slope_down = calculate_linear_trend_slope(downward_trend, 10)
        assert slope_down < 0
        
        # Test with insufficient data
        slope_insufficient = calculate_linear_trend_slope(sample_price_data.iloc[:5], 10)
        assert slope_insufficient == 0.0
    
    def test_detect_false_support_break(self, sample_ohlcv_data):
        """Test false support break detection."""
        # Create test data with known false break
        test_prices = pd.Series([100, 101, 99, 98, 97, 99, 100, 101, 102])
        support_level = 98.5
        
        is_false_break, recovery_days = detect_false_support_break(
            test_prices, support_level, 0.01, 3
        )
        
        assert isinstance(is_false_break, bool)
        assert isinstance(recovery_days, int)
        
        # Test with no break
        no_break_prices = pd.Series([100, 101, 102, 103, 104])
        is_false_no_break, _ = detect_false_support_break(
            no_break_prices, 99, 0.01, 3
        )
        assert is_false_no_break == False
    
    def test_calculate_drawdown_metrics(self, sample_price_data):
        """Test drawdown metrics calculation."""
        drawdown_pct, recovery_pct = calculate_drawdown_metrics(sample_price_data)
        
        assert isinstance(drawdown_pct, float)
        assert isinstance(recovery_pct, float)
        assert drawdown_pct <= 0  # Drawdown should be negative or zero
        
        # Test with insufficient data
        short_data = sample_price_data.iloc[:1]
        dd_short, rec_short = calculate_drawdown_metrics(short_data)
        assert dd_short == 0.0
        assert rec_short == 0.0
    
    def test_calculate_candle_patterns(self, sample_ohlcv_data):
        """Test candle pattern calculation."""
        red_candles = calculate_candle_patterns(
            sample_ohlcv_data['Open'],
            sample_ohlcv_data['High'],
            sample_ohlcv_data['Low'],
            sample_ohlcv_data['Close']
        )
        
        # Should return binary values (0 or 1)
        assert ((red_candles == 0) | (red_candles == 1)).all()
        assert len(red_candles) == len(sample_ohlcv_data)
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Empty data
        empty_series = pd.Series([], dtype=float)
        sma_empty = simple_moving_average(empty_series, 10)
        assert sma_empty.empty
        
        # Single data point
        single_point = pd.Series([100])
        sma_single = simple_moving_average(single_point, 10)
        assert sma_single.empty or sma_single.isna().all()
        
        # Data with NaN values
        nan_data = pd.Series([100, np.nan, 102, 103, np.nan])
        sma_nan = simple_moving_average(nan_data, 3)
        # Should handle NaN gracefully
        assert len(sma_nan) == len(nan_data)


if __name__ == "__main__":
    pytest.main([__file__]) 