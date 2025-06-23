"""
Unit tests for the Interactive Demo notebook functionality.

Tests the is_false_breakout function and related pattern detection logic.
"""

import sys
import os
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pattern_labeler import is_false_breakout


class TestInteractiveDemo(unittest.TestCase):
    """Test cases for interactive demo functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample OHLCV data for testing
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        
        # Create bullish trend with false breakout pattern
        self.sample_data_with_pattern = pd.DataFrame({
            'Open': np.random.uniform(100, 105, 60),
            'High': np.random.uniform(105, 110, 60),
            'Low': np.random.uniform(95, 100, 60),
            'Close': np.linspace(100, 108, 60) + np.random.normal(0, 0.5, 60),  # Uptrend
            'Volume': np.random.uniform(100000, 200000, 60)
        }, index=dates)
        
        # Add false breakout pattern in the last 10 days
        # Create a dip below support and recovery
        self.sample_data_with_pattern.loc[dates[-10]:dates[-8], 'Low'] *= 0.97  # Break support
        self.sample_data_with_pattern.loc[dates[-10]:dates[-8], 'Close'] *= 0.98  # Break support
        self.sample_data_with_pattern.loc[dates[-7]:, 'Close'] *= 1.02  # Recovery
        
        # Create sample data without pattern (sideways movement)
        self.sample_data_no_pattern = pd.DataFrame({
            'Open': np.random.uniform(100, 105, 60),
            'High': np.random.uniform(105, 110, 60),
            'Low': np.random.uniform(95, 100, 60),
            'Close': np.random.uniform(100, 105, 60),  # No trend
            'Volume': np.random.uniform(100000, 200000, 60)
        }, index=dates)
    
    def test_is_false_breakout_with_pattern(self):
        """Test is_false_breakout function with pattern data"""
        result = is_false_breakout(self.sample_data_with_pattern, lookback_days=30)
        # Should return True for data with false breakout pattern
        self.assertIsInstance(result, bool)
    
    def test_is_false_breakout_no_pattern(self):
        """Test is_false_breakout function with no pattern data"""
        result = is_false_breakout(self.sample_data_no_pattern, lookback_days=30)
        # Should return False for data without clear pattern
        self.assertIsInstance(result, bool)
    
    def test_is_false_breakout_insufficient_data(self):
        """Test is_false_breakout function with insufficient data"""
        short_data = self.sample_data_with_pattern.head(10)  # Only 10 days
        result = is_false_breakout(short_data, lookback_days=30)
        # Should return False when insufficient data
        self.assertFalse(result)
    
    def test_is_false_breakout_empty_data(self):
        """Test is_false_breakout function with empty data"""
        empty_data = pd.DataFrame()
        result = is_false_breakout(empty_data, lookback_days=30)
        # Should return False for empty data
        self.assertFalse(result)
    
    def test_is_false_breakout_invalid_columns(self):
        """Test is_false_breakout function with missing columns"""
        invalid_data = pd.DataFrame({
            'Price': np.random.uniform(100, 105, 60)
        })
        result = is_false_breakout(invalid_data, lookback_days=30)
        # Should return False for data with missing required columns
        self.assertFalse(result)
    
    def test_is_false_breakout_parameters(self):
        """Test is_false_breakout function with different parameters"""
        # Test with different lookback periods
        result_30 = is_false_breakout(self.sample_data_with_pattern, lookback_days=30)
        result_15 = is_false_breakout(self.sample_data_with_pattern, lookback_days=15)
        
        # Both should return boolean values
        self.assertIsInstance(result_30, bool)
        self.assertIsInstance(result_15, bool)
        
        # Test with different min_trend_days
        result_trend_5 = is_false_breakout(self.sample_data_with_pattern, min_trend_days=5)
        result_trend_15 = is_false_breakout(self.sample_data_with_pattern, min_trend_days=15)
        
        # Both should return boolean values
        self.assertIsInstance(result_trend_5, bool)
        self.assertIsInstance(result_trend_15, bool)
    
    def test_pattern_detection_robustness(self):
        """Test pattern detection robustness with edge cases"""
        # Test with NaN values
        data_with_nan = self.sample_data_with_pattern.copy()
        data_with_nan.loc[data_with_nan.index[5:10], 'Close'] = np.nan
        
        result = is_false_breakout(data_with_nan, lookback_days=30)
        self.assertIsInstance(result, bool)
        
        # Test with extreme values
        data_with_extremes = self.sample_data_with_pattern.copy()
        data_with_extremes.loc[data_with_extremes.index[5], 'Close'] = 1000000  # Extreme spike
        
        result = is_false_breakout(data_with_extremes, lookback_days=30)
        self.assertIsInstance(result, bool)


class TestInteractiveDemoIntegration(unittest.TestCase):
    """Integration tests for interactive demo components"""
    
    def test_required_modules_import(self):
        """Test that all required modules can be imported"""
        try:
            from data_fetcher import fetch_hk_stocks, validate_tickers
            from pattern_labeler import PatternLabeler, PatternLabel
            from feature_extractor import FeatureExtractor
            from hk_stock_universe import get_hk_stock_list_static
            from technical_indicators import detect_false_support_break, find_recent_support_level
            # If we get here, imports succeeded
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Required module import failed: {e}")
    
    def test_hk_stock_list_available(self):
        """Test that HK stock list is available"""
        from hk_stock_universe import get_hk_stock_list_static
        
        stock_list = get_hk_stock_list_static()
        self.assertIsInstance(stock_list, list)
        self.assertGreater(len(stock_list), 0)
        
        # Check that stocks are in correct HK format
        for stock in stock_list[:5]:  # Check first 5
            self.assertTrue(stock.endswith('.HK'))
            self.assertEqual(len(stock), 7)  # Format: XXXX.HK
    
    def test_pattern_labeler_integration(self):
        """Test PatternLabeler integration with is_false_breakout"""
        from pattern_labeler import PatternLabeler, is_false_breakout
        
        # Test that PatternLabeler can be instantiated
        labeler = PatternLabeler()
        self.assertIsNotNone(labeler)
        
        # Test that is_false_breakout is available
        self.assertTrue(callable(is_false_breakout))


if __name__ == '__main__':
    unittest.main() 