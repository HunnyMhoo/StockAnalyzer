"""
Unit tests for the data_fetcher module.

This module tests all public and private functions in the data fetcher,
using mocked Yahoo Finance calls to ensure reliability and speed.
"""

import os
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np

# Import the module to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_fetcher import (
    fetch_hk_stocks,
    validate_tickers,
    preview_cached_data,
    list_cached_tickers,
    _validate_hk_ticker,
    _get_cache_filename,
    _load_cached_data,
    _save_data_to_cache,
    _fetch_from_yahoo,
    _determine_date_range,
    _merge_with_cache,
    DATA_DIR
)


class TestTickerValidation:
    """Test ticker validation functions."""
    
    def test_validate_hk_ticker_valid(self):
        """Test valid HK ticker formats."""
        valid_tickers = ['0700.HK', '0005.HK', '1234.HK']
        for ticker in valid_tickers:
            assert _validate_hk_ticker(ticker), f"{ticker} should be valid"
    
    def test_validate_hk_ticker_invalid(self):
        """Test invalid ticker formats."""
        invalid_tickers = [
            '700.HK',     # Missing leading zero
            '07000.HK',   # Too many digits
            '0700.HKG',   # Wrong suffix
            '0700',       # Missing .HK
            'AAPL',       # US ticker
            '0700.US',    # Wrong exchange
            '',           # Empty string
            '0700.hk',    # Lowercase
        ]
        for ticker in invalid_tickers:
            assert not _validate_hk_ticker(ticker), f"{ticker} should be invalid"
    
    def test_validate_tickers_mixed(self):
        """Test validation with mixed valid/invalid tickers."""
        tickers = ['0700.HK', 'INVALID', '0005.HK', '123.HK']
        valid, invalid = validate_tickers(tickers)
        
        assert valid == ['0700.HK', '0005.HK']
        assert invalid == ['INVALID', '123.HK']
    
    def test_validate_tickers_empty(self):
        """Test validation with empty list."""
        valid, invalid = validate_tickers([])
        assert valid == []
        assert invalid == []


class TestCacheManagement:
    """Test cache management functions."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_data_dir = DATA_DIR
        # Monkey patch DATA_DIR for testing
        import data_fetcher
        data_fetcher.DATA_DIR = self.temp_dir
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        # Restore original DATA_DIR
        import data_fetcher
        data_fetcher.DATA_DIR = self.original_data_dir
    
    def test_get_cache_filename(self):
        """Test cache filename generation."""
        filename = _get_cache_filename('0700.HK')
        expected = os.path.join(self.temp_dir, '0700_HK.csv')
        assert filename == expected
    
    def test_save_and_load_cached_data(self):
        """Test saving and loading cached data."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        sample_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Adj Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
        
        # Save data
        _save_data_to_cache('0700.HK', sample_data)
        
        # Load data
        loaded_data = _load_cached_data('0700.HK')
        
        # Verify data integrity
        assert loaded_data is not None
        assert len(loaded_data) == 5
        assert list(loaded_data.columns) == ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        pd.testing.assert_frame_equal(loaded_data, sample_data)
    
    def test_load_nonexistent_cache(self):
        """Test loading cache for non-existent ticker."""
        result = _load_cached_data('9999.HK')
        assert result is None
    
    def test_merge_with_cache(self):
        """Test merging new data with cached data."""
        # Create cached data
        cached_dates = pd.date_range('2023-01-01', periods=3, freq='D')
        cached_data = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0]
        }, index=cached_dates)
        
        # Create new data with overlap
        new_dates = pd.date_range('2023-01-03', periods=3, freq='D')
        new_data = pd.DataFrame({
            'Close': [102.5, 103.0, 104.0]  # Overlapping date should be overwritten
        }, index=new_dates)
        
        # Merge data
        merged = _merge_with_cache('0700.HK', new_data, cached_data)
        
        # Verify merge
        assert len(merged) == 5  # No duplicates
        assert merged.loc['2023-01-03', 'Close'] == 102.5  # New data overwrites cached


class TestDateRangeLogic:
    """Test date range determination logic."""
    
    def test_determine_date_range_no_cache(self):
        """Test date range when no cache exists."""
        start, end, needs_fetch = _determine_date_range(None, '2023-01-01', '2023-12-31')
        
        assert start == '2023-01-01'
        assert end == '2023-12-31'
        assert needs_fetch is True
    
    def test_determine_date_range_cache_covers_range(self):
        """Test when cache covers requested range."""
        # Create cached data that covers requested range
        dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
        cached_data = pd.DataFrame({'Close': range(len(dates))}, index=dates)
        
        start, end, needs_fetch = _determine_date_range(
            cached_data, '2023-01-01', '2023-12-31'
        )
        
        assert needs_fetch is False
    
    def test_determine_date_range_need_recent_data(self):
        """Test when we need more recent data."""
        # Create cached data that ends before requested end
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
        cached_data = pd.DataFrame({'Close': range(len(dates))}, index=dates)
        
        start, end, needs_fetch = _determine_date_range(
            cached_data, '2023-01-01', '2023-12-31'
        )
        
        assert needs_fetch is True
        assert start == '2023-07-01'  # Day after cached end
        assert end == '2023-12-31'


class TestYahooFinanceIntegration:
    """Test Yahoo Finance API integration."""
    
    @patch('data_fetcher.yf.download')
    def test_fetch_from_yahoo_success(self, mock_download):
        """Test successful fetch from Yahoo Finance."""
        # Create mock data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        mock_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Adj Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
        
        mock_download.return_value = mock_data
        
        result = _fetch_from_yahoo('0700.HK', '2023-01-01', '2023-01-05')
        
        assert result is not None
        assert len(result) == 5
        mock_download.assert_called_once_with(
            '0700.HK',
            start='2023-01-01',
            end='2023-01-05',
            progress=False,
            show_errors=False
        )
    
    @patch('data_fetcher.yf.download')
    def test_fetch_from_yahoo_empty_data(self, mock_download):
        """Test handling of empty data from Yahoo Finance."""
        mock_download.return_value = pd.DataFrame()
        
        result = _fetch_from_yahoo('0700.HK', '2023-01-01', '2023-01-05')
        assert result is None
    
    @patch('data_fetcher.yf.download')
    def test_fetch_from_yahoo_with_retry(self, mock_download):
        """Test retry logic on API failure."""
        # First two calls fail, third succeeds
        mock_download.side_effect = [
            Exception("API Error"),
            Exception("API Error"),
            pd.DataFrame({
                'Open': [100.0], 'High': [105.0], 'Low': [95.0],
                'Close': [102.0], 'Adj Close': [102.0], 'Volume': [1000000]
            }, index=pd.date_range('2023-01-01', periods=1))
        ]
        
        result = _fetch_from_yahoo('0700.HK', '2023-01-01', '2023-01-01')
        
        assert result is not None
        assert len(result) == 1
        assert mock_download.call_count == 3


class TestMainFetchFunction:
    """Test the main fetch_hk_stocks function."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_data_dir = DATA_DIR
        # Monkey patch DATA_DIR for testing
        import data_fetcher
        data_fetcher.DATA_DIR = self.temp_dir
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        # Restore original DATA_DIR
        import data_fetcher
        data_fetcher.DATA_DIR = self.original_data_dir
    
    @patch('data_fetcher._fetch_from_yahoo')
    def test_fetch_hk_stocks_success(self, mock_fetch):
        """Test successful data fetching for multiple tickers."""
        # Mock successful fetch
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        mock_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Adj Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
        
        mock_fetch.return_value = mock_data
        
        result = fetch_hk_stocks(['0700.HK', '0005.HK'], '2023-01-01', '2023-01-05')
        
        assert len(result) == 2
        assert '0700.HK' in result
        assert '0005.HK' in result
        assert len(result['0700.HK']) == 5
        assert len(result['0005.HK']) == 5
    
    def test_fetch_hk_stocks_invalid_tickers(self):
        """Test handling of invalid tickers."""
        result = fetch_hk_stocks(['INVALID', '123.HK'], '2023-01-01', '2023-01-05')
        assert len(result) == 0
    
    @patch('data_fetcher._fetch_from_yahoo')
    def test_fetch_hk_stocks_mixed_success_failure(self, mock_fetch):
        """Test handling when some tickers succeed and others fail."""
        def mock_fetch_side_effect(ticker, start, end):
            if ticker == '0700.HK':
                dates = pd.date_range(start, periods=5, freq='D')
                return pd.DataFrame({
                    'Open': [100.0] * 5, 'High': [105.0] * 5, 'Low': [95.0] * 5,
                    'Close': [102.0] * 5, 'Adj Close': [102.0] * 5, 'Volume': [1000000] * 5
                }, index=dates)
            else:
                return None  # Simulate failure for other tickers
        
        mock_fetch.side_effect = mock_fetch_side_effect
        
        result = fetch_hk_stocks(['0700.HK', '0005.HK'], '2023-01-01', '2023-01-05')
        
        assert len(result) == 1
        assert '0700.HK' in result
        assert '0005.HK' not in result


class TestUtilityFunctions:
    """Test utility functions for notebook usage."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_data_dir = DATA_DIR
        # Monkey patch DATA_DIR for testing
        import data_fetcher
        data_fetcher.DATA_DIR = self.temp_dir
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        # Restore original DATA_DIR
        import data_fetcher
        data_fetcher.DATA_DIR = self.original_data_dir
    
    def test_preview_cached_data_exists(self, capsys):
        """Test preview function with existing cached data."""
        # Create and save sample data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        sample_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Adj Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
        
        _save_data_to_cache('0700.HK', sample_data)
        
        # Test preview
        preview_cached_data('0700.HK')
        
        captured = capsys.readouterr()
        assert "📊 Cached data for 0700.HK:" in captured.out
        assert "📅 Date range:" in captured.out
        assert "📈 Records: 5" in captured.out
    
    def test_preview_cached_data_not_exists(self, capsys):
        """Test preview function with non-existent data."""
        preview_cached_data('9999.HK')
        
        captured = capsys.readouterr()
        assert "📁 No cached data found for 9999.HK" in captured.out
    
    def test_preview_cached_data_invalid_ticker(self, capsys):
        """Test preview function with invalid ticker."""
        preview_cached_data('INVALID')
        
        captured = capsys.readouterr()
        assert "❌ Invalid ticker format: INVALID" in captured.out
    
    def test_list_cached_tickers_empty(self, capsys):
        """Test listing tickers when cache is empty."""
        list_cached_tickers()
        
        captured = capsys.readouterr()
        assert "📁 No cached data found" in captured.out
    
    def test_list_cached_tickers_with_data(self, capsys):
        """Test listing tickers with cached data."""
        # Create sample data for two tickers
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        sample_data = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0, 103.0, 104.0]
        }, index=dates)
        
        _save_data_to_cache('0700.HK', sample_data)
        _save_data_to_cache('0005.HK', sample_data)
        
        list_cached_tickers()
        
        captured = capsys.readouterr()
        assert "📊 Cached tickers:" in captured.out
        assert "0700.HK" in captured.out
        assert "0005.HK" in captured.out


if __name__ == '__main__':
    pytest.main([__file__]) 