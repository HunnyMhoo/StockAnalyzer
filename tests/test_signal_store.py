"""
Unit tests for the SignalStore module.

Tests cover signal storage, retrieval, validation, and error handling.
"""

import os
import pytest
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from stock_analyzer.data.signal_store import (
    SignalStore,
    SignalRecord,
    SignalQueryResult,
    SignalStoreError,
    convert_scanner_results_to_signals,
    write_daily_signals,
    query_signals
)


class TestSignalRecord:
    """Test SignalRecord validation."""
    
    def test_valid_signal_record(self):
        """Test creating a valid signal record."""
        record = SignalRecord(
            date=datetime.now(),
            ticker="0700.HK",
            pattern_id="bear_flag",
            confidence=0.85,
            window_start=datetime.now() - timedelta(days=30),
            window_end=datetime.now() - timedelta(days=1)
        )
        
        assert record.ticker == "0700.HK"
        assert record.pattern_id == "bear_flag"
        assert 0.0 <= record.confidence <= 1.0
    
    def test_ticker_validation(self):
        """Test ticker validation and normalization."""
        # Valid ticker should be normalized to uppercase
        record = SignalRecord(
            date=datetime.now(),
            ticker="  0700.hk  ",
            pattern_id="bear_flag",
            confidence=0.85
        )
        assert record.ticker == "0700.HK"
        
        # Empty ticker should raise error
        with pytest.raises(ValueError, match="Ticker cannot be empty"):
            SignalRecord(
                date=datetime.now(),
                ticker="",
                pattern_id="bear_flag",
                confidence=0.85
            )
    
    def test_pattern_id_validation(self):
        """Test pattern ID validation and normalization."""
        # Valid pattern ID should be normalized to lowercase
        record = SignalRecord(
            date=datetime.now(),
            ticker="0700.HK",
            pattern_id="  BEAR_FLAG  ",
            confidence=0.85
        )
        assert record.pattern_id == "bear_flag"
        
        # Empty pattern ID should raise error
        with pytest.raises(ValueError, match="Pattern ID cannot be empty"):
            SignalRecord(
                date=datetime.now(),
                ticker="0700.HK",
                pattern_id="",
                confidence=0.85
            )
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence scores
        for confidence in [0.0, 0.5, 1.0]:
            record = SignalRecord(
                date=datetime.now(),
                ticker="0700.HK",
                pattern_id="bear_flag",
                confidence=confidence
            )
            assert record.confidence == confidence
        
        # Invalid confidence scores should raise error
        for invalid_confidence in [-0.1, 1.1, 2.0]:
            with pytest.raises(ValueError):
                SignalRecord(
                    date=datetime.now(),
                    ticker="0700.HK",
                    pattern_id="bear_flag",
                    confidence=invalid_confidence
                )


class TestSignalStore:
    """Test SignalStore functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def signal_store(self, temp_dir):
        """Create a SignalStore instance for testing."""
        return SignalStore(base_dir=temp_dir, auto_create_dirs=True)
    
    @pytest.fixture
    def sample_signals_df(self):
        """Create sample signals DataFrame for testing."""
        data = [
            {
                'date': datetime.now(),
                'ticker': '0700.HK',
                'pattern_id': 'bear_flag',
                'confidence': 0.85,
                'window_start': datetime.now() - timedelta(days=30),
                'window_end': datetime.now() - timedelta(days=1)
            },
            {
                'date': datetime.now(),
                'ticker': '0388.HK',
                'pattern_id': 'bull_flag',
                'confidence': 0.92,
                'window_start': datetime.now() - timedelta(days=25),
                'window_end': datetime.now() - timedelta(days=2)
            },
            {
                'date': datetime.now(),
                'ticker': '0005.HK',
                'pattern_id': 'bear_flag',
                'confidence': 0.78,
                'window_start': datetime.now() - timedelta(days=20),
                'window_end': datetime.now() - timedelta(days=3)
            }
        ]
        return pd.DataFrame(data)
    
    def test_initialization(self, temp_dir):
        """Test SignalStore initialization."""
        store = SignalStore(base_dir=temp_dir, auto_create_dirs=True)
        
        assert store.base_dir == Path(temp_dir)
        assert store.enable_compression is True
        assert (Path(temp_dir) / "daily").exists()
        assert (Path(temp_dir) / "archive").exists()
    
    def test_write_signals_success(self, signal_store, sample_signals_df):
        """Test successful signal writing."""
        test_date = datetime.now().date()
        
        file_path = signal_store.write_signals(sample_signals_df, date=datetime.combine(test_date, datetime.min.time()))
        
        assert os.path.exists(file_path)
        
        # Verify file content
        saved_df = pd.read_parquet(file_path)
        assert len(saved_df) == len(sample_signals_df)
        assert 'ticker' in saved_df.columns
        assert 'pattern_id' in saved_df.columns
        assert 'confidence' in saved_df.columns
    
    def test_write_signals_with_invalid_data(self, signal_store):
        """Test writing signals with invalid data."""
        # DataFrame with missing required columns
        invalid_df = pd.DataFrame([
            {'ticker': '0700.HK', 'confidence': 0.85}  # Missing pattern_id
        ])
        
        with pytest.raises(SignalStoreError, match="Missing required columns"):
            signal_store.write_signals(invalid_df)
    
    def test_write_signals_append_mode(self, signal_store, sample_signals_df):
        """Test appending signals to existing file."""
        test_date = datetime.now().date()
        
        # Write initial signals
        file_path = signal_store.write_signals(sample_signals_df, date=datetime.combine(test_date, datetime.min.time()))
        
        # Create additional signals
        additional_signals = pd.DataFrame([
            {
                'date': datetime.now(),
                'ticker': '0001.HK',
                'pattern_id': 'triangle',
                'confidence': 0.88,
                'window_start': datetime.now() - timedelta(days=15),
                'window_end': datetime.now() - timedelta(days=1)
            }
        ])
        
        # Append to existing file
        signal_store.write_signals(additional_signals, date=datetime.combine(test_date, datetime.min.time()), overwrite=False)
        
        # Verify combined content
        saved_df = pd.read_parquet(file_path)
        assert len(saved_df) == len(sample_signals_df) + len(additional_signals)
    
    def test_read_signals_basic(self, signal_store, sample_signals_df):
        """Test basic signal reading."""
        test_date = datetime.now().date()
        
        # Write signals first
        signal_store.write_signals(sample_signals_df, date=datetime.combine(test_date, datetime.min.time()))
        
        # Read signals back
        result = signal_store.read_signals()
        
        assert isinstance(result, SignalQueryResult)
        assert len(result.signals_df) > 0
        assert result.total_records > 0
        assert len(result.patterns_found) > 0
        assert len(result.tickers_found) > 0
    
    def test_read_signals_with_filters(self, signal_store, sample_signals_df):
        """Test reading signals with various filters."""
        test_date = datetime.now().date()
        
        # Write signals first
        signal_store.write_signals(sample_signals_df, date=datetime.combine(test_date, datetime.min.time()))
        
        # Test pattern ID filter
        result = signal_store.read_signals(pattern_id="bear_flag")
        bear_flag_signals = result.signals_df
        assert all(bear_flag_signals['pattern_id'] == 'bear_flag')
        
        # Test confidence filter
        result = signal_store.read_signals(min_confidence=0.90)
        high_conf_signals = result.signals_df
        assert all(high_conf_signals['confidence'] >= 0.90)
        
        # Test ticker filter
        result = signal_store.read_signals(tickers=['0700.HK'])
        ticker_signals = result.signals_df
        assert all(ticker_signals['ticker'] == '0700.HK')
        
        # Test limit
        result = signal_store.read_signals(limit=1)
        assert len(result.signals_df) <= 1
    
    def test_read_signals_date_range(self, signal_store, sample_signals_df):
        """Test reading signals with date range filtering."""
        test_date = datetime.now().date()
        
        # Write signals first
        signal_store.write_signals(sample_signals_df, date=datetime.combine(test_date, datetime.min.time()))
        
        # Test date range
        start_date = datetime.combine(test_date, datetime.min.time()) - timedelta(days=1)
        end_date = datetime.combine(test_date, datetime.min.time()) + timedelta(days=1)
        
        result = signal_store.read_signals(start_date=start_date, end_date=end_date)
        assert len(result.signals_df) > 0
        
        # Test narrow date range (should return empty)
        past_start = datetime.combine(test_date, datetime.min.time()) - timedelta(days=10)
        past_end = datetime.combine(test_date, datetime.min.time()) - timedelta(days=5)
        
        result = signal_store.read_signals(start_date=past_start, end_date=past_end)
        assert len(result.signals_df) == 0
    
    def test_read_signals_no_data(self, signal_store):
        """Test reading signals when no data exists."""
        result = signal_store.read_signals()
        
        assert isinstance(result, SignalQueryResult)
        assert len(result.signals_df) == 0
        assert result.total_records == 0
        assert len(result.patterns_found) == 0
        assert len(result.tickers_found) == 0
    
    def test_get_available_patterns(self, signal_store, sample_signals_df):
        """Test getting available patterns."""
        test_date = datetime.now().date()
        
        # Initially should be empty
        patterns = signal_store.get_available_patterns()
        assert len(patterns) == 0
        
        # Write signals and check patterns
        signal_store.write_signals(sample_signals_df, date=datetime.combine(test_date, datetime.min.time()))
        patterns = signal_store.get_available_patterns()
        
        expected_patterns = ['bear_flag', 'bull_flag']
        assert sorted(patterns) == sorted(expected_patterns)
    
    def test_get_date_range(self, signal_store, sample_signals_df):
        """Test getting available date range."""
        # Initially should return None, None
        start_date, end_date = signal_store.get_date_range()
        assert start_date is None
        assert end_date is None
        
        # Write signals and check date range
        test_date = datetime.now().date()
        signal_store.write_signals(sample_signals_df, date=datetime.combine(test_date, datetime.min.time()))
        
        start_date, end_date = signal_store.get_date_range()
        assert start_date is not None
        assert end_date is not None
        assert start_date <= end_date
    
    def test_get_statistics(self, signal_store, sample_signals_df):
        """Test getting storage statistics."""
        # Initially should return empty stats
        stats = signal_store.get_statistics()
        assert stats['total_signals'] == 0
        
        # Write signals and check stats
        test_date = datetime.now().date()
        signal_store.write_signals(sample_signals_df, date=datetime.combine(test_date, datetime.min.time()))
        
        stats = signal_store.get_statistics(days=30)
        assert stats['total_signals'] == len(sample_signals_df)
        assert stats['unique_tickers'] > 0
        assert stats['unique_patterns'] > 0
        assert 0.0 <= stats['avg_confidence'] <= 1.0
        assert isinstance(stats['top_patterns'], dict)
        assert isinstance(stats['top_tickers'], dict)
    
    def test_validate_signals_dataframe(self, signal_store):
        """Test DataFrame validation functionality."""
        # Valid DataFrame
        valid_df = pd.DataFrame([
            {
                'date': datetime.now(),
                'ticker': '0700.HK',
                'pattern_id': 'bear_flag',
                'confidence': 0.85
            }
        ])
        
        validated_df = signal_store._validate_signals_dataframe(valid_df)
        assert len(validated_df) == 1
        assert validated_df['ticker'].iloc[0] == '0700.HK'
        assert validated_df['pattern_id'].iloc[0] == 'bear_flag'
        
        # Invalid DataFrame with out-of-range confidence
        invalid_df = pd.DataFrame([
            {
                'date': datetime.now(),
                'ticker': '0700.HK',
                'pattern_id': 'bear_flag',
                'confidence': 1.5  # Invalid confidence > 1.0
            }
        ])
        
        validated_df = signal_store._validate_signals_dataframe(invalid_df)
        assert len(validated_df) == 0  # Should be filtered out
    
    def test_file_path_generation(self, signal_store):
        """Test daily file path generation."""
        test_date = datetime(2024, 1, 15)
        file_path = signal_store._get_daily_file_path(test_date)
        
        expected_path = signal_store.base_dir / "daily" / "signals_20240115.parquet"
        assert file_path == expected_path


class TestConversionFunctions:
    """Test utility conversion functions."""
    
    def test_convert_scanner_results_to_signals(self):
        """Test converting scanner results to signals format."""
        # Mock scanner results
        mock_results = MagicMock()
        mock_results.matches_df = pd.DataFrame([
            {
                'ticker': '0700.HK',
                'confidence_score': 0.85,
                'window_start_date': '2024-01-01',
                'window_end_date': '2024-01-15'
            },
            {
                'ticker': '0388.HK',
                'confidence_score': 0.92,
                'window_start_date': '2024-01-05',
                'window_end_date': '2024-01-20'
            }
        ])
        
        # Convert to signals
        signals_df = convert_scanner_results_to_signals(
            mock_results, 
            pattern_id="test_pattern",
            signal_date=datetime(2024, 1, 25)
        )
        
        assert len(signals_df) == 2
        assert 'date' in signals_df.columns
        assert 'ticker' in signals_df.columns
        assert 'pattern_id' in signals_df.columns
        assert 'confidence' in signals_df.columns
        assert all(signals_df['pattern_id'] == 'test_pattern')
        assert all(signals_df['confidence'] >= 0.0)
        assert all(signals_df['confidence'] <= 1.0)
    
    def test_convert_empty_scanner_results(self):
        """Test converting empty scanner results."""
        mock_results = MagicMock()
        mock_results.matches_df = pd.DataFrame()
        
        signals_df = convert_scanner_results_to_signals(mock_results)
        assert len(signals_df) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_signals_df(self):
        """Create sample signals DataFrame for testing."""
        return pd.DataFrame([
            {
                'date': datetime.now(),
                'ticker': '0700.HK',
                'pattern_id': 'bear_flag',
                'confidence': 0.85
            }
        ])
    
    def test_write_daily_signals(self, temp_dir, sample_signals_df):
        """Test write_daily_signals convenience function."""
        file_path = write_daily_signals(
            sample_signals_df,
            pattern_id="test_pattern",
            base_dir=temp_dir
        )
        
        assert os.path.exists(file_path)
    
    def test_query_signals(self, temp_dir, sample_signals_df):
        """Test query_signals convenience function."""
        # Write signals first
        write_daily_signals(sample_signals_df, base_dir=temp_dir)
        
        # Query signals
        result = query_signals(base_dir=temp_dir)
        
        assert isinstance(result, SignalQueryResult)
        assert len(result.signals_df) > 0


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_invalid_base_directory(self):
        """Test SignalStore with invalid base directory."""
        # Test with read-only directory scenario
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(SignalStoreError, match="Failed to create directory structure"):
                SignalStore(base_dir="/invalid/path", auto_create_dirs=True)
    
    def test_write_signals_io_error(self, temp_dir):
        """Test write_signals with I/O error."""
        store = SignalStore(base_dir=temp_dir)
        
        sample_df = pd.DataFrame([
            {
                'date': datetime.now(),
                'ticker': '0700.HK',
                'pattern_id': 'bear_flag',
                'confidence': 0.85
            }
        ])
        
        # Mock parquet write to raise an exception
        with patch('pandas.DataFrame.to_parquet', side_effect=Exception("Write failed")):
            with pytest.raises(SignalStoreError, match="Failed to write signals"):
                store.write_signals(sample_df)
    
    def test_read_signals_with_corrupted_file(self, temp_dir):
        """Test reading signals with corrupted file."""
        store = SignalStore(base_dir=temp_dir)
        
        # Create a corrupted file
        daily_dir = Path(temp_dir) / "daily"
        daily_dir.mkdir(exist_ok=True)
        corrupted_file = daily_dir / "signals_20240101.parquet"
        
        with open(corrupted_file, 'w') as f:
            f.write("corrupted data")
        
        # Should handle corrupted file gracefully
        result = store.read_signals()
        assert isinstance(result, SignalQueryResult)
        assert len(result.signals_df) == 0


if __name__ == "__main__":
    pytest.main([__file__]) 