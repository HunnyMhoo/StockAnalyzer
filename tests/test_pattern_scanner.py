"""
Unit Tests for Pattern Scanner

This module tests the PatternScanner class functionality including:
- Model loading and validation
- Sliding window generation
- Feature extraction for unlabeled windows
- Confidence filtering and ranking
- File output and timestamping
- Error handling for missing data/invalid models
"""

import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pattern_scanner import (
    PatternScanner, ScanningConfig, ScanningResults, 
    PatternScanningError, scan_hk_stocks_for_patterns,
    DEFAULT_WINDOW_SIZE, DEFAULT_MIN_CONFIDENCE
)
from feature_extractor import FeatureExtractor


class TestScanningConfig:
    """Test the ScanningConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ScanningConfig()
        
        assert config.window_size == DEFAULT_WINDOW_SIZE
        assert config.min_confidence == DEFAULT_MIN_CONFIDENCE
        assert config.max_windows_per_ticker == 5
        assert config.top_matches_display == 5
        assert config.save_results is True
        assert config.output_filename is None
        assert config.include_feature_values is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ScanningConfig(
            window_size=20,
            min_confidence=0.8,
            max_windows_per_ticker=3,
            top_matches_display=10,
            save_results=False,
            output_filename="custom.csv",
            include_feature_values=True
        )
        
        assert config.window_size == 20
        assert config.min_confidence == 0.8
        assert config.max_windows_per_ticker == 3
        assert config.top_matches_display == 10
        assert config.save_results is False
        assert config.output_filename == "custom.csv"
        assert config.include_feature_values is True


class TestScanningResults:
    """Test the ScanningResults dataclass."""
    
    def test_results_creation(self):
        """Test creation of ScanningResults."""
        matches_df = pd.DataFrame({'ticker': ['0700.HK'], 'confidence_score': [0.8]})
        summary = {'total_tickers_scanned': 1}
        config = ScanningConfig()
        model_info = {'model_type': 'xgboost'}
        
        results = ScanningResults(
            matches_df=matches_df,
            scanning_summary=summary,
            config=config,
            model_info=model_info,
            scanning_time=10.5
        )
        
        assert len(results.matches_df) == 1
        assert results.scanning_summary['total_tickers_scanned'] == 1
        assert results.scanning_time == 10.5
        assert results.output_path is None


class TestPatternScanner:
    """Test the PatternScanner class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_model_package(self):
        """Create mock model package."""
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        return {
            'model': mock_model,
            'scaler': None,
            'feature_names': ['sma_5', 'rsi_14', 'volatility'],
            'config': Mock(model_type='xgboost'),
            'metadata': {'training_date': '2023-01-01', 'n_features': 3}
        }
    
    @pytest.fixture
    def sample_ticker_data(self):
        """Create sample ticker data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        }, index=dates)
        
        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        return data
    
    @patch('pattern_scanner.load_trained_model')
    def test_init_success(self, mock_load_model, mock_model_package, temp_dir):
        """Test successful PatternScanner initialization."""
        mock_load_model.return_value = mock_model_package
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        scanner = PatternScanner(model_path, signals_dir=temp_dir)
        
        assert scanner.model_path == model_path
        assert scanner.signals_dir == temp_dir
        assert scanner.model is not None
        assert scanner.feature_names == ['sma_5', 'rsi_14', 'volatility']
        mock_load_model.assert_called_once_with(model_path)
    
    @patch('pattern_scanner.load_trained_model')
    def test_init_model_load_failure(self, mock_load_model, temp_dir):
        """Test PatternScanner initialization with model load failure."""
        mock_load_model.side_effect = Exception("Model load failed")
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        with pytest.raises(PatternScanningError, match="Failed to load model"):
            PatternScanner(model_path, signals_dir=temp_dir)
    
    @patch('pattern_scanner.load_trained_model')
    def test_signals_directory_creation(self, mock_load_model, mock_model_package, temp_dir):
        """Test signals directory creation."""
        mock_load_model.return_value = mock_model_package
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        signals_dir = os.path.join(temp_dir, 'new_signals')
        
        scanner = PatternScanner(model_path, signals_dir=signals_dir)
        
        assert os.path.exists(signals_dir)
    
    @patch('pattern_scanner.load_trained_model')
    def test_validate_feature_schema_success(self, mock_load_model, mock_model_package, temp_dir):
        """Test successful feature schema validation."""
        mock_load_model.return_value = mock_model_package
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        scanner = PatternScanner(model_path, signals_dir=temp_dir)
        
        features = {
            'ticker': '0700.HK',
            'sma_5': 150.0,
            'rsi_14': 50.0,
            'volatility': 0.02
        }
        
        assert scanner._validate_feature_schema(features) is True
    
    @patch('pattern_scanner.load_trained_model')
    def test_validate_feature_schema_missing_features(self, mock_load_model, mock_model_package, temp_dir):
        """Test feature schema validation with missing features."""
        mock_load_model.return_value = mock_model_package
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        scanner = PatternScanner(model_path, signals_dir=temp_dir)
        
        features = {
            'ticker': '0700.HK',
            'sma_5': 150.0,
            # missing rsi_14 and volatility
        }
        
        assert scanner._validate_feature_schema(features) is False
    
    @patch('pattern_scanner.load_trained_model')
    def test_extract_sliding_windows(self, mock_load_model, mock_model_package, sample_ticker_data, temp_dir):
        """Test sliding window extraction."""
        mock_load_model.return_value = mock_model_package
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        scanner = PatternScanner(model_path, signals_dir=temp_dir)
        config = ScanningConfig(window_size=30, max_windows_per_ticker=3)
        
        windows = scanner._extract_sliding_windows(sample_ticker_data, '0700.HK', config)
        
        assert len(windows) == 3  # Should generate 3 non-overlapping windows
        
        # Check first window
        window_data, prior_context_data, start_date, end_date = windows[0]
        assert len(window_data) == 30
        assert isinstance(start_date, str)
        assert isinstance(end_date, str)
    
    @patch('pattern_scanner.load_trained_model')
    def test_extract_sliding_windows_insufficient_data(self, mock_load_model, mock_model_package, temp_dir):
        """Test sliding window extraction with insufficient data."""
        mock_load_model.return_value = mock_model_package
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        scanner = PatternScanner(model_path, signals_dir=temp_dir)
        config = ScanningConfig(window_size=30)
        
        # Create very small dataset
        small_data = pd.DataFrame({
            'Open': [100], 'High': [105], 'Low': [95], 'Close': [102], 'Volume': [1000000]
        }, index=[datetime.now()])
        
        windows = scanner._extract_sliding_windows(small_data, '0700.HK', config)
        
        assert len(windows) == 0  # Should return empty list
    
    @patch('pattern_scanner.load_trained_model')
    def test_predict_pattern_probability(self, mock_load_model, mock_model_package, temp_dir):
        """Test pattern probability prediction."""
        mock_load_model.return_value = mock_model_package
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        scanner = PatternScanner(model_path, signals_dir=temp_dir)
        
        features = {
            'ticker': '0700.HK',
            'sma_5': 150.0,
            'rsi_14': 50.0,
            'volatility': 0.02
        }
        
        probability = scanner._predict_pattern_probability(features)
        
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0
        assert probability == 0.7  # Based on mock return value
    
    @patch('pattern_scanner.load_trained_model')
    def test_predict_pattern_probability_missing_features(self, mock_load_model, mock_model_package, temp_dir):
        """Test pattern probability prediction with missing features."""
        mock_load_model.return_value = mock_model_package
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        scanner = PatternScanner(model_path, signals_dir=temp_dir)
        
        features = {
            'ticker': '0700.HK',
            'sma_5': 150.0,
            # missing rsi_14 and volatility
        }
        
        probability = scanner._predict_pattern_probability(features)
        
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0
    
    @patch('pattern_scanner.load_trained_model')
    def test_filter_and_rank_results(self, mock_load_model, mock_model_package, temp_dir):
        """Test results filtering and ranking."""
        mock_load_model.return_value = mock_model_package
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        scanner = PatternScanner(model_path, signals_dir=temp_dir)
        config = ScanningConfig(min_confidence=0.5)
        
        results = [
            {'ticker': '0700.HK', 'confidence_score': 0.8, 'window_start_date': '2023-01-01'},
            {'ticker': '0005.HK', 'confidence_score': 0.3, 'window_start_date': '2023-01-01'},
            {'ticker': '0941.HK', 'confidence_score': 0.9, 'window_start_date': '2023-01-01'},
        ]
        
        filtered_df = scanner._filter_and_rank_results(results, config)
        
        assert len(filtered_df) == 2  # Only scores >= 0.5
        assert filtered_df.iloc[0]['confidence_score'] == 0.9  # Highest first
        assert filtered_df.iloc[1]['confidence_score'] == 0.8
        assert filtered_df.iloc[0]['rank'] == 1
        assert filtered_df.iloc[1]['rank'] == 2
    
    @patch('pattern_scanner.load_trained_model')
    def test_save_results(self, mock_load_model, mock_model_package, temp_dir):
        """Test results saving to CSV."""
        mock_load_model.return_value = mock_model_package
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        scanner = PatternScanner(model_path, signals_dir=temp_dir)
        config = ScanningConfig(save_results=True)
        
        results_df = pd.DataFrame([
            {'ticker': '0700.HK', 'confidence_score': 0.8, 'window_start_date': '2023-01-01'}
        ])
        
        output_path = scanner._save_results(results_df, config)
        
        assert output_path != ""
        assert os.path.exists(output_path)
        assert output_path.endswith('.csv')
        
        # Verify file contents
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 1
        assert loaded_df.iloc[0]['ticker'] == '0700.HK'
    
    @patch('pattern_scanner.load_trained_model')
    def test_save_results_disabled(self, mock_load_model, mock_model_package, temp_dir):
        """Test results saving when disabled."""
        mock_load_model.return_value = mock_model_package
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        scanner = PatternScanner(model_path, signals_dir=temp_dir)
        config = ScanningConfig(save_results=False)
        
        results_df = pd.DataFrame([
            {'ticker': '0700.HK', 'confidence_score': 0.8}
        ])
        
        output_path = scanner._save_results(results_df, config)
        
        assert output_path == ""
    
    @patch('pattern_scanner.load_trained_model')
    def test_generate_scanning_summary(self, mock_load_model, mock_model_package, temp_dir):
        """Test scanning summary generation."""
        mock_load_model.return_value = mock_model_package
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        scanner = PatternScanner(model_path, signals_dir=temp_dir)
        
        tickers = ['0700.HK', '0005.HK']
        results_df = pd.DataFrame([
            {'ticker': '0700.HK', 'confidence_score': 0.8},
            {'ticker': '0005.HK', 'confidence_score': 0.6}
        ])
        scanning_time = 15.5
        
        summary = scanner._generate_scanning_summary(tickers, results_df, scanning_time)
        
        assert summary['total_tickers_scanned'] == 2
        assert summary['matches_found'] == 2
        assert summary['scanning_time_seconds'] == 15.5
        assert summary['average_confidence'] == 0.7
        assert summary['max_confidence'] == 0.8
        assert summary['tickers_with_matches'] == 2


class TestEndToEndScanning:
    """End-to-end integration tests for pattern scanning."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('pattern_scanner.load_trained_model')
    @patch('pattern_scanner._load_cached_data')
    def test_scan_tickers_success(self, mock_load_data, mock_load_model, temp_dir):
        """Test successful end-to-end ticker scanning."""
        # Setup mocks
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        
        mock_model_package = {
            'model': mock_model,
            'scaler': None,
            'feature_names': ['sma_5', 'rsi_14', 'volatility'],
            'config': Mock(model_type='xgboost'),
            'metadata': {'training_date': '2023-01-01'}
        }
        mock_load_model.return_value = mock_model_package
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        }, index=dates)
        
        # Ensure OHLC consistency
        sample_data['High'] = np.maximum(sample_data['High'], 
                                       np.maximum(sample_data['Open'], sample_data['Close']))
        sample_data['Low'] = np.minimum(sample_data['Low'], 
                                      np.minimum(sample_data['Open'], sample_data['Close']))
        
        mock_load_data.return_value = sample_data
        
        # Initialize scanner
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        scanner = PatternScanner(model_path, signals_dir=temp_dir)
        
        # Mock feature extraction
        with patch.object(scanner.feature_extractor, 'extract_features_from_window_data') as mock_extract:
            mock_extract.return_value = {
                'ticker': '0700.HK',
                'sma_5': 150.0,
                'rsi_14': 50.0,
                'volatility': 0.02
            }
            
            # Run scanning
            config = ScanningConfig(min_confidence=0.7, max_windows_per_ticker=2)
            results = scanner.scan_tickers(['0700.HK'], config)
            
            # Verify results
            assert isinstance(results, ScanningResults)
            assert len(results.matches_df) > 0
            assert results.scanning_summary['total_tickers_scanned'] == 1
            assert results.scanning_time > 0
    
    @patch('pattern_scanner.load_trained_model')
    @patch('pattern_scanner._load_cached_data')
    def test_scan_tickers_no_data(self, mock_load_data, mock_load_model, temp_dir):
        """Test scanning with no available data."""
        mock_model_package = {
            'model': Mock(),
            'scaler': None,
            'feature_names': ['sma_5'],
            'config': Mock(model_type='xgboost'),
            'metadata': {'training_date': '2023-01-01'}
        }
        mock_load_model.return_value = mock_model_package
        mock_load_data.return_value = None  # No data available
        
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        scanner = PatternScanner(model_path, signals_dir=temp_dir)
        
        config = ScanningConfig()
        results = scanner.scan_tickers(['0700.HK'], config)
        
        assert len(results.matches_df) == 0
        assert results.scanning_summary['total_tickers_scanned'] == 0


class TestConvenienceFunction:
    """Test the convenience function for pattern scanning."""
    
    @patch('pattern_scanner.PatternScanner')
    @patch('pattern_scanner.get_top_hk_stocks')
    def test_scan_hk_stocks_for_patterns_default_tickers(self, mock_get_stocks, mock_scanner_class):
        """Test convenience function with default ticker list."""
        mock_get_stocks.return_value = ['0700.HK', '0005.HK']
        mock_scanner = Mock()
        mock_scanner.scan_tickers.return_value = Mock()
        mock_scanner_class.return_value = mock_scanner
        
        result = scan_hk_stocks_for_patterns('test_model.pkl')
        
        mock_get_stocks.assert_called_once_with(50)
        mock_scanner.scan_tickers.assert_called_once()
    
    @patch('pattern_scanner.PatternScanner')
    def test_scan_hk_stocks_for_patterns_custom_tickers(self, mock_scanner_class):
        """Test convenience function with custom ticker list."""
        mock_scanner = Mock()
        mock_scanner.scan_tickers.return_value = Mock()
        mock_scanner_class.return_value = mock_scanner
        
        custom_tickers = ['0700.HK', '0388.HK']
        result = scan_hk_stocks_for_patterns('test_model.pkl', ticker_list=custom_tickers)
        
        mock_scanner.scan_tickers.assert_called_once()
        # Verify the config passed to scan_tickers has the right parameters
        call_args = mock_scanner.scan_tickers.call_args
        config = call_args[0][1]  # Second argument should be the config
        assert config.window_size == DEFAULT_WINDOW_SIZE
        assert config.min_confidence == DEFAULT_MIN_CONFIDENCE


if __name__ == '__main__':
    pytest.main([__file__]) 